"""HFC / EBC: training-free detail restoration for one-step editing.

Two modules live here, both operating on the *decoded pixel* tensor so that the
model, the UNet call count of the transport and the Chord field itself stay
untouched:

HFC  — High-Frequency Completion
    * VRR  : VAE Residual Recovery.  The VAE round trip of the source image loses
             roughly 12% of the high-frequency energy, and any latent-space
             background anchoring is capped by that loss.  ``vae_residual``
             re-adds the *known* round-trip error ``I_src - Dec(Enc(I_src))`` on
             the non-edited pixels, so the background returns to the original
             pixels instead of the VAE-smoothed ones.
    * DP   : Detail Probe.  One extra UNet evaluation at a *low* noise level
             re-synthesises the high band on top of the already edited structure.
             Only the high band is taken, and only inside the gated edit region.
    * EM   : Energy Matching.  The probe gain is not hand tuned; it is derived
             from the ratio between the source image's high-band energy and the
             probe's high-band energy inside the gate, which removes the
             over-shoot (halo) failure of unsharp-mask style sharpening.

EBC  — Edge-Boundary Compositing
    * guided-filter alpha : the soft edit mask coming from the latent energy map
             is 8x upsampled and blocky; a guided filter steers it to the source
             image's real edges, so the composite boundary hugs the object
             contour instead of cutting a 8-pixel staircase.
    * seam compensation   : a local DC (colour / exposure) offset is estimated in
             the transition band and removed, which kills the bright seam that a
             plain linear blend leaves behind.

Everything is differentiable-free, needs no training and no extra parameters to
learn; the cost is one extra UNet call (DP) and two VAE decodes (VRR + DP).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# small tensor helpers
# ---------------------------------------------------------------------------


def gaussian_kernel1d(sigma: float, device=None, dtype=torch.float32) -> torch.Tensor:
    radius = max(1, int(round(2.5 * float(sigma))))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(coords ** 2) / (2.0 * float(sigma) ** 2))
    return kernel / kernel.sum()


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur for (B, C, H, W) tensors."""
    if sigma is None or sigma <= 0:
        return x
    channels = x.shape[1]
    k = gaussian_kernel1d(sigma, device=x.device, dtype=x.dtype)
    radius = (k.numel() - 1) // 2
    kernel_h = k.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    kernel_w = k.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    x = F.conv2d(x, kernel_h, padding=(radius, 0), groups=channels)
    x = F.conv2d(x, kernel_w, padding=(0, radius), groups=channels)
    return x


def _box_filter(x: torch.Tensor, radius: int) -> torch.Tensor:
    """Normalised box filter over a (B, 1, H, W) map (replicate padding)."""
    kernel = 2 * radius + 1
    weight = torch.ones(1, 1, kernel, kernel, device=x.device, dtype=x.dtype)
    weight = weight / (kernel * kernel)
    x = F.pad(x, (radius, radius, radius, radius), mode="replicate")
    return F.conv2d(x, weight)


def _luminance(guide: torch.Tensor) -> torch.Tensor:
    if guide.shape[1] == 1:
        return guide
    weights = torch.tensor([0.299, 0.587, 0.114], device=guide.device, dtype=guide.dtype)
    return (guide[:, :3] * weights.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)


def guided_filter(alpha: torch.Tensor, guide: torch.Tensor, radius: int = 8, eps: float = 1e-3) -> torch.Tensor:
    """Edge-aware refinement of a soft mask with the source image as guidance.

    alpha : (B, 1, H, W) soft mask in [0, 1]
    guide : (B, 3, H, W) image in [0, 1]

    The output keeps the *low frequency* of ``alpha`` (which edit region was
    selected) but takes its *edge positions* from ``guide``, i.e. the composite
    boundary snaps onto real image contours.
    """
    if radius <= 0:
        return alpha
    guide_f = guide.to(dtype=torch.float32)
    alpha_f = alpha.to(dtype=torch.float32)
    lum = _luminance(guide_f)

    mean_lum = _box_filter(lum, radius)
    mean_alpha = _box_filter(alpha_f, radius)
    corr_lum = _box_filter(lum * lum, radius)
    corr_lum_alpha = _box_filter(lum * alpha_f, radius)

    var_lum = corr_lum - mean_lum * mean_lum
    cov_lum_alpha = corr_lum_alpha - mean_lum * mean_alpha

    a = cov_lum_alpha / (var_lum + eps)
    b = mean_alpha - a * mean_lum

    mean_a = _box_filter(a, radius)
    mean_b = _box_filter(b, radius)
    refined = mean_a * lum + mean_b
    return refined.clamp(0.0, 1.0).to(dtype=alpha.dtype)


# ---------------------------------------------------------------------------
# HFC — high frequency completion
# ---------------------------------------------------------------------------


def split_bands(img: torch.Tensor, sigma: float) -> Tuple[torch.Tensor, torch.Tensor]:
    low = gaussian_blur(img, sigma)
    return low, img - low


def energy_match_scale(
    high_src: torch.Tensor,
    high_probe: torch.Tensor,
    gate: Optional[torch.Tensor] = None,
    lo: float = 0.5,
    hi: float = 2.5,
    ref: str = "region",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Scalar probe gain such that the injected band matches the source energy.

    ``ref="region"`` measures the source energy inside the gate (the report's
    formulation), ``ref="global"`` uses the whole image, which is more robust
    when the edit introduces a texture that the source region does not have
    (e.g. a beard on smooth skin).
    """
    if ref == "global" or gate is None:
        e_src = high_src.abs().mean()
        e_prb = high_probe.abs().mean()
    else:
        w = gate.clamp_min(0.0)
        denom = w.sum().clamp_min(eps)
        e_src = (high_src.abs() * w).sum() / denom
        e_prb = (high_probe.abs() * w).sum() / denom
    scale = e_src / e_prb.clamp_min(eps)
    return scale.clamp(float(lo), float(hi))


def gate_from_alpha(alpha: torch.Tensor, center: float, width: float) -> torch.Tensor:
    """Smooth step gate around ``center`` with half-width ``width``."""
    if width <= 1e-6:
        return (alpha >= center).to(dtype=alpha.dtype)
    return ((alpha - (center - width)) / (2.0 * width)).clamp(0.0, 1.0)


def detail_completion(
    img_edit: torch.Tensor,
    img_probe: torch.Tensor,
    alpha: torch.Tensor,
    img_src: torch.Tensor,
    cfg: Dict[str, float],
) -> torch.Tensor:
    """Replace the high band of ``img_edit`` by the scaled probe high band.

    Low frequency (structure / colour / the actual edit) always comes from
    ``img_edit``; only the detail band is refreshed, and only where the gate is
    open, so the background is never touched.
    """
    sigma = float(cfg.get("probe_hf_sigma", 1.0))
    gate = gate_from_alpha(alpha, float(cfg.get("probe_gate", 0.5)), float(cfg.get("probe_gate_width", 0.15)))

    low_edit, high_edit = split_bands(img_edit, sigma)
    _, high_probe = split_bands(img_probe, sigma)
    _, high_src = split_bands(img_src, sigma)

    scale = energy_match_scale(
        high_src,
        high_probe,
        gate=gate,
        lo=float(cfg.get("energy_match_lo", 0.5)),
        hi=float(cfg.get("energy_match_hi", 2.5)),
        ref=str(cfg.get("energy_match_ref", "global")),
    )
    blended = low_edit + scale * high_probe
    out = gate * blended + (1.0 - gate) * img_edit
    return out


def vae_residual_restore(
    img_edit: torch.Tensor,
    vae_recon: torch.Tensor,
    img_src: torch.Tensor,
    alpha: torch.Tensor,
    strength: float = 1.0,
) -> torch.Tensor:
    """Add back the known VAE round-trip error on the non-edited region."""
    if strength <= 0.0:
        return img_edit
    residual = img_src - vae_recon
    return img_edit + (1.0 - alpha) * residual * float(strength)


# ---------------------------------------------------------------------------
# EBC — edge / boundary compositing
# ---------------------------------------------------------------------------


def seam_compensation(
    img: torch.Tensor,
    img_src: torch.Tensor,
    alpha: torch.Tensor,
    radius: int = 8,
    strength: float = 0.5,
) -> torch.Tensor:
    """Remove the local DC offset between edit and source inside the seam band.

    ``band`` peaks exactly on the alpha=0.5 contour; the offset is estimated with
    a box filter (i.e. a local mean) and re-injected with the same profile, which
    removes the bright / dark fringe without moving any structure.
    """
    if strength <= 0.0:
        return img
    band = (4.0 * alpha * (1.0 - alpha)).clamp(0.0, 1.0)  # peaks at the boundary
    diff = (img_src - img) * band
    channels = img.shape[1]
    mean_num = F.conv2d(
        F.pad(diff, (radius, radius, radius, radius), mode="replicate"),
        torch.ones(channels, 1, 2 * radius + 1, 2 * radius + 1, device=img.device, dtype=img.dtype),
        groups=channels,
    ) / float((2 * radius + 1) ** 2)
    mean_den = F.conv2d(
        F.pad(band, (radius, radius, radius, radius), mode="replicate"),
        torch.ones(1, 1, 2 * radius + 1, 2 * radius + 1, device=img.device, dtype=img.dtype),
    ) / float((2 * radius + 1) ** 2)
    offset = mean_num / mean_den.clamp_min(1e-4)
    return img + float(strength) * offset * band


def composite_background(
    img_edit: torch.Tensor,
    img_src: torch.Tensor,
    alpha: torch.Tensor,
    beta: float = 0.0,
) -> torch.Tensor:
    """Legacy anchoring: hard-copy the source pixels on the background."""
    weight = alpha + (1.0 - alpha) * float(beta)
    return img_edit * weight + img_src * (1.0 - weight)


# ---------------------------------------------------------------------------
# boundary metrics (used by the evaluation script)
# ---------------------------------------------------------------------------


def boundary_gap(img: torch.Tensor, img_ref: torch.Tensor, alpha: torch.Tensor, band_width: float = 2.0) -> float:
    """Mean absolute colour discontinuity along the composite contour.

    Lower is better: it only looks at the pixels that sit on the mask boundary,
    which is exactly where a linear blend shows its seam.
    """
    band = (4.0 * alpha * (1.0 - alpha)).clamp(0.0, 1.0)
    mask = (band > (1.0 / max(band_width, 1e-6))).to(dtype=img.dtype)
    denom = mask.sum().clamp_min(1.0)
    gap = ((img - img_ref).abs() * mask).sum() / denom
    return float(gap)
