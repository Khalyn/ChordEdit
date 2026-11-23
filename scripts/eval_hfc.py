"""Evaluate the HFC / EBC modules on the bundled example images.

    python scripts/eval_hfc.py                      # 8 images, 5 variants
    python scripts/eval_hfc.py --limit 2            # quick check
    python scripts/eval_hfc.py --only full,baseline
    python scripts/eval_hfc.py --tag v1 --device cuda

Reported metrics (background = mask < 0.5, foreground = mask >= 0.5):

  PSNR_bg / SSIM_bg : fidelity of the non-edited region (higher is better)
  HF_bg/fg/all      : high-band energy of the result divided by that of the
                      source image (1.0 = as crisp as the photo, >1 = over-sharp)
  lap_var_fg        : Laplacian variance inside the edit region (classic acuity)
  seam              : mean |鈭嘔_result - 鈭嘔_source| inside the mask boundary band,
                      i.e. how visible the composite contour is (lower is better)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import hfc as hfc_ops  # noqa: E402
from pipeline_chord import ChordEditPipeline  # noqa: E402

BASE_CONFIG: Dict[str, Any] = {
    "noise_samples": 1,
    "n_steps": 1,
    "t_start": 0.90,
    "t_end": 0.30,
    "t_delta": 0.15,
    "step_scale": 1.0,
    "cleanup": True,
    "use_local_chord": True,
    "local_beta": 0.1,
    "mask_tau_low": 0.1,
    "mask_tau_high": 0.4,
    "mask_sigma": 2.0,
    "use_adaptive_delta": True,
    "adaptive_alpha": 0.5,
    "adaptive_floor": 0.2,
}

# name -> (overrides, nfe)
VARIANTS: Dict[str, Tuple[Dict[str, Any], int]] = {
    # upstream: soft mask + pixel beta 0.1 + no high-frequency completion
    "upstream": (
        {
            "use_alpha_remap": False,
            "pixel_beta": 0.1,
            "use_vae_residual": False,
            "use_detail_probe": False,
            "use_edge_fit": False,
        },
        2,
    ),
    # + near-binary alpha: the edit no longer leaks into the background
    "alpha": (
        {
            "use_alpha_remap": True,
            "pixel_beta": 0.0,
            "use_vae_residual": False,
            "use_detail_probe": False,
            "use_edge_fit": False,
        },
        2,
    ),
    # + edge/boundary compositing (guided-filter alpha + seam DC compensation)
    "ebc": (
        {
            "use_alpha_remap": True,
            "pixel_beta": 0.0,
            "use_vae_residual": False,
            "use_detail_probe": False,
            "use_edge_fit": True,
        },
        2,
    ),
    # + detail probe with energy matching (full method)
    "full": (
        {
            "use_alpha_remap": True,
            "pixel_beta": 0.0,
            "use_vae_residual": False,
            "use_detail_probe": True,
            "use_edge_fit": True,
        },
        3,
    ),
    # control: VAE residual recovery (VRR from the technical report)
    "vrr": (
        {
            "use_alpha_remap": True,
            "pixel_beta": 0.0,
            "use_vae_residual": True,
            "use_detail_probe": False,
            "use_edge_fit": False,
        },
        2,
    ),
    # control: unsharp masking (widely used by others, baseline only)
    "unsharp": (
        {
            "use_alpha_remap": False,
            "pixel_beta": 0.1,
            "use_vae_residual": False,
            "use_detail_probe": False,
            "use_edge_fit": False,
            "use_sharpen_u": True,
        },
        2,
    ),
}


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def _to_tensor(image: Image.Image, size: int = 512) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB").resize((size, size), Image.LANCZOS), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)[None]


def psnr(a: torch.Tensor, b: torch.Tensor, weight: torch.Tensor | None = None) -> float:
    mse = ((a - b) ** 2)
    if weight is not None:
        w = weight.expand_as(mse)
        mse = (mse * w).sum() / w.sum().clamp_min(1.0)
    else:
        mse = mse.mean()
    mse = float(mse)
    if mse <= 1e-12:
        return 99.0
    return 10.0 * np.log10(1.0 / mse)


def _gauss_window(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def ssim(a: torch.Tensor, b: torch.Tensor, weight: torch.Tensor | None = None) -> float:
    win = _gauss_window().to(a.dtype)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    a_gray = a.mean(dim=1, keepdim=True)
    b_gray = b.mean(dim=1, keepdim=True)
    pad = win.shape[-1] // 2
    mu_a = F.conv2d(F.pad(a_gray, (pad,) * 4, mode="replicate"), win)
    mu_b = F.conv2d(F.pad(b_gray, (pad,) * 4, mode="replicate"), win)
    sigma_a = F.conv2d(F.pad(a_gray * a_gray, (pad,) * 4, mode="replicate"), win) - mu_a ** 2
    sigma_b = F.conv2d(F.pad(b_gray * b_gray, (pad,) * 4, mode="replicate"), win) - mu_b ** 2
    sigma_ab = F.conv2d(F.pad(a_gray * b_gray, (pad,) * 4, mode="replicate"), win) - mu_a * mu_b
    ssim_map = ((2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (sigma_a + sigma_b + c2))
    if weight is not None:
        w = weight.expand_as(ssim_map)
        return float((ssim_map * w).sum() / w.sum().clamp_min(1.0))
    return float(ssim_map.mean())


def hf_ratio(img: torch.Tensor, ref: torch.Tensor, weight: torch.Tensor | None = None, sigma: float = 1.0) -> float:
    hi_img = img - hfc_ops.gaussian_blur(img, sigma)
    hi_ref = ref - hfc_ops.gaussian_blur(ref, sigma)
    if weight is not None:
        w = weight.expand_as(hi_ref)
        e_img = (hi_img.abs() * w).sum() / w.sum().clamp_min(1.0)
        e_ref = (hi_ref.abs() * w).sum() / w.sum().clamp_min(1.0)
    else:
        e_img, e_ref = hi_img.abs().mean(), hi_ref.abs().mean()
    return float(e_img / e_ref.clamp_min(1e-8))


def lap_var(img: torch.Tensor, weight: torch.Tensor | None = None) -> float:
    k = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=img.dtype).view(1, 1, 3, 3)
    gray = img.mean(dim=1, keepdim=True)
    lap = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="replicate"), k)
    lap = lap * weight if weight is not None else lap
    return float(lap.var(unbiased=False))


def seam_metric(img: torch.Tensor, ref: torch.Tensor, alpha: torch.Tensor) -> float:
    """Mean |鈭噄mg - 鈭噐ef| inside the composite boundary band (lower = cleaner)."""
    band = (4.0 * alpha * (1.0 - alpha)).clamp(0, 1)  # peaks at alpha = 0.5
    mask = (band > 0.6).to(dtype=img.dtype)
    if float(mask.sum()) < 16.0:
        return 0.0

    def grad(x: torch.Tensor) -> torch.Tensor:
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        gx = F.pad(dx, (0, 1, 0, 0))
        gy = F.pad(dy, (0, 0, 0, 1))
        return torch.cat([gx, gy], dim=1)

    diff = (grad(img) - grad(ref)).abs().mean(dim=1, keepdim=True) * mask
    return float(diff.sum() / mask.sum().clamp_min(1.0))


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def load_pipeline(device: str | None, precision: str) -> ChordEditPipeline:
    root = ROOT / "models" / "sd-turbo"
    paths = {
        k: str((root / s).resolve())
        for k, s in {
            "unet_path": "unet",
            "scheduler_path": "scheduler",
            "text_encoder_path": "text_encoder",
            "tokenizer_path": "tokenizer",
            "vae_path": "vae",
        }.items()
    }
    dtype = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}[precision]
    return ChordEditPipeline.from_local_weights(
        component_paths=paths,
        default_edit_config=BASE_CONFIG,
        device=device,
        torch_dtype=dtype,
        compute_dtype=dtype,
    )


def read_prompts(image_path: Path) -> Tuple[str, str]:
    with (image_path.parent / "meta.jsonl").open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                return rec["original_prompt"], rec["edited_prompt"]
    raise ValueError("empty meta.jsonl for " + str(image_path))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="upstream,alpha,ebc,full,vrr,unsharp")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--precision", default="fp16", choices=["fp16", "fp32", "bf16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default="run")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    names = [n.strip() for n in args.only.split(",") if n.strip()]
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        raise SystemExit("unknown variant(s): " + ", ".join(unknown))

    folders = sorted(p for p in (ROOT / "images").iterdir() if p.is_dir())[: args.limit]
    pipe = load_pipeline(args.device, args.precision)

    out_root = ROOT / "outputs" / "eval_hfc" / args.tag
    out_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    # warm up once so the timing column is meaningful
    first = folders[0] / "i.jpg"
    src_p, tgt_p = read_prompts(first)
    pipe(image=Image.open(first).convert("RGB"), source_prompt=src_p, target_prompt=tgt_p, seed=args.seed)

    # upstream 先跑，它给出的软 mask 作为所有变体共用的背景/前景分区
    order = (["upstream"] if "upstream" in names else []) + [n for n in names if n != "upstream"]
    ref_alpha: Dict[str, torch.Tensor] = {}

    for folder in folders:
        image_path = folder / "i.jpg"
        src_p, tgt_p = read_prompts(image_path)
        src_pil = Image.open(image_path).convert("RGB")

        for name in order:
            overrides, nfe = VARIANTS[name]
            cfg = dict(BASE_CONFIG)
            cfg.update(overrides)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.time()
            result = pipe(
                image=src_pil,
                source_prompt=src_p,
                target_prompt=tgt_p,
                edit_config=cfg,
                seed=args.seed,
                return_mask=True,
                return_source=True,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.time() - t0

            out_pil = result.images[0]
            if not args.no_save:
                (out_root / name).mkdir(parents=True, exist_ok=True)
                out_pil.save(out_root / name / (folder.name + ".png"))

            out_t = _to_tensor(out_pil)
            # reference must be the pipeline-internal source tensor (not a PIL resize)
            src_t = result.sources.float()
            if result.alphas is not None:
                alpha = result.alphas.float().clamp(0, 1)
            else:
                alpha = torch.ones(1, 1, 512, 512)
            if name == "upstream" or folder.name not in ref_alpha:
                ref_alpha[folder.name] = alpha
            alpha_ref = ref_alpha[folder.name]
            # 三分区：严格背景 / 严格前景 / 边界带（过渡带单独由 seam 指标衡量）
            bg_strict = (alpha_ref < 0.1).to(dtype=torch.float32)
            fg_strict = (alpha_ref > 0.9).to(dtype=torch.float32)
            bg_cov = float(bg_strict.mean())
            fg_cov = float(fg_strict.mean())
            # 像 006（秋→冬）这种全图级编辑没有"干净背景"，背景指标无意义
            has_bg = bg_cov >= 0.05
            has_fg = fg_cov >= 0.05
            bg_w = bg_strict if has_bg else None
            fg_w = fg_strict if has_fg else None

            nan = float("nan")
            rows.append(
                {
                    "image": folder.name,
                    "variant": name,
                    "nfe": nfe,
                    "time_s": round(elapsed, 3),
                    "psnr_bg": round(psnr(out_t, src_t, bg_w), 3) if has_bg else nan,
                    "psnr_all": round(psnr(out_t, src_t), 3),
                    "ssim_bg": round(ssim(out_t, src_t, bg_w), 4) if has_bg else nan,
                    "hf_bg": round(hf_ratio(out_t, src_t, bg_w), 4) if has_bg else nan,
                    "hf_fg": round(hf_ratio(out_t, src_t, fg_w), 4) if has_fg else nan,
                    "hf_all": round(hf_ratio(out_t, src_t), 4),
                    "lap_var_fg": round(lap_var(out_t, fg_w), 5) if has_fg else nan,
                    "seam": round(seam_metric(out_t, src_t, alpha), 5),
                    "bg_cov_strict": round(bg_cov, 4),
                    "fg_cov_strict": round(fg_cov, 4),
                }
            )
            print(
                "%-6s %-8s psnr_bg %6.2f  ssim_bg %.4f  hf_bg %.3f  hf_fg %.3f  seam %.5f  %.2fs"
                % (
                    folder.name,
                    name,
                    rows[-1]["psnr_bg"],
                    rows[-1]["ssim_bg"],
                    rows[-1]["hf_bg"],
                    rows[-1]["hf_fg"],
                    rows[-1]["seam"],
                    elapsed,
                ),
                flush=True,
            )

    csv_path = out_root / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # average table
    print("\n=== average over %d image(s) ===" % len(folders))
    header = "variant   nfe  psnr_bg  ssim_bg  hf_bg  hf_fg  hf_all  lap_var  seam   time"
    print(header)
    summary: List[Dict[str, Any]] = []
    for name in names:
        sub = [r for r in rows if r["variant"] == name]
        if not sub:
            continue
        avg = {
            k: float(np.nanmean([r[k] for r in sub]))
            for k in ("psnr_bg", "ssim_bg", "hf_bg", "hf_fg", "hf_all", "lap_var_fg", "seam", "time_s")
        }
        summary.append({"variant": name, "nfe": sub[0]["nfe"], **{k: round(v, 4) for k, v in avg.items()}})
        print(
            "%-9s %3d  %7.2f  %.4f  %.3f  %.3f  %.3f  %.5f  %.5f  %.2fs"
            % (
                name,
                sub[0]["nfe"],
                avg["psnr_bg"],
                avg["ssim_bg"],
                avg["hf_bg"],
                avg["hf_fg"],
                avg["hf_all"],
                avg["lap_var_fg"],
                avg["seam"],
                avg["time_s"],
            )
        )

    with (out_root / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    md = ["| variant | NFE | PSNR_bg鈫?| SSIM_bg鈫?| HF_bg鈫? | HF_fg鈫? | HF_all鈫? | lap_var_fg鈫?| seam鈫?| time |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for s in summary:
        md.append(
            "| %s | %d | %.2f | %.4f | %.3f | %.3f | %.3f | %.5f | %.5f | %.2fs |"
            % (s["variant"], s["nfe"], s["psnr_bg"], s["ssim_bg"], s["hf_bg"], s["hf_fg"], s["hf_all"], s["lap_var_fg"], s["seam"], s["time_s"])
        )
    (out_root / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\nwrote", csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

