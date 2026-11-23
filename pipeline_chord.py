from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageFilter

import hfc as hfc_ops
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.utils import BaseOutput
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPTextModel

DEFAULT_SEED = 42
DEFAULT_COMPUTE_DTYPE = torch.float32
DEFAULT_SAFETY_CHECKER_ID = "CompVis/stable-diffusion-safety-checker"

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline output container
# ---------------------------------------------------------------------------
@dataclass
class ChordEditPipelineOutput(BaseOutput):
    images: List[Image.Image] | torch.Tensor
    latents: torch.Tensor
    masks: Optional[torch.Tensor] = None  # ← 新增，仅 return_mask=True 时非 None
    alphas: Optional[torch.Tensor] = None  # ← 实际用于融合的 512x512 soft mask
    sources: Optional[torch.Tensor] = None  # ← 内部对齐后的源图 (B,3,512,512) ∈ [0,1]


class _CenterSquareCropTransform:
    """Center-crop the shorter image dimension before resizing."""

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        if width == height:
            return image
        target = min(width, height)
        try:
            resample = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
        except AttributeError:  # pragma: no cover
            resample = Image.LANCZOS
        return ImageOps.fit(
            image,
            (target, target),
            method=resample,
            centering=(0.5, 0.5),
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"{self.__class__.__name__}()"


# ---------------------------------------------------------------------------
# ChordEdit Pipeline
# ---------------------------------------------------------------------------
class ChordEditPipeline(DiffusionPipeline):
    """Standalone pipeline that wires up diffusers modules with the Chord editor."""

    def __init__(
            self,
            unet: UNet2DConditionModel,
            scheduler: DDPMScheduler,
            vae: AutoencoderKL,
            tokenizer,
            text_encoder: CLIPTextModel,
            default_edit_config: Optional[Dict[str, Any]] = None,
            image_size: int = 512,
            device: Optional[str | torch.device] = None,
            compute_dtype: torch.dtype = DEFAULT_COMPUTE_DTYPE,
            use_attention_mask: bool = False,
            use_center_crop: bool = True,
            use_safety_checker: bool = False,
            safety_checker_id: Optional[str] = DEFAULT_SAFETY_CHECKER_ID,
    ) -> None:
        super().__init__()
        self.register_modules(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
        )
        self._device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._compute_dtype = compute_dtype
        self._use_attention_mask = bool(use_attention_mask)
        self.to(self._device)
        self._set_compute_precision()

        self.default_edit_config = default_edit_config or {}
        self.image_size = int(image_size)
        self._use_center_crop = bool(use_center_crop)
        self._vae_transform = self._build_vae_transform()
        self.unet.eval()
        self.vae.eval()
        self.text_encoder.eval()
        self._max_unet_timestep = self.scheduler.config.num_train_timesteps - 1
        self._use_safety_checker = bool(use_safety_checker)
        self._safety_checker_id = safety_checker_id
        self._safety_checker: Optional[StableDiffusionSafetyChecker] = None
        self._safety_feature_extractor: Optional[CLIPImageProcessor] = None
        if self._use_safety_checker:
            self._init_safety_checker()

    def _set_compute_precision(self) -> None:
        modules = (self.unet, self.vae, self.text_encoder)
        for module in modules:
            if module is not None:
                module.to(device=self._device, dtype=self._compute_dtype)

    def _init_safety_checker(self) -> None:
        if not self._safety_checker_id:
            LOGGER.warning("Safety checker requested but no identifier provided; disabling safety checks.")
            self._use_safety_checker = False
            return
        try:
            self._safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                self._safety_checker_id,
                torch_dtype=self._compute_dtype,
            ).to(self._device)
            self._safety_feature_extractor = CLIPImageProcessor.from_pretrained(self._safety_checker_id)
        except Exception as exc:  # pragma: no cover - runtime dependency
            LOGGER.warning("Failed to initialize safety checker (%s). Safety checks disabled.", exc)
            self._safety_checker = None
            self._safety_feature_extractor = None
            self._use_safety_checker = False

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def from_local_weights(
            cls,
            component_paths: Dict[str, str],
            *,
            default_edit_config: Optional[Dict[str, Any]] = None,
            device: Optional[str | torch.device] = None,
            torch_dtype: torch.dtype = torch.float32,
            image_size: int = 512,
            use_center_crop: bool = True,
            compute_dtype: torch.dtype = DEFAULT_COMPUTE_DTYPE,
            use_attention_mask: bool = False,
            use_safety_checker: bool = False,
            safety_checker_id: Optional[str] = DEFAULT_SAFETY_CHECKER_ID,
    ) -> "ChordEditPipeline":
        """Instantiate the pipeline from individual component checkpoints."""

        unet = UNet2DConditionModel.from_pretrained(
            component_paths["unet_path"],
            torch_dtype=torch_dtype,
        )
        scheduler = DDPMScheduler.from_pretrained(component_paths["scheduler_path"])
        vae = AutoencoderKL.from_pretrained(component_paths["vae_path"], torch_dtype=torch_dtype)
        tokenizer = AutoTokenizer.from_pretrained(component_paths["tokenizer_path"])
        text_encoder = CLIPTextModel.from_pretrained(
            component_paths["text_encoder_path"],
            torch_dtype=torch_dtype,
        )
        return cls(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            default_edit_config=default_edit_config,
            image_size=image_size,
            device=device,
            compute_dtype=compute_dtype,
            use_attention_mask=use_attention_mask,
            use_center_crop=use_center_crop,
            use_safety_checker=use_safety_checker,
            safety_checker_id=safety_checker_id,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def __call__(
            self,
            image: Image.Image | torch.Tensor,
            *,
            source_prompt: str,
            target_prompt: str,
            edit_config: Optional[Dict[str, Any]] = None,
            seed: Optional[int] = None,
            output_type: str = "pil",
            return_mask: bool = False,
            return_source: bool = False,
    ) -> ChordEditPipelineOutput:
        """Run ChordEdit once on a single image."""

        cfg = dict(self.default_edit_config)
        if edit_config:
            cfg.update(edit_config)
        required_keys = ["noise_samples", "n_steps", "t_start", "t_end", "t_delta", "step_scale"]
        missing = [k for k in required_keys if k not in cfg]
        if missing:
            raise ValueError(f"edit_config is missing required keys: {missing}")

        pixel_values = self._prepare_image_tensor(image)
        latents = self._encode_image_to_latent(pixel_values)
        src_embed = self.encode_prompt([source_prompt])
        tgt_embed = self.encode_prompt([target_prompt])

        output_latents: List[torch.Tensor] = []

        edit_params = self._prepare_edit_params(cfg)
        seed_value = int(seed) if seed is not None else DEFAULT_SEED

        noise_list = self._prepare_noise_list(
            latents=latents,
            seed_value=seed_value,
            num_noises=edit_params["noise_samples"],
        )

        x0_pred, final_mask = self._run_edit(
            x_src=latents,
            src_embed=src_embed,
            edit_embed=tgt_embed,
            noise=noise_list,
            params=edit_params,
        )

        decoded = self._decode_latent_to_image(x0_pred)
        decoded, _ = self._apply_safety_checker(decoded)

        # =======================================================
        # HFC / EBC：像素级高频补全 + 边缘轮廓贴合
        #   （VRR 残差回填 / DP 细节探针 + 能量匹配 / 引导滤波 alpha + 接缝补偿）
        # =======================================================
        images_tensor, alpha_tensor = self._enhance_pixels(
            decoded=decoded,
            x0_pred=x0_pred,
            x_src=latents,
            src_pixels=pixel_values,
            edit_embed=tgt_embed,
            noise=noise_list,
            mask=final_mask,
            params=edit_params,
        )

        output_latents.append(x0_pred.detach().cpu())
        latents_tensor = torch.cat(output_latents, dim=0)
        images = self._tensor_to_pil(images_tensor) if output_type == "pil" else images_tensor

        # ---- 像素空间的高频清晰度增强（团队既有模块，保持原样，与本分支新增模块互不耦合）----
        if isinstance(images, list) and edit_params.get("use_sharpen_u", False):
            sharpened_images = []
            percent = int(edit_params.get("sharpen_gamma", 0.5) * 100)
            if percent > 0:
                for img in images:
                    sharpened = img.filter(ImageFilter.UnsharpMask(
                        radius=2.0,
                        percent=percent,
                        threshold=3
                    ))
                    sharpened_images.append(sharpened)
                images = sharpened_images
        # =======================================================
        return ChordEditPipelineOutput(
            images=images,
            latents=latents_tensor,
            masks=final_mask.detach().cpu() if (return_mask and final_mask is not None) else None,
            alphas=alpha_tensor.detach().cpu() if return_mask else None,
            sources=(pixel_values.to(dtype=torch.float32) * 0.5 + 0.5).clamp(0.0, 1.0).detach().cpu()
            if return_source
            else None,
        )

    def encode_prompt(self, prompts: Sequence[str]) -> torch.Tensor:
        """Public helper mirroring diffusers pipelines for text encoding."""
        return self._encode_text(prompts)

    # ------------------------------------------------------------------ #
    # HFC / EBC : pixel-space detail restoration
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _enhance_pixels(
            self,
            *,
            decoded: torch.Tensor,
            x0_pred: torch.Tensor,
            x_src: torch.Tensor,
            src_pixels: torch.Tensor,
            edit_embed: torch.Tensor,
            noise: List[torch.Tensor],
            mask: Optional[torch.Tensor],
            params: Dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply VRR / DP / energy matching / edge fitting on the decoded image.

        All arithmetic happens in float32 pixels in [0, 1]; the edit structure is
        never modified -- only the detail band and the composite boundary are.

        Returns ``(image, alpha)`` where ``alpha`` is the 512x512 soft mask that
        was actually used for compositing (useful for metrics and debugging).
        """
        device = decoded.device
        out_dtype = decoded.dtype
        batch, _, height, width = decoded.shape

        img_src = (src_pixels.to(device=device, dtype=torch.float32) * 0.5 + 0.5).clamp(0.0, 1.0)
        img = decoded.to(dtype=torch.float32)

        use_local = bool(params.get("use_local_chord", False)) and mask is not None
        if use_local:
            alpha = F.interpolate(
                mask.to(device=device, dtype=torch.float32),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).clamp(0.0, 1.0)
            # 能量 mask 为了不切断编辑被 gamma 抬升过（背景里 α 常有 0.2~0.4），
            # 直接拿来做融合会把编辑漏进背景。近二值化统一在 _run_edit 里做一次，
            # 这里只负责上采样与轮廓对齐（否则会重复 remap）。
            if params.get("use_edge_fit", False):
                # 让边界贴着原图的真实轮廓走，而不是 8x 上采样出的方块边
                alpha = hfc_ops.guided_filter(
                    alpha, img_src, radius=int(params.get("edge_radius", 8)), eps=float(params.get("edge_eps", 1e-3))
                )
                alpha = hfc_ops.gaussian_blur(alpha, sigma=0.5)
            else:
                feather = float(params.get("mask_feather_px", 3.0))
                if feather > 0:
                    alpha = hfc_ops.gaussian_blur(alpha, sigma=feather)
        else:
            alpha = torch.ones((batch, 1, height, width), device=device, dtype=torch.float32)

        # ---- (1) DP：低噪声细节探针，只替换被门控选中的高频带 ----
        if params.get("use_detail_probe", False) and float(params.get("probe_t", 0.0)) > 0.0:
            probe_steps = self._time_to_index(batch, float(params["probe_t"]), device=device)
            x_probe = self._pred_x0(x0_pred, probe_steps, edit_embed, noise[0])
            img_probe = self._decode_latent_to_image(x_probe).to(device=device, dtype=torch.float32)
            img = hfc_ops.detail_completion(img, img_probe, alpha, img_src, params)

        # ---- (2) VRR：把 VAE 往返的已知损失补回非编辑区 ----
        if use_local:
            if params.get("use_vae_residual", False) and float(params.get("vae_residual_alpha", 1.0)) > 0.0:
                vae_recon = self._decode_latent_to_image(x_src).to(device=device, dtype=torch.float32)
                img = hfc_ops.vae_residual_restore(
                    img, vae_recon, img_src, alpha, strength=float(params.get("vae_residual_alpha", 1.0))
                )
            else:
                img = hfc_ops.composite_background(
                    img, img_src, alpha, beta=float(params.get("pixel_beta", 0.0))
                )

            # ---- (3) EBC：消掉羽化带上的色差/曝光差，去掉亮边 ----
            strength = float(params.get("edge_dc_strength", 0.0)) if params.get("use_edge_fit", False) else 0.0
            if strength > 0.0:
                img = hfc_ops.seam_compensation(
                    img,
                    img_src,
                    alpha,
                    radius=max(4, int(params.get("edge_radius", 8))),
                    strength=strength,
                )

        return img.clamp(0.0, 1.0).to(dtype=out_dtype), alpha

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _prepare_image_tensor(self, image: Image.Image | torch.Tensor) -> torch.Tensor:
        if isinstance(image, Image.Image):
            vae_tensor = self._vae_transform(image)
        elif torch.is_tensor(image):
            tensor = image.float()
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            if tensor.max() > 1.0:
                tensor = tensor / 255.0
            tensor = tensor * 2.0 - 1.0
            vae_tensor = tensor
        else:
            raise TypeError("image must be a PIL.Image or a torch.Tensor.")

        if vae_tensor.ndim == 3:
            vae_tensor = vae_tensor.unsqueeze(0)
        if self._use_center_crop and vae_tensor.ndim == 4:
            _, _, height, width = vae_tensor.shape
            if height != width:
                side = min(height, width)
                top = (height - side) // 2
                left = (width - side) // 2
                vae_tensor = vae_tensor[:, :, top: top + side, left: left + side]
        return vae_tensor.to(device=self._device, dtype=self._compute_dtype)

    def _encode_image_to_latent(self, pixel_values: torch.Tensor) -> torch.Tensor:
        scaling_factor = getattr(self.vae.config, "scaling_factor", 1.0)
        pixel_values = pixel_values.to(device=self._device, dtype=self._compute_dtype)
        latents = self.vae.encode(pixel_values).latent_dist.mode()
        latents = latents * scaling_factor
        return latents.to(device=self._device, dtype=self._compute_dtype)

    def _decode_latent_to_image(self, latents: torch.Tensor) -> torch.Tensor:
        scaling_factor = getattr(self.vae.config, "scaling_factor", 1.0)
        latents = latents.to(device=self._device, dtype=self._compute_dtype)
        decoded = self.vae.decode(latents / scaling_factor).sample
        decoded = (decoded.clamp(-1.0, 1.0) + 1.0) / 2.0
        return decoded.to(dtype=self._compute_dtype)

    def _apply_safety_checker(self, images: torch.Tensor) -> tuple[torch.Tensor, List[bool]]:
        batch = images.shape[0]
        if (
                not self._use_safety_checker
                or self._safety_checker is None
                or self._safety_feature_extractor is None
                or batch == 0
        ):
            return images, [False] * batch

        images_clamped = images.detach().clamp(0.0, 1.0)
        pil_images = self._tensor_to_pil(images_clamped)
        try:
            clip_input = self._safety_feature_extractor(images=pil_images, return_tensors="pt").to(self._device)
            images_np = np.stack([np.array(img).astype(np.float32) / 255.0 for img in pil_images], axis=0)
            images_np = images_np * 2.0 - 1.0
            _, has_nsfw_concept = self._safety_checker(
                images=images_np,
                clip_input=clip_input.pixel_values.to(self._device),
            )
        except Exception as exc:  # pragma: no cover - runtime guard
            LOGGER.warning("Safety checker failed (%s). Skipping safety checks.", exc)
            return images, [False] * batch

        if isinstance(has_nsfw_concept, torch.Tensor):
            has_nsfw = has_nsfw_concept.detach().cpu().to(dtype=torch.bool).tolist()
        else:
            has_nsfw = [bool(flag) for flag in has_nsfw_concept]

        if any(has_nsfw):
            for idx, flagged in enumerate(has_nsfw):
                if flagged:
                    images[idx] = torch.zeros_like(images[idx])
        return images, has_nsfw

    def _encode_text(self, prompts: Sequence[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            list(prompts),
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        input_ids = inputs.input_ids.to(self._device)
        attn_mask = inputs.attention_mask.to(self._device) if self._use_attention_mask else None
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attn_mask)
        if hasattr(outputs, "last_hidden_state"):
            hidden = outputs.last_hidden_state
        else:
            hidden = outputs[0]
        return hidden.to(device=self._device, dtype=self._compute_dtype)

    def _tensor_to_pil(self, tensor: torch.Tensor) -> List[Image.Image]:
        tensor = tensor.detach().cpu().clamp(0.0, 1.0)
        to_pil = transforms.ToPILImage()
        return [to_pil(sample) for sample in tensor]

    def _build_vae_transform(self) -> transforms.Compose:
        """Create image->latent preprocessing transform."""
        ops: List[Any] = []
        if self._use_center_crop:
            ops.append(_CenterSquareCropTransform())
            resize_interp = InterpolationMode.LANCZOS
        else:
            resize_interp = InterpolationMode.BILINEAR
        ops.append(
            transforms.Resize(
                (self.image_size, self.image_size),
                interpolation=resize_interp,
            )
        )
        ops.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        return transforms.Compose(ops)

    def _prepare_edit_params(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        params = dict(cfg)
        params["noise_samples"] = int(max(1, params["noise_samples"]))
        params["n_steps"] = int(max(1, params["n_steps"]))
        params["t_start"] = float(max(0.0, min(1.0, params["t_start"])))
        params["t_end"] = float(max(0.0, min(params["t_start"], params["t_end"])))
        # ---- 时间自适应（能量驱动 δ）----
        params["use_adaptive_delta"] = bool(params.get("use_adaptive_delta", False))
        params["adaptive_alpha"] = float(
            max(0.0, min(1.0, float(params.get("adaptive_alpha", 1.0))))
        )
        params["adaptive_floor"] = float(
            max(0.0, min(1.0, float(params.get("adaptive_floor", 0.2))))
        )
        t_delta = float(max(0.0, min(1.0, params["t_delta"])))
        if t_delta >= params["t_start"]:
            safe_max = max(1, self._max_unet_timestep)
            t_delta = max(0.0, params["t_start"] - 1.0 / safe_max)
        params["t_delta"] = t_delta
        params["step_scale"] = float(params["step_scale"])
        params["cleanup"] = bool(params.get("cleanup", False))
        # ---- 新增：局部 Chord 参数 ----
        params["use_local_chord"] = bool(params.get("use_local_chord", False))
        mode = str(params.get("local_mode", "B")).upper()
        params["local_mode"] = mode if mode in ("A", "B") else "B"
        params["local_beta"] = float(
            max(0.0, min(1.0, float(params.get("local_beta", 0.2))))
        )
        tau_lo = float(max(0.0, float(params.get("mask_tau_low", 0.1))))
        tau_hi = float(max(tau_lo + 1e-3, float(params.get("mask_tau_high", 0.4))))
        params["mask_tau_low"] = tau_lo
        params["mask_tau_high"] = tau_hi
        params["mask_sigma"] = float(max(0.0, float(params.get("mask_sigma", 2.0))))

        # ---- 新增：高频增强/清晰度模块 ----
        params["use_sharpen_u"] = bool(params.get("use_sharpen_u", False))
        params["sharpen_gamma"] = float(max(0.0, float(params.get("sharpen_gamma", 0.5))))

        params["use_latent_sharp"] = bool(params.get("use_latent_sharp", False))
        params["latent_sharp_bg"] = float(max(0.0, float(params.get("latent_sharp_bg", 0.5))))
        params["latent_sharp_fg"] = float(max(0.0, float(params.get("latent_sharp_fg", 0.2))))
        # -------------------------------------

        # ---- HFC：高频补全（VRR + DP + 能量匹配）----
        params["use_vae_residual"] = bool(params.get("use_vae_residual", True))
        params["vae_residual_alpha"] = float(max(0.0, float(params.get("vae_residual_alpha", 1.0))))
        # 像素阶段背景回填比例（与 latent 阶段的 local_beta 解耦）：
        # 0 = 背景完全取原图像素（保真最高），1 = 完全用生成结果
        params["pixel_beta"] = float(max(0.0, min(1.0, float(params.get("pixel_beta", 0.0)))))
        params["use_detail_probe"] = bool(params.get("use_detail_probe", True))
        params["probe_t"] = float(max(0.0, min(1.0, float(params.get("probe_t", 0.10)))))
        params["probe_hf_sigma"] = float(max(0.1, float(params.get("probe_hf_sigma", 1.0))))
        params["probe_gate"] = float(max(0.0, min(1.0, float(params.get("probe_gate", 0.5)))))
        params["probe_gate_width"] = float(max(0.0, min(0.5, float(params.get("probe_gate_width", 0.15)))))
        # 实测：DP 是"频带替换"而非"增益放大"，标量能量匹配会把新增纹理（胡子、毛皮）
        # 拉回原区域的光滑水平，反而变糊（HF_fg 1.14 -> 0.94）。默认关闭，作为消融保留。
        params["use_energy_match"] = bool(params.get("use_energy_match", False))
        if params["use_energy_match"]:
            params["energy_match_lo"] = float(max(0.0, float(params.get("energy_match_lo", 0.5))))
            params["energy_match_hi"] = float(max(0.0, float(params.get("energy_match_hi", 2.5))))
        else:
            params["energy_match_lo"] = 1.0
            params["energy_match_hi"] = 1.0
        ref = str(params.get("energy_match_ref", "global")).lower()
        params["energy_match_ref"] = ref if ref in ("region", "global") else "global"

        # ---- EBC：边缘轮廓贴合（引导滤波 mask + 接缝色差补偿）----
        params["use_edge_fit"] = bool(params.get("use_edge_fit", True))
        params["edge_radius"] = int(max(0, int(params.get("edge_radius", 8))))
        params["edge_eps"] = float(max(1e-6, float(params.get("edge_eps", 1e-3))))
        params["edge_dc_strength"] = float(max(0.0, min(1.0, float(params.get("edge_dc_strength", 0.5)))))
        params["mask_feather_px"] = float(max(0.0, float(params.get("mask_feather_px", 3.0))))
        params["use_alpha_remap"] = bool(params.get("use_alpha_remap", True))
        params["use_latent_alpha_remap"] = bool(params.get("use_latent_alpha_remap", True))
        params["alpha_remap_lo"] = float(max(0.0, min(1.0, float(params.get("alpha_remap_lo", 0.30)))))
        params["alpha_remap_width"] = float(max(1e-3, float(params.get("alpha_remap_width", 0.30))))
        # -------------------------------------

        return params

    def _prepare_noise_list(
            self,
            latents: torch.Tensor,
            seed_value: int,
            num_noises: int,
    ) -> List[torch.Tensor]:
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_value)
        noise_list = [
            torch.randn_like(latents, device=latents.device, dtype=self._compute_dtype)
            for _ in range(num_noises)
        ]
        return noise_list

    def _gaussian_blur(self, mask: torch.Tensor, sigma: float) -> torch.Tensor:
        """Separable-free 2D Gaussian blur on a (B, 1, H, W) mask."""
        if sigma <= 0:
            return mask
        import torch.nn.functional as F

        radius = max(1, int(round(2.5 * sigma)))  # 覆盖 ±2.5σ
        kernel_size = 2 * radius + 1
        coords = torch.arange(kernel_size, device=mask.device, dtype=torch.float32) - radius
        g = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
        g = g / g.sum()
        kernel2d = (g[:, None] * g[None, :]).to(dtype=mask.dtype)
        channels = mask.shape[1]
        kernel2d = kernel2d.expand(channels, 1, kernel_size, kernel_size).contiguous()
        return F.conv2d(mask, kernel2d, padding=radius, groups=channels)

    def _energy_mask(
            self,
            u_hat: torch.Tensor,
            tau_low: float = 0.1,
            tau_high: float = 0.4,
            sigma: float = 2.0,
            gamma: float = 0.5,  # ← 新增：mask 增益（<1 抬高 M）
            dilate: int = 2,  # ← 新增：最大池化膨胀核半径
            quantile: float = 0.95,  # ← 新增：归一化分位（0.99 → 0.95）
    ) -> torch.Tensor:
        """Derive a soft spatial mask from the control-field energy map.

        u_hat: (B, C, H, W) — 单步编辑方向
        返回:  (B, 1, H, W) ∈ [0, 1]
        """
        B = u_hat.shape[0]
        energy = (u_hat ** 2).sum(dim=1, keepdim=True)  # (B,1,H,W)
        flat = energy.reshape(B, -1).float()  # quantile 需 float32
        if flat.shape[1] >= 2:
            scale = torch.quantile(flat, 0.99, dim=1)
        else:
            scale = flat.max(dim=1).values
        scale = scale.clamp_min(1e-8).view(B, 1, 1, 1).to(dtype=u_hat.dtype)
        e_norm = energy / scale
        denom = max(tau_high - tau_low, 1e-6)
        mask = torch.clamp((e_norm - tau_low) / denom, 0.0, 1.0)
        # ① 膨胀：把强编辑点的邻域也纳入（M>0 的区域向外扩 dilate 像素）
        if dilate > 0:
            mask = F.max_pool2d(mask, kernel_size=2 * dilate + 1, stride=1, padding=dilate)

        # ② gamma 增益：M ← M^γ，γ<1 时中低值被显著抬高
        if gamma != 1.0:
            mask = mask.clamp_min(1e-8) ** gamma

        # ③ 高斯羽化（接缝 + 理论需要）
        mask = self._gaussian_blur(mask, sigma=sigma)
        return mask

    def _time_to_index(self, batch: int, t_scalar: float, device, dtype=torch.long):
        idx = round(self._max_unet_timestep * float(t_scalar))
        idx = max(0, min(self._max_unet_timestep, idx))
        return torch.full((batch,), idx, device=device, dtype=dtype)

    def _get_alpha_sigma(self, tensor: torch.Tensor, timesteps: torch.Tensor):
        alphas_cumprod = self.scheduler.alphas_cumprod.to(dtype=torch.float32, device=tensor.device)
        alpha_t = alphas_cumprod[timesteps].sqrt().view(-1, 1, 1, 1)
        sigma_t = (1 - alphas_cumprod[timesteps]).sqrt().view(-1, 1, 1, 1)
        alpha_t = alpha_t.to(dtype=tensor.dtype, device=tensor.device)
        sigma_t = sigma_t.to(dtype=tensor.dtype, device=tensor.device)
        eps = torch.finfo(alpha_t.dtype).eps
        alpha_t = alpha_t.clamp(min=eps)
        return alpha_t, sigma_t

    def _pred_x0(self, x_anchor, timesteps, cond, noise):
        alpha_t, sigma_t = self._get_alpha_sigma(x_anchor, timesteps)
        z_t = alpha_t * x_anchor + sigma_t * noise
        noise_pred = self.unet(
            sample=z_t,
            timestep=timesteps,
            encoder_hidden_states=cond,
            return_dict=False,
        )[0]
        x0_pred = (z_t - sigma_t * noise_pred) / alpha_t
        return x0_pred

    def _u_estimate(self, x_anchor, src_embed, edit_embed, noise, t_s: float, delta: float, adaptive_cfg):
        batch, device = x_anchor.shape[0], x_anchor.device
        t_idx_s = self._time_to_index(batch, t_s, device=device)
        t_idx_s0 = self._time_to_index(batch, max(0.0, t_s - delta), device=device)

        noises = noise if isinstance(noise, (list, tuple)) else [noise]

        alpha_s, sigma_s = self._get_alpha_sigma(x_anchor, t_idx_s)
        alpha_prev, sigma_prev = self._get_alpha_sigma(x_anchor, t_idx_s0)

        num_noises = len(noises)
        noise_stack = torch.stack(noises, dim=0)

        x_anchor_b = x_anchor.unsqueeze(0).expand(num_noises, -1, -1, -1, -1)
        alpha_s_b = alpha_s.unsqueeze(0).expand(num_noises, -1, -1, -1, -1)
        alpha_prev_b = alpha_prev.unsqueeze(0).expand(num_noises, -1, -1, -1, -1)
        sigma_s_b = sigma_s.unsqueeze(0).expand(num_noises, -1, -1, -1, -1)
        sigma_prev_b = sigma_prev.unsqueeze(0).expand(num_noises, -1, -1, -1, -1)

        z_s = alpha_s_b * x_anchor_b + sigma_s_b * noise_stack
        z_prev = alpha_prev_b * x_anchor_b + sigma_prev_b * noise_stack

        samples = torch.stack([z_s, z_s, z_prev, z_prev], dim=1)
        samples = samples.reshape(num_noises * 4 * batch, *x_anchor.shape[1:])

        conds = torch.cat([src_embed, edit_embed, src_embed, edit_embed], dim=0)
        repeat_dims = [num_noises] + [1] * (conds.dim() - 1)
        conds = conds.repeat(*repeat_dims)

        timesteps = torch.cat([t_idx_s, t_idx_s, t_idx_s0, t_idx_s0], dim=0)
        timesteps = timesteps.repeat(num_noises)

        alpha_cat = torch.stack(
            [alpha_s_b, alpha_s_b, alpha_prev_b, alpha_prev_b],
            dim=1,
        ).reshape(num_noises * 4 * batch, 1, 1, 1)
        sigma_cat = torch.stack(
            [sigma_s_b, sigma_s_b, sigma_prev_b, sigma_prev_b],
            dim=1,
        ).reshape(num_noises * 4 * batch, 1, 1, 1)

        noise_pred = self.unet(
            sample=samples,
            timestep=timesteps,
            encoder_hidden_states=conds,
            return_dict=False,
        )[0]

        x0_all = (samples - sigma_cat * noise_pred) / alpha_cat
        x0_all = x0_all.reshape(num_noises, 4, batch, *x_anchor.shape[1:])
        x_src_p_s, x_tar_p_s, x_src_p_s0, x_tar_p_s0 = x0_all.unbind(dim=1)

        dv_s = (x_tar_p_s - x_src_p_s).sum(dim=0) / float(num_noises)
        dv_s0 = (x_tar_p_s0 - x_src_p_s0).sum(dim=0) / float(num_noises)

        if adaptive_cfg is None or not adaptive_cfg.get("enable", False):
            denom = (t_s + delta)
            if denom <= 1e-6:
                return dv_s
            return (delta * dv_s + t_s * dv_s0) / denom

        # === 【重构版：时间自适应】基于全局能量计算标量 delta ===
        alpha = float(adaptive_cfg.get("alpha", 1.0))
        floor = float(adaptive_cfg.get("floor", 0.2))

        mean_energy = (dv_s0 ** 2).mean().item()
        threshold = 0.1
        scale = threshold / (mean_energy + 1e-6)
        energy_factor = torch.tanh(torch.tensor(scale - 1.0)).item()

        delta_eff = delta * ((1.0 - alpha) + alpha * (floor + (1.0 - floor) * (0.5 + 0.5 * energy_factor)))
        delta_eff = max(0.0, min(t_s - 1e-6, delta_eff))

        return (delta_eff * dv_s + t_s * dv_s0) / (t_s + delta_eff)

    def _run_edit(
            self,
            x_src: torch.Tensor,
            src_embed: torch.Tensor,
            edit_embed: torch.Tensor,
            noise: List[torch.Tensor],
            params: Dict[str, Any],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = x_src.device
        if params["n_steps"] == 1:
            t_grid = [params["t_start"]]
        else:
            t_grid = torch.linspace(
                params["t_start"], params["t_end"],
                steps=params["n_steps"], device=device,
            ).tolist()

        use_local = params["use_local_chord"]
        beta = params["local_beta"]
        tau_lo = params["mask_tau_low"]
        tau_hi = params["mask_tau_high"]
        sigma = params["mask_sigma"]
        remap_lo = float(params.get("alpha_remap_lo", 0.30))
        remap_w = max(1e-6, float(params.get("alpha_remap_width", 0.30)))
        remap_latent = bool(params.get("use_alpha_remap", True))

        x_curr = x_src
        last_mask: Optional[torch.Tensor] = None
        adaptive_cfg = None

        # 修复：清理重复的 adaptive_cfg 判断
        if params["use_adaptive_delta"]:
            adaptive_cfg = {
                "enable": True,
                "alpha": params["adaptive_alpha"],
                "floor": params["adaptive_floor"],
            }

        for t_s in t_grid:
            # 1. 获取方向 (包含全局时间自适应)
            u_hat = self._u_estimate(
                x_curr, src_embed, edit_embed, noise,
                float(t_s), params["t_delta"], adaptive_cfg=adaptive_cfg,
            )

            # 2. Transport 更新
            if use_local:
                mask = self._energy_mask(u_hat, tau_lo, tau_hi, sigma)  # (B,1,H,W)
                if remap_latent:
                    # 与像素阶段同一套近二值化，保证"背景 latent == 源 latent"严格成立
                    mask = ((mask - remap_lo) / remap_w).clamp(0.0, 1.0)
                last_mask = mask
                if params.get("local_apply_transport", True):
                    u_local = mask * u_hat + (1.0 - mask) * beta * u_hat
                    x_curr = x_curr + params["step_scale"] * u_local
                else:
                    # 原版 transport，力度完全保留
                    x_curr = x_curr + params["step_scale"] * u_hat
            else:
                x_curr = x_curr + params["step_scale"] * u_hat

        # 3. Cleanup 阶段（恢复潜空间背景锚定）
        if params["cleanup"]:
            t_end_idx = self._time_to_index(x_src.shape[0], params["t_end"], device=device)

            if use_local and last_mask is not None:
                # 在潜空间里，把背景拉回原图，让 Prox 看到的背景是原图
                x_pred_reg = last_mask * x_curr + (1.0 - last_mask) * x_src
                x_prox = self._pred_x0(x_pred_reg, t_end_idx, edit_embed, noise[0])
                # 输出潜空间再次锚定，防止背景微变
                x_curr = last_mask * x_prox + (1.0 - last_mask) * x_src
            else:
                x_curr = self._pred_x0(x_curr, t_end_idx, edit_embed, noise[0])

        return x_curr, last_mask