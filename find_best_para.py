from __future__ import annotations

import os

# 镜像加速
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import json
import logging
import itertools
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # 引入进度条

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import lpips
import open_clip

from pipeline_chord import ChordEditPipeline

LOGGER = logging.getLogger("tune_pie_bench")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

COMPONENT_SUBDIRS = {
    "unet_path": "unet", "scheduler_path": "scheduler",
    "text_encoder_path": "text_encoder", "tokenizer_path": "tokenizer", "vae_path": "vae",
}
DEFAULT_MODEL_ROOT = "/sd-turbo"
DEFAULT_PIE_ROOT = Path(__file__).resolve().parent / "pie_bench"
DEFAULT_MAPPING_FILE = "mapping_file.json"
DEFAULT_IMAGE_SUBDIR = "annotation_images"

# 基础固定参数 (不参与网格搜索)
BASE_EDIT_CONFIG = {
    "noise_samples": 1,
    "n_steps": 1,
    "t_start": 0.90,
    "t_end": 0.30,
    "t_delta": 0.15,
    "step_scale": 1.0,
    "cleanup": True,
    # 开启三大创新模块
    "use_local_chord": True,
    "local_apply_transport": False,  # 关键：潜空间不拼接，像素空间锚定
    "use_adaptive_delta": True,
    "use_sharpen_u": True,
}

# ==========================================
# 模块划分与参数搜索空间 (按参数数量从少到多排序)
# ==========================================
MODULES = {
    # 1个参数
    "Field_Sharpen": {
        "sharpen_gamma": [0.01, 0.02, 0.05, 0.1],
    },
    # 2个参数
    "Adaptive_Delta": {
        "adaptive_alpha": [0.0, 0.1, 0.5, 1.0],
        "adaptive_floor": [0.0, 0.2, 0.4, 0.6, 1.0],
    },
    # 4个参数
    "Local_Chord": {
        "local_beta": [0.0, 0.05, 0.1, 0.3, 0.5, 1.0],
        "mask_sigma": [0.0, 1.0, 2.0],
        "mask_tau_low": [0.0, 0.1, 0.3, 0.5],
        "mask_tau_high": [0.6, 0.8, 1.0],  # 强制比 tau_low 大，防止 Mask 失效
    }
}

# 全局默认中间值 (初始起点)
DEFAULT_PARAMS = {
    "local_beta": 0.05, "mask_sigma": 1.0, "mask_tau_low": 0.9, "mask_tau_high": 0.95,
    "adaptive_alpha": 0.5, "adaptive_floor": 0.6,
    "sharpen_gamma": 0.01,
}


@dataclass(frozen=True)
class PieRecord:
    sample_id: str
    image_path: Path
    target_image_path: Path
    relative_path: Path
    original_prompt: str
    edited_prompt: str


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=str, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--pie-root", type=str, default=str(DEFAULT_PIE_ROOT))
    parser.add_argument("--mapping-file", type=str, default=DEFAULT_MAPPING_FILE)
    parser.add_argument("--image-subdir", type=str, default=DEFAULT_IMAGE_SUBDIR)
    parser.add_argument("--output-csv", type=str, default="tuning_results.csv")
    parser.add_argument("--max-samples", type=int, default=3, help="每个实验使用的图片数量")
    parser.add_argument("--workers", type=int, default=1, help="多线程数量")
    return parser.parse_args()


def resolve_path(base: Path, maybe_relative: str | Path) -> Path:
    candidate = Path(maybe_relative)
    return candidate.expanduser().resolve() if candidate.is_absolute() else (base / candidate).expanduser().resolve()


def load_pie_records(root: Path, mapping_path: Path, image_subdir: str) -> List[PieRecord]:
    with mapping_path.open("r", encoding="utf-8") as handle:
        mapping = json.load(handle)
    img_root = (root / image_subdir).expanduser().resolve()
    records = []
    for sample_id in sorted(mapping.keys()):
        meta = mapping[sample_id]
        rel_value = meta.get("image_path")
        if not rel_value: continue

        rel_path = Path(rel_value)
        abs_path = (img_root / rel_path).expanduser().resolve()
        target_rel_path = Path(str(rel_value).replace("annotation_images", "target_images"))
        target_abs_path = (img_root / target_rel_path).expanduser().resolve()
        if not target_abs_path.exists(): target_abs_path = abs_path

        records.append(PieRecord(
            sample_id=sample_id, image_path=abs_path, target_image_path=target_abs_path,
            relative_path=rel_path, original_prompt=meta.get("original_prompt", ""),
            edited_prompt=meta.get("edited_prompt", ""),
        ))
    return records


class MetricsCalculator:
    def __init__(self, device):
        self.device = device
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.lpips_model = lpips.LPIPS(net='alex').to(device)
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32',
                                                                                         pretrained='openai')
        self.clip_model = self.clip_model.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')

    def _get_clip_score(self, img: Image.Image, prompt: str) -> float:
        img_input = self.clip_preprocess(img).unsqueeze(0).to(self.device)
        text_input = self.tokenizer([prompt]).to(self.device)
        with torch.no_grad():
            img_features = self.clip_model.encode_image(img_input)
            text_features = self.clip_model.encode_text(text_input)
            img_features /= img_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            return (100.0 * img_features @ text_features.T).item()

    def compute(self, img_gen: Image.Image, img_gt: Image.Image, prompt: str, mask: Optional[torch.Tensor] = None) -> \
    Dict[str, float]:
        t_gen = torch.tensor(np.array(img_gen.resize((512, 512))), dtype=torch.float32).permute(2, 0, 1).unsqueeze(
            0) / 255.0
        t_gt = torch.tensor(np.array(img_gt.resize((512, 512))), dtype=torch.float32).permute(2, 0, 1).unsqueeze(
            0) / 255.0
        t_gen, t_gt = t_gen.to(self.device), t_gt.to(self.device)

        # 1. 整图指标
        psnr_val = self.psnr(t_gen, t_gt).item()
        ssim_val = self.ssim(t_gen, t_gt).item()
        lpips_val = self.lpips_model(t_gen * 2.0 - 1.0, t_gt * 2.0 - 1.0).item()

        # 2. 纯背景指标 (对应原论文的表格)
        bg_psnr, bg_ssim, bg_lpips = psnr_val, ssim_val, lpips_val
        clip_edited = 0.0

        if mask is not None:
            try:
                # 把 64x64 的 mask 放大到 512x512
                mask_pil = Image.fromarray((mask[0, 0].cpu().numpy() * 255).astype(np.uint8)).resize((512, 512),
                                                                                                     Image.BILINEAR)
                mask_np = np.array(mask_pil) > 127

                # 计算背景指标 (Mask 为 False 的区域)
                bg_mask_tensor = torch.tensor(~mask_np).to(self.device).unsqueeze(0).unsqueeze(0)
                if bg_mask_tensor.sum() > 0:
                    t_gen_bg = t_gen * bg_mask_tensor
                    t_gt_bg = t_gt * bg_mask_tensor
                    bg_psnr = self.psnr(t_gen_bg, t_gt_bg).item()
                    bg_ssim = self.ssim(t_gen_bg, t_gt_bg).item()
                    bg_lpips = self.lpips_model(t_gen_bg * 2.0 - 1.0, t_gt_bg * 2.0 - 1.0).item()

                # 计算编辑区域 CLIP
                if mask_np.sum() > 512 * 512 * 0.01:
                    img_edited = np.array(img_gen.resize((512, 512)))
                    img_edited[~mask_np] = 0
                    clip_edited = self._get_clip_score(Image.fromarray(img_edited), prompt)
            except Exception as e:
                pass

        clip_whole = self._get_clip_score(img_gen, prompt)

        return {
            "PSNR": psnr_val, "SSIM": ssim_val, "LPIPS": lpips_val,
            "BG_PSNR": bg_psnr, "BG_SSIM": bg_ssim, "BG_LPIPS": bg_lpips,  # 新增背景指标
            "Struct_Dist_1000": (1.0 - bg_ssim) * 1000.0,  # 原论文的结构距离其实是背景的
            "CLIP_Whole": clip_whole, "CLIP_Edited": clip_edited
        }


gpu_lock = threading.Lock()


def compute_composite_score(m):
    return (
            0.10 * (m["PSNR"] / 30.0) + 0.10 * m["SSIM"] + 0.10 * (1.0 - m["LPIPS"]) +
            0.20 * (m["BG_PSNR"] / 30.0) + 0.20 * m["BG_SSIM"] + # 重点奖励背景
            0.15 * (m["CLIP_Whole"] / 30.0) + 0.15 * (m["CLIP_Edited"] / 30.0)
    )


# ==========================================
# 打印输出工具
# ==========================================
def print_best_metrics(df, stage_name=None, title="BEST CONFIG"):
    if stage_name:
        df_stage = df[df["stage_name"] == stage_name]
    else:
        df_stage = df

    if df_stage.empty:
        return

    best_row = df_stage.loc[df_stage["Composite_Score"].idxmax()]
    LOGGER.info("=" * 60)
    LOGGER.info(f"🏆 {title}")
    LOGGER.info(f"  Score       : {best_row['Composite_Score']:.4f}")
    LOGGER.info(f"  BG_PSNR     : {best_row['BG_PSNR']:.2f}  (原论文: 22.63)")
    LOGGER.info(f"  BG_SSIM     : {best_row['BG_SSIM']:.4f} (原论文: 0.7671)")
    LOGGER.info(f"  BG_LPIPS    : {best_row['BG_LPIPS']:.4f}")
    LOGGER.info(f"  CLIP_Whole  : {best_row['CLIP_Whole']:.2f}")
    LOGGER.info(f"  CLIP_Edited : {best_row['CLIP_Edited']:.2f}")
    LOGGER.info("  --- Parameter Config ---")
    for k in DEFAULT_PARAMS.keys():
        if k in best_row:
            LOGGER.info(f"    {k} = {best_row[k]}")
    LOGGER.info("=" * 60)


# ==========================================
# 任务生成器
# ==========================================
def build_module_tasks(module_name, current_best_params):
    tasks = []
    grid = MODULES[module_name]
    keys, vals = zip(*grid.items())

    for combo in itertools.product(*vals):
        exp = dict(current_best_params)
        exp.update(dict(zip(keys, combo)))
        tasks.append({"stage": f"Stage1_{module_name}", "params": exp})
    return tasks


def get_stage1_best(df, stage_name, module_keys):
    df_stage = df[df["stage_name"] == stage_name]
    if df_stage.empty: return None
    best_row = df_stage.loc[df_stage["Composite_Score"].idxmax()]
    return {k: best_row[k] for k in module_keys}


def build_stage2_task(best_params):
    return [{"stage": "Stage2_Joint_Best", "params": best_params}]


def build_stage3_tasks(best_params):
    tasks = []
    all_param_names = list(DEFAULT_PARAMS.keys())
    for param_name in all_param_names:
        module_name = next(m for m in MODULES if param_name in MODULES[m])
        vals = MODULES[module_name][param_name]
        for v in vals:
            exp = dict(best_params)
            exp[param_name] = v
            tasks.append({"stage": f"Stage3_OAT_{param_name}", "params": exp})
    return tasks


# ==========================================
# 执行器（带进度条）
# ==========================================
def run_tasks(tasks, pipeline, metrics_calc, records, output_csv, workers):
    if not tasks: return

    completed_params = []
    if os.path.exists(output_csv):
        df_exist = pd.read_csv(output_csv)
        if "stage_name" in df_exist.columns:
            for _, row in df_exist.iterrows():
                completed_params.append(
                    (row["stage_name"], str({k: round(float(row[k]), 4) for k in DEFAULT_PARAMS.keys()})))

    todo = []
    for t in tasks:
        t_str = str({k: round(float(v), 4) for k, v in t["params"].items()})
        if (t["stage"], t_str) not in completed_params:
            todo.append(t)

    LOGGER.info(f"Tasks: {len(tasks)} | Already Done: {len(tasks) - len(todo)} | Remaining: {len(todo)}")
    if not todo: return

    def process_task(task):
        exp_params = task["params"]
        edit_config = dict(BASE_EDIT_CONFIG)
        edit_config.update(exp_params)

        exp_metrics = {
            "PSNR": [], "SSIM": [], "LPIPS": [],
            "BG_PSNR": [], "BG_SSIM": [], "BG_LPIPS": [],  # 新增补齐
            "Struct_Dist_1000": [], "CLIP_Whole": [], "CLIP_Edited": []
        }

        for record in records:
            try:
                with Image.open(record.image_path) as img:
                    source_image = img.convert("RGB")
                with Image.open(record.target_image_path) as img_gt:
                    gt_image = img_gt.convert("RGB")

                with gpu_lock:
                    result = pipeline(
                        image=source_image, source_prompt=record.original_prompt,
                        target_prompt=record.edited_prompt, edit_config=edit_config,
                        seed=42, output_type="pil", return_mask=True
                    )
                    gen_img = result.images[0]
                    mask = result.masks

                    m = metrics_calc.compute(gen_img, gt_image, record.edited_prompt, mask)

                for k, v in m.items():
                    exp_metrics[k].append(v)
            except Exception as e:
                LOGGER.warning(f"Failed {record.sample_id}: {e}")

        if exp_metrics["PSNR"]:
            avg_m = {k: sum(v) / len(v) for k, v in exp_metrics.items()}
            avg_m["Composite_Score"] = compute_composite_score(avg_m)
            avg_m["stage_name"] = task["stage"]
            avg_m.update(exp_params)
            return avg_m
        return None

    # 使用 tqdm 包装进度条
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_task, t): t for t in todo}
        for future in tqdm(as_completed(futures), total=len(todo), desc="Running Experiments", unit="exp"):
            res = future.result()
            if res:
                df_res = pd.DataFrame([res])
                header = not os.path.exists(output_csv)
                df_res.to_csv(output_csv, mode='a', header=header, index=False)
                # 隐藏内部 Log，只通过进度条和最后的打印输出展示
                # LOGGER.info(f"Progress: {i + 1}/{len(todo)} done. Stage: {futures[future]['stage']}")


# ==========================================
# 绘图工具
# ==========================================
def plot_sensitivity(csv_path):
    if not os.path.exists(csv_path): return
    df = pd.read_csv(csv_path)
    df_oat = df[df["stage_name"].str.startswith("Stage3_OAT_", na=False)]
    if df_oat.empty: return

    metrics_to_plot = ["PSNR", "SSIM", "LPIPS", "CLIP_Whole", "CLIP_Edited", "Struct_Dist_1000"]
    params = list(DEFAULT_PARAMS.keys())

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(len(params), len(metrics_to_plot), figsize=(24, 16))
    plt.subplots_adjust(hspace=0.5, wspace=0.3)

    for r_idx, param in enumerate(params):
        df_sub = df_oat[df_oat["stage_name"] == f"Stage3_OAT_{param}"].sort_values(by=param)

        for c_idx, metric in enumerate(metrics_to_plot):
            ax = axes[r_idx, c_idx]
            if not df_sub.empty:
                sns.lineplot(data=df_sub, x=param, y=metric, marker="o", ax=ax, color="b")
                ax.set_title(f"{metric} vs {param}")
                if metric in ["Struct_Dist_1000", "LPIPS"]:
                    ax.invert_yaxis()
            else:
                ax.text(0.5, 0.5, "No data", ha='center', va='center')

    plt.tight_layout()
    plt.savefig("sensitivity_plots.png", dpi=150)
    LOGGER.info("Sensitivity plots saved to sensitivity_plots.png")


# ==========================================
# 主流程控制
# ==========================================
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        LOGGER.info("=" * 70)
        LOGGER.info(f"🚀 检测到 CUDA 设备：{torch.cuda.get_device_name(0)}")
        LOGGER.info("   已开启硬件加速优化 (TF32 & cuDNN Benchmark)")
        LOGGER.info(f"   并发线程数: {args.workers}")
        LOGGER.info("=" * 70)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch_dtype = torch.float32
    else:
        LOGGER.warning("⚠️ 未检测到 CUDA，正在使用 CPU。速度会极慢！")
        args.workers = 1
        if args.max_samples > 2:
            args.max_samples = 2
        torch_dtype = torch.float32

    pie_root = Path(args.pie_root).resolve()
    records = load_pie_records(pie_root, resolve_path(pie_root, args.mapping_file), args.image_subdir)[
        :args.max_samples]
    if not records:
        LOGGER.error("No records found!")
        return

    LOGGER.info(f"Loading models on {device}...")
    component_paths = {key: str((Path(args.model_root) / sub).resolve()) for key, sub in COMPONENT_SUBDIRS.items()}
    pipeline = ChordEditPipeline.from_local_weights(
        component_paths=component_paths,
        device=device,
        torch_dtype=torch_dtype,
        image_size=512
    )

    LOGGER.info("Loading Metrics Calculator (LPIPS / OpenCLIP)...")
    metrics_calc = MetricsCalculator(device)

    best_params = dict(DEFAULT_PARAMS)

    # ==========================================
    # Stage 1: 模块接力网格搜索
    # ==========================================
    LOGGER.info("========== STAGE 1: 模块接力网格搜索 ==========")
    for module_name in MODULES.keys():
        stage_name = f"Stage1_{module_name}"
        LOGGER.info(f"--- Starting search for {module_name} (Base params: {best_params}) ---")

        tasks = build_module_tasks(module_name, best_params)
        run_tasks(tasks, pipeline, metrics_calc, records, args.output_csv, args.workers)

        # 跑完后，打印该模块的最优结果
        df = pd.read_csv(args.output_csv)
        print_best_metrics(df, stage_name=stage_name, title=f"{module_name} Module Best")

        best_module_params = get_stage1_best(df, stage_name, MODULES[module_name].keys())
        if best_module_params:
            best_params.update(best_module_params)
            LOGGER.info(f"✅ {module_name} finished. Updated params: {best_params}")
        else:
            LOGGER.warning(f"⚠️ No results found for {module_name}. Keeping previous params.")

    LOGGER.info(f"--- Stage 1 Final Best Combined Params: {best_params} ---")

    # ==========================================
    # Stage 2: 最优联合测试
    # ==========================================
    LOGGER.info("========== STAGE 2: 最优联合测试 ==========")
    run_tasks(build_stage2_task(best_params), pipeline, metrics_calc, records, args.output_csv, args.workers)

    # 跑完后，打印联合最优结果
    df = pd.read_csv(args.output_csv)
    print_best_metrics(df, stage_name="Stage2_Joint_Best", title="Joint Best Performance (Stage 2)")

    # ==========================================
    # Stage 3: 单变量敏感性分析
    # ==========================================
    LOGGER.info("========== STAGE 3: 最优前提下的单变量敏感性分析 ==========")
    run_tasks(build_stage3_tasks(best_params), pipeline, metrics_calc, records, args.output_csv, args.workers)

    LOGGER.info("========== 生成图表 ==========")
    plot_sensitivity(args.output_csv)

    # 打印全局最终最优结果
    df = pd.read_csv(args.output_csv)
    print_best_metrics(df, title="FINAL GLOBAL BEST")


if __name__ == "__main__":
    main()