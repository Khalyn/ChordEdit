"""Minimal smoke test: run ChordEdit once on one example image.

    .venv\\Scripts\\python.exe tools\\test_edit.py                       # fp32 (default)
    .venv\\Scripts\\python.exe tools\\test_edit.py --dtype fp16
    .venv\\Scripts\\python.exe tools\\test_edit.py --image images\\005\\i.jpg --out outputs\\cat.png

Prompts are read from the sibling meta.jsonl when --src/--tgt are not given.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from pipeline_chord import ChordEditPipeline  # noqa: E402


def read_prompts(image_path: Path):
    meta = image_path.parent / "meta.jsonl"
    if meta.exists():
        with meta.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    return rec.get("original_prompt", ""), rec.get("edited_prompt", "")
    return "", ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", default=str(ROOT / "models" / "sd-turbo"))
    parser.add_argument("--image", default=str(ROOT / "images" / "001" / "i.jpg"))
    parser.add_argument("--src", default=None)
    parser.add_argument("--tgt", default=None)
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"])
    parser.add_argument("--out", default=None)
    parser.add_argument("--t-start", type=float, default=0.90)
    parser.add_argument("--t-end", type=float, default=0.30)
    parser.add_argument("--t-delta", type=float, default=0.15)
    parser.add_argument("--step-scale", type=float, default=1.0)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    image_path = Path(args.image)
    src, tgt = read_prompts(image_path)
    src = args.src or src
    tgt = args.tgt or tgt
    if not src or not tgt:
        print("source/target prompt missing; pass --src/--tgt")
        return 2

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    root = Path(args.model_root)
    paths = {
        key: str((root / sub).resolve())
        for key, sub in {
            "unet_path": "unet",
            "scheduler_path": "scheduler",
            "text_encoder_path": "text_encoder",
            "tokenizer_path": "tokenizer",
            "vae_path": "vae",
        }.items()
    }

    print("loading pipeline (dtype={}) ...".format(args.dtype))
    t0 = time.time()
    pipe = ChordEditPipeline.from_local_weights(
        component_paths=paths,
        default_edit_config={
            "noise_samples": args.samples,
            "n_steps": 1,
            "t_start": args.t_start,
            "t_end": args.t_end,
            "t_delta": args.t_delta,
            "step_scale": args.step_scale,
            "cleanup": True,
        },
        torch_dtype=dtype,
        compute_dtype=dtype,
        image_size=512,
        use_center_crop=True,
        use_attention_mask=False,
        use_safety_checker=False,
    )
    print("loaded in {:.1f}s".format(time.time() - t0))
    if torch.cuda.is_available():
        print("vram after load: {:.2f} GB".format(torch.cuda.max_memory_allocated() / 2**30))

    image = Image.open(image_path).convert("RGB")
    print("editing {} ...".format(image_path))
    t0 = time.time()
    result = pipe(image=image, source_prompt=src, target_prompt=tgt, seed=args.seed)
    elapsed = time.time() - t0
    print("edit done in {:.2f}s".format(elapsed))
    if torch.cuda.is_available():
        print("peak vram: {:.2f} GB".format(torch.cuda.max_memory_allocated() / 2**30))

    out = Path(args.out) if args.out else ROOT / "outputs" / (image_path.parent.name + "_{}.png".format(args.dtype))
    out.parent.mkdir(parents=True, exist_ok=True)
    result.images[0].save(out)
    print("saved ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
