"""Freeze local ChordEdit inputs and export a manifest from the official runner."""
import argparse
import hashlib
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PIE_SOURCE = "https://github.com/cure-lab/PnPInversion#benchmark-download"
PIE_TERMS = ("PIE-Bench annotations: CC BY-NC-SA 4.0. Source-image copyrights remain with "
             "the rights holders. No private redistribution per authorized access email.")
PIE_TERMS_URL = "https://docs.google.com/document/d/1K7pjAgGwS8-lsA-D6Me4jAKmYszKpcyydiAEyZuTm2I/edit"
DEMO_SOURCE = "https://github.com/ChordEdit/ChordEdit/tree/a33d8cc26108851c6545b998c0e939006f89c9bd"
PARAMS = {
    "seed": 42, "precision": "fp32", "compute_dtype": "fp32", "device": "cuda:0",
    "noise_samples": 1, "n_steps": 1, "t_start": 0.9, "t_end": 0.3,
    "t_delta": 0.15, "step_scale": 1.0, "cleanup": True, "image_size": 512,
    "center_crop": True, "use_attention_mask": False, "use_safety_checker": False,
}
MODEL = {"id": "stabilityai/sd-turbo", "revision": "b261bac6fd2cf515557d5d0707481eafa0485ec2"}

def dump(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(data, encoding="utf-8")
    temp.replace(path)

def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()

def build_pie(origin):
    mapping = json.loads((origin / "mapping_file.json").read_text())
    by_type = defaultdict(list)
    for sid, meta in sorted(mapping.items()):
        rel = Path(meta["image_path"])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"Unexpected image path for {sid}")
        category = str(meta.get("editing_type_id", rel.parts[0].split("_")[0]))
        image = origin / "annotation_images" / rel
        if meta.get("original_prompt") and meta.get("editing_prompt"):
            by_type[category].append((sid, meta, image))
    if set(by_type) != {str(n) for n in range(10)} or min(map(len, by_type.values())) < 3:
        raise ValueError("Expected 10 categories with at least 3 valid samples in each")
    rng = random.Random(20261104)
    selection = [
        (sid, meta, image, cat, PIE_SOURCE, PIE_TERMS)
        for cat, rows in sorted(by_type.items()) for sid, meta, image in rng.sample(rows, 3)
    ]
    missing = [str(image) for _, _, image, _, _, _ in selection if not image.is_file()]
    if missing:
        raise FileNotFoundError("Selected images are missing:\n" + "\n".join(missing))
    return selection

def build_demo():
    images = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    rows = []
    for meta_file in sorted((ROOT / "upstream").rglob("meta.jsonl")):
        candidates = sorted(p for p in meta_file.parent.iterdir() if p.suffix.lower() in images)
        if not candidates:
            continue
        image = next((p for p in candidates if p.stem.lower() in {"i", "image", "original"}), candidates[0])
        for index, line in enumerate(meta_file.read_text().splitlines(), 1):
            if not line.strip():
                continue
            meta = json.loads(line)
            if meta.get("original_prompt") and meta.get("edited_prompt"):
                record = {"original_prompt": meta["original_prompt"].strip(),
                          "editing_prompt": meta["edited_prompt"].strip()}
                rows.append((f"{meta_file.parent.name}-{index:02}", record, image,
                             "official_demo", DEMO_SOURCE,
                             "Upstream demo asset. Individual image redistribution rights unverified."))
    if not rows:
        raise ValueError("No upstream demo examples found")
    return rows

def manifest_row(bundle, rel, row, acquisition):
    sid, meta, image, category, source, terms = row
    source_path = Path("annotation_images") / rel
    return {
        "id": sid, "category": category,
        "source_prompt": meta["original_prompt"], "target_prompt": meta["editing_prompt"],
        "edit_instruction": meta.get("editing_instruction"),
        "source": source, "acquisition": acquisition, "terms": terms,
        "terms_url": PIE_TERMS_URL if source == PIE_SOURCE else None,
        "upstream_image_path": meta.get("image_path") or image.relative_to(ROOT).as_posix(),
        "input_image": source_path.as_posix(), "input_sha256": digest(bundle / source_path),
        "expected_output_image": (Path("output/ChordEdit/annotation_images") / rel).as_posix(),
        "output_image": None, "output_sha256": None, "status": "pending",
        "params": PARAMS, "model": MODEL, "chordedit_commit": "a33d8cc26108851c6545b998c0e939006f89c9bd",
    }

def prepare(bundle, rows, acquisition):
    bundle.mkdir(parents=True)
    manifest, subset = [], {}
    for row in rows:
        sid, meta, image = row[:3]
        rel = Path(sid) / image.name
        dest = bundle / "annotation_images" / rel
        dest.parent.mkdir(parents=True)
        shutil.copyfile(image, dest)
        subset[sid] = {**meta, "image_path": rel.as_posix()}
        manifest.append(manifest_row(bundle, rel, row, acquisition))
    dump(bundle / "mapping_file.json", json.dumps(subset, indent=2, ensure_ascii=False) + "\n")
    write_manifest(bundle, manifest)

def prepare_full(bundle, acquisition):
    from PIL import Image
    mapping = json.loads((bundle / "mapping_file.json").read_text())
    if len(mapping) != 700:
        raise ValueError(f"Expected 700 annotations, got {len(mapping)}")
    manifest, paths = [], set()
    for sid, meta in sorted(mapping.items()):
        rel = Path(meta["image_path"])
        if rel.is_absolute() or ".." in rel.parts or rel in paths:
            raise ValueError(f"Unexpected or duplicate image path for {sid}: {rel}")
        paths.add(rel)
        if not meta["original_prompt"] or not meta["editing_prompt"]:
            raise ValueError(f"Missing prompts for {sid}")
        image = bundle / "annotation_images" / rel
        with Image.open(image) as source:
            if source.size != (512, 512):
                raise ValueError(f"Unexpected image size for {sid}: {source.size}")
            source.load()
        row = (sid, meta, image, str(meta["editing_type_id"]), PIE_SOURCE, PIE_TERMS)
        manifest.append(manifest_row(bundle, rel, row, acquisition))
    write_manifest(bundle, manifest)

def write_manifest(bundle, manifest):
    dump(bundle / "manifest.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in manifest))

def finalize(bundle):
    from PIL import Image
    rows = [json.loads(line) for line in (bundle / "manifest.jsonl").read_text().splitlines()]
    log = bundle / "run.log"
    logs = log.read_text(errors="replace").splitlines() if log.is_file() else []
    for row in rows:
        rel = row["expected_output_image"]
        path = bundle / rel
        row.update(output_image=None, output_sha256=None, status="failed", failure=None)
        try:
            with Image.open(path) as image:
                if image.size != (512, 512):
                    raise ValueError(f"Unexpected image size {image.size}")
                image.verify()
            row.update(output_image=rel, output_sha256=digest(path), status="generated")
        except (OSError, ValueError) as exc:
            details = next((line for line in logs if "Pipeline failed" in line and row["id"] in line), None)
            row["failure"] = details or str(exc)
    write_manifest(bundle, rows)
    completed = sum(r["status"] == "generated" for r in rows)
    print(f"{completed}/{len(rows)} valid outputs")
    return completed == len(rows)

parser = argparse.ArgumentParser()
parser.add_argument("action", choices=["demo", "pie", "pie-full", "finalize"])
parser.add_argument("bundle", type=Path)
parser.add_argument("--pie-root", type=Path)
parser.add_argument("--acquisition", default=PIE_SOURCE)
args = parser.parse_args()
if args.action == "finalize":
    sys.exit(0 if finalize(args.bundle) else 1)
elif args.action == "demo":
    prepare(args.bundle, build_demo(), DEMO_SOURCE)
elif args.action == "pie-full":
    prepare_full(args.bundle, args.acquisition)
else:
    if args.pie_root is None:
        parser.error("--pie-root required for pie")
    prepare(args.bundle, build_pie(args.pie_root), args.acquisition)

