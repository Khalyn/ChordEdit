"""Score ChordEdit outputs with PIE-Bench's released metric functions."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import fmean

import numpy as np
import torch
from PIL import Image

from evaluation.evaluate import MetricsCalculator, calculate_metric, mask_decode

# ChordEdit Table 1 and Appendix G scales.
METRICS = {
    "psnr_unedit_part": ("PSNR", 1, 22.20),
    "mse_unedit_part": ("MSE × 1000", 1000, 6.84),
    "lpips_unedit_part": ("LPIPS × 1000", 1000, 128.25),
    "clip_similarity_target_image": ("CLIP-Whole", 1, 25.58),
    "clip_similarity_target_image_edit_part": ("CLIP-Edited", 1, 22.96),
    "ssim_unedit_part": ("SSIM × 100", 100, 75.91),
    "structure_distance": ("Structure distance × 1000", 1000, 16.58),
}
def metric_rows(mapping, source, output, scored, metrics):
    calculator = MetricsCalculator("cuda:0")
    with torch.inference_mode():
        for sid, row in mapping.items():
            if sid in scored:
                continue
            rel = row["image_path"]
            mask = np.repeat(mask_decode(row["mask"])[..., None], 3, axis=2)
            source_prompt = row["original_prompt"].replace("[", "").replace("]", "")
            target_prompt = row["editing_prompt"].replace("[", "").replace("]", "")
            with Image.open(source / rel) as before, Image.open(output / rel) as after:
                assert before.size == after.size == (512, 512), sid
                values = [calculate_metric(calculator, name, before, after, mask, mask,
                                           source_prompt, target_prompt) for name in metrics]
            yield {"id": sid, "category": row["editing_type_id"], **dict(zip(metrics, values))}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    bundle = args.bundle
    mapping = json.loads((bundle / "mapping_file.json").read_text())
    assert len(mapping) == 700
    rows_path = bundle / "metrics_per_image.csv"
    columns = ["id", "category", *METRICS]
    seen = set()
    if rows_path.exists():
        with rows_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            assert reader.fieldnames == columns
            for row in reader:
                assert row["id"] in mapping and row["id"] not in seen
                assert all(row[k] is not None for k in columns)
                seen.add(row["id"])
    else:
        with rows_path.open("x", newline="") as handle:
            csv.DictWriter(handle, fieldnames=columns).writeheader()
    if len(seen) != len(mapping):
        with rows_path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            for row in metric_rows(mapping, bundle / "annotation_images",
                                   bundle / "output/ChordEdit/annotation_images", seen, METRICS):
                writer.writerow(row)
                handle.flush()
                seen.add(row["id"])
                if len(seen) in (1, len(mapping)) or len(seen) % 25 == 0:
                    print(f"Scored {len(seen)}/{len(mapping)}", flush=True)
    with rows_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(mapping) and {r["id"] for r in rows} == set(mapping)
    summary = {}
    for name, (label, scale, paper) in METRICS.items():
        values = [float(row[name]) for row in rows]
        valid = [x for x in values if math.isfinite(x)]
        mean = fmean(valid) * scale
        summary[label] = {"local": mean, "paper": paper, "delta": mean - paper,
                          "valid": len(valid), "nan": sum(math.isnan(x) for x in values),
                          "infinite": sum(math.isinf(x) for x in values)}
    (bundle / "metrics_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()

