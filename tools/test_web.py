"""End-to-end check through the running web server (same path as the browser).

    .venv\\Scripts\\python.exe tools\\test_web.py
    .venv\\Scripts\\python.exe tools\\test_web.py --url http://127.0.0.1:7860/ --image images\\005\\i.jpg

The argument list follows app.py::run_edit; names are resolved from the
endpoint metadata so the script keeps working when new controls are added.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gradio_client import Client, handle_file  # noqa: E402
from PIL import Image  # noqa: E402

FALLBACK = {
    "seed": 42,
    "n_samples": 1,
    "t_start": 0.90,
    "t_end": 0.30,
    "t_delta": 0.15,
    "step_scale": 1.0,
    "use_local_chord": True,
    "local_beta": 0.1,
    "mask_tau_low": 0.1,
    "mask_tau_high": 0.4,
    "mask_sigma": 2.0,
    "use_adaptive_delta": True,
    "adaptive_alpha": 0.5,
    "adaptive_floor": 0.2,
    "use_detail_probe": True,
    "probe_t": 0.10,
    "probe_hf_sigma": 1.0,
    "probe_gate": 0.5,
    "probe_gate_width": 0.15,
    "use_energy_match": False,
    "use_edge_fit": True,
    "edge_radius": 8,
    "edge_dc_strength": 0.5,
    "use_alpha_remap": True,
    "alpha_remap_lo": 0.30,
    "alpha_remap_width": 0.30,
    "use_sharpen_u": False,
    "sharpen_gamma": 0.5,
}


def read_prompts(image_path: Path):
    with (image_path.parent / "meta.jsonl").open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                return rec["original_prompt"], rec["edited_prompt"]
    raise SystemExit("no meta.jsonl next to " + str(image_path))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:7860/")
    parser.add_argument("--image", default=str(ROOT / "images" / "005" / "i.jpg"))
    parser.add_argument("--out", default=str(ROOT / "outputs" / "web_test.png"))
    args = parser.parse_args()

    image_path = Path(args.image)
    src, tgt = read_prompts(image_path)

    client = Client(args.url, verbose=False)
    info = client.view_api(return_format="dict")
    endpoint = info["named_endpoints"]["/run_edit"]
    names = [p.get("parameter_name") or p.get("label") for p in endpoint["parameters"]]
    print("endpoint params:", len(names))

    values = []
    for name in names:
        if name in ("image", "source_image"):
            values.append(handle_file(str(image_path)))
        elif name in ("source_prompt", "src_prompt"):
            values.append(src)
        elif name in ("target_prompt", "tgt_prompt"):
            values.append(tgt)
        elif name in FALLBACK:
            values.append(FALLBACK[name])
        else:
            values.append(None)
            print("  (unknown parameter, passing None):", name)

    t0 = time.time()
    result = client.predict(*values, api_name="/run_edit")
    print("edit round-trip {:.1f}s".format(time.time() - t0))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    produced = Path(result[0] if isinstance(result, (list, tuple)) else result)
    Image.open(produced).convert("RGB").save(out)
    print("saved ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
