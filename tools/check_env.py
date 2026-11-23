"""Environment self-check for ChordEdit.

Run with the project virtualenv:
    .venv\\Scripts\\python.exe tools\\check_env.py
"""

import importlib
import sys


def main() -> int:
    print("python      :", sys.version.split()[0], "|", sys.executable)
    print("-" * 60)

    failures = 0
    for name in [
        "torch",
        "torchvision",
        "numpy",
        "PIL",
        "yaml",
        "matplotlib",
        "seaborn",
        "diffusers",
        "transformers",
        "gradio",
    ]:
        try:
            mod = importlib.import_module(name)
            version = getattr(mod, "__version__", "?")
            print("{:<14s} OK    {}".format(name, version))
        except Exception as exc:  # pragma: no cover - diagnostic only
            failures += 1
            print("{:<14s} FAIL  {}: {}".format(name, type(exc).__name__, str(exc)[:200]))

    print("-" * 60)
    try:
        import torch

        print("torch cuda build :", torch.version.cuda)
        print("cuda available   :", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("gpu              :", torch.cuda.get_device_name(0))
            free, total = torch.cuda.mem_get_info()
            print("vram             : {:.1f} GB free / {:.1f} GB total".format(free / 2**30, total / 2**30))
            x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
            torch.cuda.synchronize()
            print("fp16 matmul      : ok ({:.3f})".format(float((x @ x).sum().item())))
    except Exception as exc:  # pragma: no cover
        failures += 1
        print("torch self-test FAIL:", exc)

    print("-" * 60)
    try:
        from diffusers.pipelines.pipeline_utils import DiffusionPipeline  # noqa: F401
        from diffusers.pipelines.stable_diffusion.safety_checker import (  # noqa: F401
            StableDiffusionSafetyChecker,
        )

        print("diffusers API    : OK")
    except Exception as exc:
        failures += 1
        print("diffusers API    FAIL:", type(exc).__name__, exc)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
