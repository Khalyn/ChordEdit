#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
bundle=$(realpath "$1")
py=${CHORDEDIT_PYTHON:-.venv/bin/python}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
"$py" -m pip freeze > "$bundle/environment.txt"
{
    "$py" --version
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
    if [[ -f upstream/REVISION ]]; then cat upstream/REVISION; else git -C upstream rev-parse HEAD; fi
} > "$bundle/runtime.txt"
set +e
"$py" upstream/run_pie_bench.py --model-root "$PWD/sd-turbo" --pie-root "$bundle" \
    --export-root "$bundle" --seed 42 --precision fp32 --device cuda:0 --image-size 512 \
    --n-steps 1 --noise-samples 1 --t-start 0.90 --t-end 0.30 --t-delta 0.15 \
    --step-scale 1.0 --cleanup --center-crop --no-safety-checker --no-sync-mapping \
    --log-every 1 2>&1 | tee "$bundle/run.log"
rc=${PIPESTATUS[0]}
set -e
"$py" prepare_samples.py finalize "$bundle"
exit "$rc"

