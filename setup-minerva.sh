#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install --upgrade pip
python3 -m pip uninstall -y torch || true
python3 -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python3 -m pip install -r requirements-compute.txt

python3 - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is still unavailable; do not start the experiment.")
print("gpu:", torch.cuda.get_device_name(0))
PY
