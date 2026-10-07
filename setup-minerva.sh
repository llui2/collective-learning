#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")" && pwd)"
cd "$repo_root"

python3 -m venv .venv
python_bin="$repo_root/.venv/bin/python"

"$python_bin" -m pip install --upgrade pip
"$python_bin" -m pip uninstall -y torch torchvision torchaudio || true

# Minerva's NVIDIA driver supports CUDA 12.2. Use the CUDA 12.1
# PyTorch wheels, which are compatible with that driver.
"$python_bin" -m pip install   torch==2.5.1   torchvision==0.20.1   --index-url https://download.pytorch.org/whl/cu121

"$python_bin" -m pip install -r requirements-compute.txt

"$python_bin" - <<'PY'
import torch
import torchvision

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is still unavailable; do not start the experiment.")
print("gpu:", torch.cuda.get_device_name(0))
PY
