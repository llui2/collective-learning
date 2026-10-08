#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
# CUDA 12.1 wheels are compatible with Minerva's CUDA 12.2 driver.
.venv/bin/python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c 'import torch; print(torch.__version__); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA unavailable")'
