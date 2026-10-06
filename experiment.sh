#!/usr/bin/env bash
set -euo pipefail

mkdir -p runs

python3 train.py \
  --lambda-coupling 0 \
  --seed 0 \
  --output runs/lambda0.json

python3 train.py \
  --lambda-coupling 1 \
  --seed 0 \
  --output runs/lambda1.json
