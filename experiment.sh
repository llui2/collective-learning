#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")" && pwd)"
cd "$repo_root"

mkdir -p runs draft/figures

python3 train.py \
  --lambda-coupling 0 \
  --seed 0 \
  --output runs/lambda0.json

python3 train.py \
  --lambda-coupling 1 \
  --seed 0 \
  --output runs/lambda1.json

export MPLBACKEND=Agg
python3 scripts/fig1.py

./build.sh
