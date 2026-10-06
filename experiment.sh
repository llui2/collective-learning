#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")" && pwd)"
cd "$repo_root"

mkdir -p results

python3 train.py --device cuda --require-cuda \
  --lambda-coupling 0 \
  --seed 0 \
  --output results/lambda0.json

python3 train.py --device cuda --require-cuda \
  --lambda-coupling 1 \
  --seed 0 \
  --output results/lambda1.json

echo "results written to results/"
