#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")" && pwd)"
cd "$repo_root"

git pull --ff-only
mkdir -p results
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

python_bin="${PYTHON:-$repo_root/.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  echo "missing compute environment: $python_bin" >&2
  echo "run ./setup-minerva.sh first" >&2
  exit 1
fi

"$python_bin" - <<'PY'
import numpy
import scipy
import torch
import torchvision

if not torch.cuda.is_available():
    raise SystemExit("PyTorch is installed but CUDA is not available")

print("python environment: ok")
print("torch:", torch.__version__)
print("gpu:", torch.cuda.get_device_name(0))
PY

result_complete() {
  local path="$1"
  [[ -s "$path" ]] || return 1
  "$python_bin" - "$path" <<'PY'
import json
import sys

try:
    with open(sys.argv[1]) as f:
        data = json.load(f)
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)

raise SystemExit(0 if data.get("complete") is True else 1)
PY
}

publish() {
  local message="$1"
  shift
  git add "$@"
  if ! git diff --cached --quiet; then
    git commit -m "$message"
    git pull --rebase origin main
    git push origin main
  fi
}

theory_out="results/theory_baseline.npz"
if [[ ! -s "$theory_out" || "${FORCE:-0}" == "1" ]]; then
  echo "== effective-theory baseline =="
  echo "workers: ${THEORY_JOBS:-8}"
  "$python_bin" -m collective_learning.theory     --jobs "${THEORY_JOBS:-8}"     --output "$theory_out"
  publish "theory baseline" "$theory_out"
else
  echo "skip $theory_out"
fi

adaptive_out="results/adaptive_theory.npz"
adaptive_current() {
  [[ -s "$adaptive_out" ]] || return 1
  "$python_bin" - "$adaptive_out" <<'PY'
import sys
import numpy as np

try:
    data = np.load(sys.argv[1])
    version = int(data["version"])
except (OSError, KeyError, ValueError):
    raise SystemExit(1)

raise SystemExit(0 if version == 2 else 1)
PY
}

if [[ "${FORCE:-0}" == "1" ]] || ! adaptive_current; then
  echo "== adaptive effective theory =="
  "$python_bin" -m collective_learning.adaptive_theory --output "$adaptive_out"
  publish "adaptive effective theory" "$adaptive_out"
else
  echo "skip $adaptive_out"
fi

mode="${MODE:-quick}"

if [[ "$mode" == "quick" ]]; then
  echo "== MNIST quick iteration =="
  seeds=(0 1 2)
  extra_args=(--quick)
  prefix="mnist_quick_seed"
elif [[ "$mode" == "full" ]]; then
  echo "== MNIST full reproduction =="
  seeds=({0..9})
  extra_args=()
  prefix="mnist_seed"
else
  echo "unknown MODE=$mode (use quick or full)" >&2
  exit 1
fi

for seed in "${seeds[@]}"; do
  out="results/${prefix}${seed}.json"
  if [[ "${FORCE:-0}" != "1" ]] && result_complete "$out"; then
    echo "skip $out"
    continue
  fi

  echo "-- seed $seed --"
  "$python_bin" -m collective_learning.mnist     --runs 1     --seed "$seed"     --device cuda     "${extra_args[@]}"     --output "$out"

  publish "MNIST $mode seed $seed" "$out"
done
