#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")" && pwd)"
cd "$repo_root"

git pull --ff-only
mkdir -p results

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

publish() {
  local message="$1"
  shift
  git add "$@"
  if ! git diff --cached --quiet; then
    git commit -m "$message"
    git push
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

echo "== MNIST baseline =="
for seed in {0..9}; do
  out="results/mnist_seed${seed}.json"
  if [[ -s "$out" && "${FORCE:-0}" != "1" ]]; then
    echo "skip $out"
    continue
  fi

  echo "-- seed $seed --"
  "$python_bin" -m collective_learning.mnist     --runs 1     --seed "$seed"     --device cuda     --output "$out"

  publish "MNIST baseline seed $seed" "$out"
done
