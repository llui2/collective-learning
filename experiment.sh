#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")" && pwd)"
cd "$repo_root"

git pull --ff-only
mkdir -p results

python_bin="${PYTHON:-$repo_root/.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  python_bin="$(command -v python3)"
fi

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
  "$python_bin" reproduce_theory.py --output "$theory_out"
  publish "theory baseline" "$theory_out"
else
  echo "skip $theory_out"
fi

for seed in {0..9}; do
  out="results/mnist_seed${seed}.json"
  if [[ -s "$out" && "${FORCE:-0}" != "1" ]]; then
    echo "skip $out"
    continue
  fi

  "$python_bin" reproduce_mnist.py     --runs 1     --seed "$seed"     --device cuda     --output "$out"

  publish "MNIST baseline seed $seed" "$out"
done
