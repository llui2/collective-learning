#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")" && pwd)"
cd "$repo_root"

git pull --ff-only

python_bin="${PYTHON:-$repo_root/.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  python_bin="$(command -v python3)"
fi

export MPLBACKEND=Agg

if [[ -s results/theory_baseline.npz ]]; then
  "$python_bin" draft/figures/fig1.py
else
  rm -f draft/figures/fig1.pdf
  echo "missing results/theory_baseline.npz; skip fig1"
fi

if [[ -s results/adaptive_theory.npz ]]; then
  "$python_bin" draft/figures/fig3.py
else
  rm -f draft/figures/fig3.pdf
  echo "missing results/adaptive_theory.npz; skip fig3"
fi

shopt -s nullglob
mnist_results=(results/mnist_seed*.json results/mnist_quick_seed*.json)
if (( ${#mnist_results[@]} > 0 )); then
  "$python_bin" draft/figures/fig2.py
else
  rm -f draft/figures/fig2.pdf
  echo "no MNIST result files; skip fig2"
fi

cd draft
latexmk -g -pdf -interaction=nonstopmode -halt-on-error main.tex
