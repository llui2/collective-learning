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
  "$python_bin" draft/figures/fig7.py
else
  rm -f draft/figures/fig3.pdf draft/figures/fig7.pdf
  echo "missing results/adaptive_theory.npz; skip fig3 and fig7"
fi

shopt -s nullglob
adaptive_neural_results=(results/adaptive_neural_seed*.json results/adaptive_neural_quick_seed*.json results/adaptive_neural_pilot_seed*.json)
if (( ${#adaptive_neural_results[@]} > 0 )); then
  "$python_bin" draft/figures/fig4.py
else
  rm -f draft/figures/fig4.pdf
  echo "no adaptive neural result files; skip fig4"
fi

# fig5 uses the completed, same-horizon quick sweep.
"$python_bin" draft/figures/fig5.py

# fig6 uses only complete, step-matched adaptive/frozen pairs.
"$python_bin" draft/figures/fig6.py

mnist_results=(results/mnist_seed*.json results/mnist_quick_seed*.json)
if (( ${#mnist_results[@]} > 0 )); then
  "$python_bin" draft/figures/fig2.py
else
  rm -f draft/figures/fig2.pdf
  echo "no MNIST result files; skip fig2"
fi

cd draft
latexmk -g -pdf -interaction=nonstopmode -halt-on-error main.tex
