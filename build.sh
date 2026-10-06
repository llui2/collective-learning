#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")" && pwd)"
cd "$repo_root"

n_sweep="$(find results -maxdepth 1 -name 'sweep_lambda*_seed*.json' 2>/dev/null | wc -l | tr -d ' ')"

if (( n_sweep >= 25 )); then
  mkdir -p draft/figures
  export MPLBACKEND=Agg
  python3 scripts/fig1.py
else
  rm -f draft/figures/fig1.pdf draft/figures/fig1.csv
  echo "sweep incomplete ($n_sweep/25); compiling manuscript without Fig. 1"
fi

cd draft
latexmk -g -pdf -interaction=nonstopmode -halt-on-error main.tex
