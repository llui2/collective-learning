#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")" && pwd)"
cd "$repo_root"

if [[ -f runs/lambda0.json && -f runs/lambda1.json ]]; then
  mkdir -p draft/figures
  export MPLBACKEND=Agg
  python3 scripts/fig1.py
fi

cd draft
latexmk -g -pdf -interaction=nonstopmode -halt-on-error main.tex
