#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python_bin="${PYTHON:-.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  python_bin="$(command -v python3)"
fi
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
"$python_bin" -m unittest discover -s tests -q
"$python_bin" draft/figures/moe.py
(cd draft && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex)
