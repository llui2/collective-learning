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
if command -v latexmk >/dev/null 2>&1; then
  "$python_bin" draft/figures/arola_results.py
  "$python_bin" draft/figures/fig1.py
  "$python_bin" draft/figures/fig2.py
  (
    cd draft
    for name in main arola_results arola_baseline; do
      latexmk -pdf -interaction=nonstopmode -halt-on-error "$name.tex"
    done
  )
elif [[ "${REQUIRE_LATEX:-0}" == "1" ]]; then
  echo "Error: latexmk is required but not installed." >&2
  exit 127
else
  echo "Skipping manuscript PDF: latexmk not installed (tests and figures succeeded)." >&2
  echo "Install latexmk or set REQUIRE_LATEX=1 to require PDF compilation." >&2
fi
