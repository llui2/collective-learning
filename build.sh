#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")" && pwd)"
cd "$repo_root/draft"

latexmk -g -pdf -interaction=nonstopmode -halt-on-error main.tex
