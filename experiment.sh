#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python_bin="${PYTHON:-.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  python_bin="$(command -v python3)"
fi
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p results
case "${1:-}" in
  theory)
    "$python_bin" -m collective_learning.theory \
      --jobs "${THEORY_JOBS:-8}" --output results/theory_baseline.npz
    ;;
  mnist)
    for seed in 0 1 2; do
      "$python_bin" -m collective_learning.mnist \
        --quick --seed "$seed" --device "${DEVICE:-auto}" \
        --output "results/mnist_quick_seed${seed}.json"
    done
    ;;
  microscopic)
    shift
    "$python_bin" -m collective_learning.microscopic "$@"
    ;;
  adaptive)
    shift
    "$python_bin" -m collective_learning.adaptive "$@"
    ;;
  smoke)
    "$python_bin" -m unittest discover -s tests -q
    "$python_bin" -m collective_learning.theory \
      --smoke --jobs 2 --output results/theory_smoke.npz
    "$python_bin" -m collective_learning.mnist \
      --smoke --device "${DEVICE:-cpu}" --output results/mnist_smoke.json
    ;;
  *)
    echo "Usage: ./experiment.sh {theory|mnist|microscopic|adaptive|smoke}" >&2
    exit 2
    ;;
esac
