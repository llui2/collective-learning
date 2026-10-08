#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")" && pwd)"
cd "$repo_root"
git pull --ff-only
mkdir -p results

python_bin="${PYTHON:-$repo_root/.venv/bin/python}"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

"$python_bin" - <<'PY'
import torch
import torchvision
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for the Minerva performance sweep")
print("GPU:", torch.cuda.get_device_name(0))
PY

"$python_bin" -m unittest discover -s tests -q

mode="${PERFORMANCE_MODE:-quick}"
if [[ "$mode" == "quick" ]]; then
  seeds=(0)
  sigmas=(0.003 0.03 0.1 0.3 1 3)
  rates=(0.2 0.7 2)
  steps="${PERFORMANCE_STEPS:-8000}"
  eval_every=1000
elif [[ "$mode" == "full" ]]; then
  seeds=(0 1 2)
  sigmas=(0.003 0.03 0.1 0.3 1 3)
  rates=(0.2 0.7 2)
  steps="${PERFORMANCE_STEPS:-12000}"
  eval_every=2000
elif [[ "$mode" == "smoke" ]]; then
  seeds=(0)
  sigmas=(0.1)
  rates=(0.2)
  steps=50
  eval_every=25
else
  echo "unknown PERFORMANCE_MODE=$mode (quick, full, smoke)" >&2
  exit 1
fi

publish_result() {
  local result="$1"
  git add "$result"
  if ! git diff --cached --quiet; then
    git commit -m "neural performance: $(basename "$result")"
    git pull --rebase origin main
    git push origin main
  fi
}

read -r -a explorations <<< "${PERFORMANCE_EXPLORATIONS:-0.0005 0.03}"

run_case() {
  local control="$1" seed="$2" sigma="$3" rate="$4" exploration="$5"
  local sigma_key="${sigma//./p}"
  local rate_key="${rate//./p}"
  local exp_key="${exploration//./p}"
  local suffix="seed${seed}_s${sigma_key}"
  if [[ "$control" == "adaptive" ]]; then
    suffix="${suffix}_r${rate_key}_e${exp_key}"
  fi
  local out="results/neural_performance_${mode}_${control}_${suffix}.json"

  if [[ -s "$out" && "${FORCE:-0}" != "1" ]]; then
    if "$python_bin" - "$out" "$steps" "$control" "$sigma" "$rate" "$exploration" <<'PY'
import json
import math
import sys
try:
    with open(sys.argv[1]) as f:
        data = json.load(f)
    cfg = data["config"]
    valid = (
        data["complete"] and data["version"] == 1
        and cfg["steps"] == int(sys.argv[2])
        and cfg["control"] == sys.argv[3]
        and math.isclose(cfg["sigma"], float(sys.argv[4]))
        and (
            cfg["control"] != "adaptive"
            or (
                math.isclose(cfg["strategy_rate"], float(sys.argv[5]))
                and math.isclose(cfg["exploration"], float(sys.argv[6]))
            )
        )
    )
except (OSError, ValueError, KeyError, TypeError):
    valid = False
raise SystemExit(0 if valid else 1)
PY
    then
      echo "skip $out"
      return
    fi
  fi

  echo "== $control seed=$seed sigma=$sigma rate=$rate exploration=$exploration =="
  args=()
  if [[ "$mode" == "smoke" ]]; then
    args+=(--probe-samples 64 --validation-samples 128
            --representation-samples 500 --batch-size 32)
  fi

  "$python_bin" -m collective_learning.neural_performance \
    --control "$control" --seed "$seed" \
    --sigma "$sigma" --strategy-rate "$rate" \
    --exploration "$exploration" \
    --steps "$steps" --eval-every "$eval_every" \
    --device cuda --output "$out" "${args[@]}"

  publish_result "$out"
}

for seed in "${seeds[@]}"; do
  for sigma in "${sigmas[@]}"; do
    run_case uniform "$seed" "$sigma" 0 0.0005
    run_case frozen "$seed" "$sigma" 0 0.0005
    for rate in "${rates[@]}"; do
      for exploration in "${explorations[@]}"; do
        run_case adaptive "$seed" "$sigma" "$rate" "$exploration"
      done
    done
  done
done
