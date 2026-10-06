#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")" && pwd)"
cd "$repo_root"

mkdir -p results

lambdas=(0 0.25 0.5 1 2)
seeds=(0 1 2 3 4)
steps=4000

total=$(( ${#lambdas[@]} * ${#seeds[@]} ))
run=0

for seed in "${seeds[@]}"; do
  for lambda in "${lambdas[@]}"; do
    run=$((run + 1))
    output="results/sweep_lambda${lambda}_seed${seed}.json"

    if [[ -f "$output" ]]; then
      echo "[$run/$total] skip lambda=$lambda seed=$seed"
      continue
    fi

    echo "[$run/$total] lambda=$lambda seed=$seed steps=$steps"
    python3 train.py --device cuda --require-cuda \
      --lambda-coupling "$lambda" \
      --seed "$seed" \
      --steps "$steps" \
      --eval-every 200 \
      --output "$output"
  done
done

echo "sweep complete: $total runs in results/"
