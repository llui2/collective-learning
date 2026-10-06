# Collective learning

Minimal experiments on coupled routing in mixture-of-experts Transformers.

The router is

```text
p0 = softmax(Q h)
C  = Q M Q^T
p  = softmax(Q h + lambda C p0)
```

with a learned low-rank symmetric metric `M = U diag(m) U^T`. The baseline is `lambda = 0`, so baseline and coupled runs have the same architecture and parameter count.

The main experiment uses five matched seeds, `lambda = 0, 0.25, 0.5, 1, 2`, and 4000 training updates per run. Completed runs are stored independently so the sweep is resumable.

## Minerva: computation

```bash
git pull
source .venv/bin/activate
./setup-minerva.sh

python3 train.py --device cuda --require-cuda --smoke
./experiment.sh
```

The sweep writes files such as

```text
results/sweep_lambda0_seed0.json
results/sweep_lambda0.25_seed0.json
...
results/sweep_lambda2_seed4.json
```

Commit and push the completed result files from Minerva.

## Mac: figures and manuscript

```bash
git pull
pip install -r requirements-plot.txt
./build.sh
```

Once all 25 runs are present, `build.sh` generates `draft/figures/fig1.pdf` and `fig1.csv`, then compiles `draft/main.pdf`. The figure shows mean validation curves with standard errors and paired final loss differences relative to `lambda = 0`. Generated figures and LaTeX artifacts are ignored by Git.
