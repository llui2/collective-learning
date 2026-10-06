# Collective learning

Minimal experiments on coupled routing in mixture-of-experts Transformers.

The router is

```text
p0 = softmax(Q h)
C  = Q M Q^T
p  = softmax(Q h + lambda C p0)
```

with a learned low-rank symmetric metric `M = U diag(m) U^T`. The baseline is `lambda = 0`, so baseline and coupled runs have the same architecture and parameter count.

The workflow is split between the GPU machine and the manuscript machine.

## Minerva: computation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-compute.txt

python3 train.py --smoke
./experiment.sh
```

This writes the compact canonical outputs

```text
results/lambda0.json
results/lambda1.json
```

Commit and push those files from Minerva.

## Mac: figures and manuscript

```bash
git pull
pip install -r requirements-plot.txt
./build.sh
```

`build.sh` reads the tracked result files, generates `draft/figures/fig1.pdf` and `fig1.csv`, then compiles `draft/main.pdf`. Generated figures and LaTeX artifacts are ignored by Git.
