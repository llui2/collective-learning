# Collective learning

Minimal experiments on coupled routing in mixture-of-experts Transformers.

The router is

```text
p0 = softmax(Q h)
C  = Q M Q^T
p  = softmax(Q h + lambda C p0)
```

with a learned low-rank symmetric metric `M = U diag(m) U^T`. The baseline is `lambda = 0`, so baseline and coupled runs have the same architecture and parameter count.

The first experiment is a small causal Transformer trained on WikiText-2 with four experts and top-2 routing. See `draft/main.tex` for the model and measurements.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 train.py --smoke
bash experiment.sh
```

Runs are written to `runs/` and ignored by Git. Build the research note with `bash build.sh`.
