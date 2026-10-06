# Current draft

The current model modifies only the MoE router.

For token representation `h`, the ordinary router is

`p0 = softmax(Q h)`.

We introduce a learned expert interaction

`C = Q M Q^T`

with a low-rank symmetric metric `M = U diag(m) U^T`, and use one coupled correction

`p = softmax(Q h + lambda C p0)`.

The baseline is exactly `lambda = 0`. Both conditions contain the same parameters, so the first experiment isolates whether the interaction changes learning rather than whether a larger model helps.

The first test uses a small decoder-only Transformer on WikiText-2 with byte-level language modeling. The code records validation loss together with routing entropy, expert-load imbalance, `KL(p || p0)`, and the interaction norm.

Training is run separately on Minerva with `./experiment.sh`, which produces `results/lambda0.json` and `results/lambda1.json`. The local `./build.sh` reads those files, regenerates `draft/figures/fig1.pdf`, and compiles the manuscript.
