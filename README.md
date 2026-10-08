# Collective learning

This repository now develops a physical description of **router–expert co-learning** in mixture-of-experts (MoE) architectures. The earlier [Arola-Fernández--Lacasa collective-learning baseline](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.6.L042040) and fixed/adaptive allocation pilots remain in source control as independent comparison models.

## Physical MoE starting point

The [main draft](draft/main.tex) studies two token populations with opposing target slopes, two trainable linear experts and a learned softmax router. There are distinct collective modes for **conditional specialization** (different tokens prefer different experts) and **load collapse** (all tokens prefer the same expert). The objective is a differentiable training loss: task error, expert/router regularization and an optional aggregate load penalty. No diversity reward or external task allocation is imposed.

The reduced equations have a tractable symmetric stationary state. Its expert-contrast/router-contrast mode becomes unstable when

\[
T < T_c = (8\kappa_e\kappa_r)^{-1/2},
\]

where \(T\) is router temperature and \(\kappa_e,\kappa_r\) are regularization strengths. The collapse mode is damped by the load-balancing penalty. Numerical gradient flow and checks against automatic differentiation verify the calculation.

**Limitations:** this is a *soft*, two-expert **reduced model**, not a sparse top-k transformer, and scalar experts cannot individually solve both conflicting token populations. The calculation establishes a mechanism and testable stability predictions, not a result about production LLMs. The next stage is an actual trainable MoE feed-forward layer with token-level top-k selection, capacity effects and diagnostics for functional specialization.

## Reproduce the reduced model

Requires Python 3.11+, NumPy, PyTorch (for tests), Matplotlib and LaTeX/latexmk:

    python3 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt
    ./build.sh

The build runs tests, generates the two-panel MoE figure and compiles `draft/main.pdf`. It does not launch the large neural experiments or push to GitHub. Run the separate reduced-model experiment with:

    ./experiment.sh moe --smoke
    ./experiment.sh moe

This saves `results/moe_reduced.json` and `results/moe_reduced.pdf`. To vary the stability boundary or timescales, adjust `--temperatures`, `--expert-rate`, `--router-rate`, `--expert-decay`, `--router-decay`, `--balance`, `--seeds`, and `--steps`. Each trajectory reports conditional routing contrast, load imbalance, expert contrast, expert-router alignment and data loss. Numerical results are ignored by Git unless explicitly added.

## Earlier experiments

- `./experiment.sh theory`: Arola effective-theory reproduction.
- `DEVICE=cuda ./experiment.sh mnist`: short MNIST baseline (not the complete published protocol).
- `./experiment.sh microscopic`: fixed-specialization learning-speed pilot.
- `./experiment.sh adaptive`: centralized mutation-selection pilot, which did **not** discover beneficial specialization.

Their implementations and tests remain available. The original Arola-focused manuscript is preserved as `draft/arola_baseline.tex`; the current `draft/main.tex` is dedicated to learned routing dynamics. Original figure generation scripts remain in `draft/figures/fig1.py` and `fig2.py`. On Minerva, `./setup-minerva.sh` configures the neural baseline environment.

## Key files

- `src/collective_learning/moe.py`: exact expected loss, gradients, stability matrix, trajectories and plotting.
- `tests/test_moe.py`: tests against PyTorch autograd, finite-difference Jacobian and symmetry/collapse modes.
- `draft/main.tex`, `draft/refs.bib`: mathematical starting point and references.
- `draft/figures/moe.py`: main two-panel numerical figure.
- `src/collective_learning/core.py`, `theory.py`, `mnist.py`, `microscopic.py`, `adaptive.py`: earlier models.
