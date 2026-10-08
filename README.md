# Collective learning

This repository now develops a physical description of **router–expert co-learning** in mixture-of-experts (MoE) architectures. The earlier [Arola-Fernández--Lacasa collective-learning baseline](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.6.L042040) and fixed/adaptive allocation pilots remain in source control as independent comparison models.

## Physical MoE starting point

The [main draft](draft/main.tex) studies two token populations with opposing target slopes, two trainable linear experts and a learned softmax router. There are distinct collective modes for **conditional specialization** (different tokens prefer different experts) and **load collapse** (all tokens prefer the same expert). The objective is a differentiable training loss: task error, expert/router regularization and an optional aggregate load penalty. No diversity reward or external task allocation is imposed.

The reduced equations have a tractable symmetric stationary state. Its expert-contrast/router-contrast mode becomes unstable when

\[
T < T_c = (8\kappa_e\kappa_r)^{-1/2},
\]

where \(T\) is router temperature and \(\kappa_e,\kappa_r\) are regularization strengths. The collapse mode is damped by the load-balancing penalty. Numerical gradient flow and checks against automatic differentiation verify the calculation.

**Limitations:** this is a *soft*, two-expert **reduced model**, not a sparse top-k transformer, and scalar experts cannot individually solve both conflicting token populations. The calculation establishes a mechanism and testable stability predictions, not a result about production LLMs. The accompanying token-level feed-forward experiment tests how much remains true when experts are expressive and routing is sparse. It is not a full transformer; learning token representations and larger expert populations remain open.

## Reproduce the reduced model

Requires Python 3.11+, NumPy, PyTorch (for tests), and Matplotlib. LaTeX and `latexmk` are optional locally (required in CI):

    python3 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt
    ./build.sh

The build runs tests and generates the two-panel MoE figure. If `latexmk` is installed, it also compiles the two manuscript PDFs; otherwise it reports that PDF compilation was skipped without failing the numerical build. To require manuscript compilation, use `REQUIRE_LATEX=1 ./build.sh`. On Ubuntu, the optional packages are `latexmk`, `texlive-latex-recommended`, and `texlive-fonts-recommended`. It does not launch large neural experiments or push to GitHub. Run the separate reduced-model experiment with:

    ./experiment.sh moe --smoke
    ./experiment.sh moe

This saves `results/moe_reduced.json` and `results/moe_reduced.pdf`. To vary the stability boundary or timescales, adjust `--temperatures`, `--expert-rate`, `--router-rate`, `--expert-decay`, `--router-decay`, `--balance`, `--seeds`, and `--steps`. Each trajectory reports conditional routing contrast, load imbalance, expert contrast, expert-router alignment and data loss. Numerical results are ignored by Git unless explicitly added.

## Token-level MoE training

A small PyTorch MoE feed-forward layer provides the next validation level:

    ./experiment.sh moe-layer --smoke
    ./experiment.sh moe-layer

Two trainable ReLU experts receive token representations containing both the
population label and a continuous feature. Unlike the scalar reduction,
**either expert has the representational capacity to learn both populations**.
We compare dense soft routing (top-2) and selected-gate top-1 routing, with
a configurable expert capacity factor and soft load-balancing penalty.

The experiment records conditional routing preference, soft and hard expert
loads, overflowed tokens and the *alignment of routing with cross-population
expert skill*. Mere route differentiation is not counted as functional
specialization. Results are written to `results/moe_layer.json` and
`results/moe_layer.pdf`. Top-1 dispatch is masked after evaluating all
experts, for interpretability rather than computational efficiency;
the example is not an end-to-end transformer and does not learn token
representations.

## Local PDF build

On your Mac, with `latexmk` already installed:

    git pull --ff-only
    ./build.sh

This regenerates the MoE and Arola figures and produces two PDFs:

- `draft/main.pdf`: current MoE physical theory.
- `draft/arola.pdf`: one-page explanation and a two-panel plot comparing allocation differentiation with actual learning benefit.

The Arola note uses committed paired runs to compare allocation polarization and individual learning benefit; it requires no retraining. The original
experiments and their datasets remain available separately. The baseline
figure scripts can still be run independently if needed.

When `latexmk` is unavailable (e.g. on Minerva), the build runs Python
tests and the MoE figure but skips manuscript compilation. To build
one PDF independently after generating figures, run
`cd draft && latexmk -pdf -interaction=nonstopmode main.tex`
or replace `main.tex` with `arola.tex`.

## Earlier experiments

- `./experiment.sh theory`: Arola effective-theory reproduction.
- `DEVICE=cuda ./experiment.sh mnist`: short MNIST baseline (not the complete published protocol).
- `./experiment.sh microscopic`: fixed-specialization learning-speed pilot.
- `./experiment.sh adaptive`: centralized mutation-selection pilot, which did **not** discover beneficial specialization.

Their implementations and tests remain available. The Arola baseline and specialization results are summarized together in `draft/arola.tex`; `draft/main.tex` is dedicated to learned routing dynamics. Original baseline figure scripts remain in `draft/figures/fig1.py` and `fig2.py`. On Minerva, `./setup-minerva.sh` configures the neural baseline environment.

## Key files

- `src/collective_learning/moe.py`: exact expected loss, gradients, stability matrix, trajectories and plotting.
- `src/collective_learning/moe_layer.py`: trainable token-level router, feed-forward experts, top-k masks and capacity.
- `tests/test_moe.py`: tests against PyTorch autograd, finite-difference Jacobian and symmetry/collapse modes.
- `draft/main.tex`, `draft/refs.bib`: current MoE mathematical starting point and references.
- `draft/arola.tex`: one-section model, allocation and learning outcomes, and limitation.
- `draft/figures/arola.py`: plot from committed paired results.
- `draft/figures/moe.py`: main two-panel numerical figure.
- `src/collective_learning/core.py`, `theory.py`, `mnist.py`, `microscopic.py`, `adaptive.py`: earlier models.
