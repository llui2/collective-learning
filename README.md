# Collective learning

Baseline reproduction of [Arola-Fernández and Lacasa, *Effective theory of collective deep learning*](https://doi.org/10.1103/PhysRevResearch.6.L042040), *Physical Review Research* **6**, L042040 (2024).

The repository contains the published effective dynamics and a coupled MNIST neural-network implementation. The stored MNIST data cover three **short**, nonadiabatic seeds, rather than the full published training protocol. No adaptive effective theory or adaptive performance predictions are retained.

## Reproducible workflow

Requires Python 3.11+, LaTeX and `latexmk`. Install dependencies and build from tracked baseline results:

    python3 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt
    ./build.sh

`build.sh` runs unit tests, regenerates two figures from tracked results, and compiles `draft/main.pdf`. It never trains, pulls or pushes.

Explicit recalculation is handled by `experiment.sh`:

    ./experiment.sh theory   # complete theoretical ensemble; expensive
    ./experiment.sh mnist    # short MNIST protocol, seeds 0--2
    ./experiment.sh microscopic --smoke # small synthetic microscopic test (CPU)
    ./experiment.sh smoke    # original baseline checks; downloads MNIST
    ./build.sh

The experiment commands overwrite their output files. Set `THEORY_JOBS` for theory parallelism and `DEVICE=cuda` for neural experiments. On Minerva, run `./setup-minerva.sh` to install CUDA-compatible PyTorch.

## Structure

- `src/collective_learning/theory.py`: original effective equations and adiabatic/nonadiabatic protocols.
- `src/collective_learning/core.py`: coupled SGD, evaluation and optional microscopic sample weights.
- `src/collective_learning/mnist.py`: coupled learning on private MNIST classes.
- `results/`: one completed theory ensemble and three short MNIST seeds, used by the two figures.
- `draft/main.tex`: baseline equations and the mathematical question behind weighted learning.
- `draft/figures/`: one script per baseline figure, plus shared typography.
- `tests/`: deterministic invariants for the retained equations and learning updates.

The proposed next step is a **derivation** from the weighted microscopic neural loss. Such weighting changes both the input-output correlation and input covariance. No effective adaptive equations have been established.

## Microscopic allocation experiment

Run `./experiment.sh microscopic --coupling 0.6` and
`./experiment.sh microscopic --coupling 0 --output results/uncoupled.json`.
This uses a small population of **two-layer linear neural networks** learning
one common regression target on two orthogonal input classes. Each learner
receives the same examples, but its fraction `a[i]` of learning effort on
the first class may evolve. The **local loss** is sample-weighted squared
error and the **coupling** is the original simultaneous parameter diffusion.

Three conditions share initial weights and minibatches: uniform allocation,
fixed slightly heterogeneous allocation, and allocation mutation-selection.
At each generation a proposed mutation to one learner's allocation is
compared against the unchanged strategy. Both branches receive identical
training samples and an identical number of neural updates. Selection uses
the **population-average validation loss**, not a reward for diversity.
A separate test set is evaluated only after training.

The command writes `results/microscopic.json` (allocation trajectories and
cross-task loss matrices) and `results/microscopic.pdf` (validation loss and
allocation trajectories). Generated files are ignored by Git. This is a
mechanistic **pilot**, not a derived adaptive effective theory. Its selection
rule uses extra validation data and counterfactual computation; any apparent
benefit requires confirmation on the untouched test set, with multiple seeds
and matched computational budgets before interpreting it as a learning gain.
