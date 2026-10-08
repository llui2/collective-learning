# Collective learning

Baseline reproduction of [Arola-Fernández and Lacasa, *Effective theory of collective deep learning*](https://doi.org/10.1103/PhysRevResearch.6.L042040), *Physical Review Research* **6**, L042040 (2024). The retained MNIST results are three **short** nonadiabatic seeds, not the complete published protocol.

## Setup and baseline

Requires Python 3.11+, LaTeX and latexmk:

    python3 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt
    ./build.sh

The build runs deterministic tests, regenerates two baseline figures from tracked data, and compiles `draft/main.pdf`. It never launches training or pushes to GitHub.

Recalculate baselines explicitly using `./experiment.sh theory` (expensive) or `DEVICE=cuda ./experiment.sh mnist`. Use `./setup-minerva.sh` for the Minerva environment.

## Microscopic allocation experiment

    ./experiment.sh microscopic --smoke
    ./experiment.sh microscopic

A finite-time test of learning allocation, without an adaptive effective equation. Four deep-linear networks of width two learn a shared identity mapping on two orthogonal input tasks; each has sufficient capacity for both tasks. Learning effort is normalized and divided as `a[i]` versus `1-a[i]`. The original parameter diffusion couples the learners. The training horizon is deliberately finite.

Four conditions share initial weights and training examples:

- `uniform`: each learner studies both tasks equally.
- `specialists`: half study only task 1, half only task 2.
- `frozen`: slightly heterogeneous, fixed allocation.
- `evolving`: starts identically to `frozen`; one allocation mutation per round is retained only if a same-data counterfactual rollout reduces **mean individual validation loss**.

This last selection rule is a computational assumption, not a model of decentralized adaptation. It consumes additional counterfactual compute. Diversity is *not* explicitly rewarded.

By default the experiment compares seven coupling values across three paired seeds. `results/microscopic_sweep.json` includes the complete histories and unseen test results for every learner and task. `results/microscopic_sweep.pdf` shows mean individual test loss and, for fixed specialists, **studied versus unstudied task loss** (mean and standard deviation across seeds). Files are written after each completed run and ignored by Git; initial seed and training budget are matched across conditions.

Override the sweep and budget using, for example,

    ./experiment.sh microscopic --couplings 0,0.3,1,2 --seeds 0,1,2,3,4 --rounds 25

The primary diagnostic of collective learning is whether a *specialist learner* improves on the task receiving **zero local training weight** when coupling is enabled. Ensemble prediction, allocation diversity and functional diversity are recorded separately; ensemble gains alone do not constitute transferred individual knowledge. A useful specialization effect would additionally have to improve mean individual generalization relative to the uniform and frozen controls. Results are a numerical pilot, not a phase diagram or derived adaptive theory.

## Files

- `src/collective_learning/core.py`: original coupled SGD with optional weighted local loss.
- `src/collective_learning/theory.py`: validated effective-theory baseline.
- `src/collective_learning/mnist.py`: short coupled-MNIST baseline.
- `src/collective_learning/microscopic.py`: finite-time microscopic allocation sweep.
- `draft/main.tex`: original theory, baseline figures and the still-open weighted effective reduction.
- `results/theory_baseline.npz` and `results/mnist_quick_seed*.json`: retained baseline data.
- `tests/`: deterministic checks of microscopic dynamics and coupling-mediated transfer.
