# Collective learning

Reproduction of [Arola-Fernández and Lacasa, *Effective theory of collective deep learning*](https://doi.org/10.1103/PhysRevResearch.6.L042040), *Physical Review Research* **6**, L042040 (2024). The tracked MNIST baseline comprises three short nonadiabatic seeds, not the full published training horizon.

## Setup

Requires Python 3.11+, LaTeX and latexmk:

    python3 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt
    ./build.sh

The build checks the code, regenerates two baseline figures from retained data and compiles `draft/main.pdf`. It does not train, pull or push. Recalculate baseline experiments explicitly with `./experiment.sh theory` (expensive) or `DEVICE=cuda ./experiment.sh mnist`. On Minerva, use `./setup-minerva.sh` to install CUDA-compatible PyTorch.

## Finite-time microscopic experiment

    ./experiment.sh microscopic --smoke
    ./experiment.sh microscopic --jobs 12

This is a controlled test of a possible *transient* specialization advantage. Learners share the deep-linear target $y=x$ on two orthogonal input tasks, and every learner can represent both tasks. There are four coupled networks with independent initial weights, shared across conditions. No adaptive effective equation is assumed.

Only the training allocation changes:

- **Uniform** learners divide their local effort equally between tasks.
- **Specialists** train on exactly one task each (two learners per task).

Both receive identical minibatches, initial weights, SGD step counts and total local loss weight. Parameters interact through the original simultaneous diffusion law. The comparison uses each learner's **mean individual generalization error**, not an averaged ensemble prediction. We also record specialists' errors on the studied and unstudied tasks; the latter checks coupling-mediated transfer against $\sigma=0$.

The default exploratory scan varies hidden linear layers `D=0,1,2`, initial parameter scales `0.03,0.1,0.3`, coupling `0,0.2,1,3`, and two paired seeds. Every trial records eleven or more logarithmically spaced checkpoints over 400 SGD updates, including $t=0$ and $t=400$. The difference

    advantage(t) = individual_loss_uniform(t) - individual_loss_specialists(t)

is positive exactly when fixed specialization improves mean individual generalization. The experiment generates one incremental `results/microscopic_timescales.json` containing the full cross-task histories and one two-panel `results/microscopic_timescales.pdf` showing a representative slice (middle depth and initial scale). The plots show mean and standard deviation across seeds; results at other depths/scales remain in JSON.

Independent trials can use parallel CPU workers on Minerva with `--jobs 12` (a single worker is the default); use `--jobs 1` for CUDA. A custom, longer scan can use:

    ./experiment.sh microscopic --jobs 12 --depths 1,2 --scales 0.03,0.1 --couplings 0,0.3,1,2 --seeds 0,1,2 --steps 800

The JSON and PDF are ignored by Git. To share only this experiment for analysis:

    git add -f results/microscopic_timescales.json
    git commit -m "Add finite-time specialization scan"
    git push origin main

This is a diagnostic pilot. The scan is exploratory and can select spurious apparent advantages across many conditions. Finite horizons and deep-linear initialization matter, and task weighting also changes stochastic-gradient variance. **Evolution of allocations remains disabled** until there is a reproducible fixed-specialization advantage. Earlier evolutionary pilots are retained in Git history, not the working experiment.

## Structure

- `src/collective_learning/core.py`: baseline coupled SGD, optional weighted local losses.
- `src/collective_learning/theory.py`: baseline effective equations.
- `src/collective_learning/mnist.py`: short neural MNIST reproduction.
- `src/collective_learning/microscopic.py`: time/depth/initialization/coupling scan.
- `draft/main.tex`: baseline manuscript and open effective-reduction question.
- `results/theory_baseline.npz`, `results/mnist_quick_seed*.json`: retained baseline data.
- `tests/`: deterministic learning, coupling and experiment checks.
