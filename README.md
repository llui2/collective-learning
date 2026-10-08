# Collective learning

Reproduction of [Arola-Fernández and Lacasa, *Effective theory of collective deep learning*](https://doi.org/10.1103/PhysRevResearch.6.L042040), *Physical Review Research* **6**, L042040 (2024). The retained MNIST baseline comprises three short nonadiabatic seeds, not the full published training protocol.

## Setup

Requires Python 3.11+, LaTeX and latexmk:

    python3 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt
    ./build.sh

The build runs tests and regenerates baseline figures and `draft/main.pdf`; it does not train, pull or push. Recalculate the expensive theory ensemble explicitly with `./experiment.sh theory`, or the short MNIST baseline with `DEVICE=cuda ./experiment.sh mnist`. On Minerva, `./setup-minerva.sh` installs CUDA-compatible PyTorch when needed.

## Finite-time specialization experiment

    ./experiment.sh microscopic --smoke
    ./experiment.sh microscopic --jobs 12

The current question is whether **fixed specialization shortens the time for each individual learner to generalize across both tasks**, and whether parameter diffusion contributes beyond the initial benefit of concentrated training. Evolution of allocations is disabled until a robust advantage is established.

The pilot uses four jointly capable, two-layer linear networks learning the shared identity teacher on two independent task classes. Two identical populations receive identical model initializations, training minibatches, total local learning budgets and SGD updates: uniform learners divide effort equally, while specialists study one task exclusively. Diffusion between corresponding parameters can transmit information on the task a specialist never studies. We evaluate every learner on both tasks, never an ensemble prediction.

The focused default tests one hidden layer, weight initialization scales 0.03 and 0.1, coupling strengths 0, 0.2, 1 and 3, and 12 paired seeds, for 600 updates. Losses are measured on an independently generated test set every 10 updates. The **predeclared thresholds are 0.2 (primary) and 0.1 (secondary)** for mean individual test loss. The first observed checkpoint at or below each threshold is the measured hitting time.

A failed threshold crossing is stored as `null` (right-censored), **not as a successful crossing at the final update**. To compare paired experiments when one run never reaches a threshold, we also report the finite-horizon *restricted speedup*
`min(T_uniform, H) - min(T_specialists, H)`, with `H=600`. Positive values indicate an earlier threshold crossing by specialists within the observed horizon; a value of zero can also mean both populations failed. Each summary includes the number of successful crossings by strategy and the number of pairs that both succeeded. Checkpoint resolution limits timing accuracy.

The experiment writes:
- `results/microscopic_speed.json`: complete individual-loss, transfer and diversity trajectories for local examination.
- `results/microscopic_speed_summary.json`: compact individual first-passage times, censoring, group averages, spread, restricted speedups and paired seed identifiers.
- `results/microscopic_speed.pdf`: two-panel diagnostic (learning curves and restricted speedup versus coupling).

You can adjust the replication protocol explicitly:

    ./experiment.sh microscopic --jobs 12 --seeds 0,1,2,3 --steps 800 --check-every 5

**Do not select thresholds post hoc based on the results.** The scanned couplings and initializations are exploratory comparisons, and 12 seeds are not enough to infer a universal phase diagram. In particular, positive specialization advantage without coupling does not establish collective transfer; off-task loss of specialists and additional benefit relative to the uncoupled condition should be considered separately.

When completed on Minerva, push only the compact summary (the full trajectories can be several MB):

    git add -f results/microscopic_speed_summary.json
    git commit -m "Add paired learning-speed results"
    git push origin main

## Adaptive task allocation

    ./experiment.sh adaptive --smoke
    ./experiment.sh adaptive --jobs 12

The positive fixed-specialization result motivates a new **mutation-selection**
experiment, separate from the original fixed-allocation replication.
Four depth-one, width-two deep-linear learners study two orthogonal tasks,
with 400 SGD updates per experiment. Default conditions compare coupling
`0, 0.2, 1, 3` at initialization scale `0.03`, with proposal intervals of
20 or 60 SGD updates and 12 paired random seeds.

Five conditions share independent initial networks, local SGD budgets, and
minibatches: uniform study, fixed complementary specialists, frozen weak
heterogeneity, neutral mutation drift, and adaptive mutation-selection.
The last three begin from the same weakly heterogeneous allocation.
At each adaptation round a single allocation mutates, without rewarding
diversity. **The evolving condition selects the mutation only if its
counterfactual rollout has lower population-mean individual validation
loss** than the unchanged rollout. Both use identical training examples,
neural updates and initial parameters. This is an *oracle-guided global
selection rule* using extra validation data and extra computation. It is
not a decentralized model of self-organization.

The key readouts are first-passage times to individual test losses 0.2
and 0.1, allocation polarization (0 = generalists, 1 = fully polarized),
allocation variance, task coverage and functional diversity. Results also
include neutral drift, which can develop allocation diversity without
fitness-based selection. For each selection window, threshold times are
only resolved to that window length. We must check that evolving
allocations beat uniform and frozen allocations, and polarize more than
neutral drift, before claiming beneficial division of labor.

The command writes full histories to `results/microscopic_adaptive.json`,
a compact shareable `results/microscopic_adaptive_summary.json`, and a
two-panel PDF. Run on Minerva, then push **only the summary**:

    git add -f results/microscopic_adaptive_summary.json
    git commit -m "Add adaptive allocation experiment results"
    git push origin main

## Files

- `src/collective_learning/core.py`: baseline diffusive coupled SGD and weighted local loss.
- `src/collective_learning/theory.py`: original effective equations.
- `src/collective_learning/mnist.py`: short MNIST baseline.
- `src/collective_learning/microscopic.py`: focused learning-speed and coupling comparison.
- `src/collective_learning/adaptive.py`: globally selected allocations and neutral-drift control.
- `draft/main.tex`: baseline manuscript and open weighted-effective-theory question.
- `results/theory_baseline.npz`, `results/mnist_quick_seed*.json`: retained baseline data.
- `tests/`: learning and reproducibility checks.
