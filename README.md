# Collective learning

Physics-first study of spontaneous differentiation in populations of coupled learning units.

The starting point is Arola-Fernández and Lacasa, *Effective theory of collective deep learning*, Phys. Rev. Research **6**, L042040 (2024). The repository now contains a clean reimplementation of their released baseline before introducing adaptive learning genotypes.

## Baseline

`collective.py` contains the microscopic coupled-SGD dynamics,

```text
theta_i <- theta_i - eta grad L_i
           + eta sigma/N sum_j q_ij (theta_j - theta_i)
```

with an optional per-sample weight argument reserved for the adaptive-allocation extension. With no sample weights it is the original baseline.

`reproduce_theory.py` reproduces the effective Ginzburg-Landau experiment used for Fig. 2(a,b).

`reproduce_mnist.py` reproduces the MNIST experiment used for Fig. 3(a-c): ten neural units, one private digit class per unit, all-to-all parameter coupling, and depths `D=0,1,2`.

The implementation is a clean reorganization of the public notebooks at `mystic-blue/collective-learning`; it is not a verbatim copy.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 reproduce_theory.py --smoke
python3 reproduce_mnist.py --smoke --device cuda

python3 reproduce_theory.py
python3 reproduce_mnist.py --device cuda
```

The full MNIST reproduction uses the paper values: batch size 32, learning rate 0.005, weight decay 0.001, 20,000 transient updates followed by 20,000 measured updates, and ten independent runs.

## Reproducibility note

The released notebook and final paper differ in two implementation details. The notebook uses PyTorch's default parameter initialization, whereas Appendix C states that parameters are initialized from a standard normal distribution. The MNIST notebook also passes `vec_sigma/N` into the pairwise diffusion update and later plots `vec_sigma/N`, while Eq. (2) denotes the prefactor as `sigma/N`. Results therefore store both `sigma_equation` and `sigma_released_plot`. The default `--init source` follows the released notebook; `--init normal` follows the initialization stated in the paper.

## Model extension

The new project replaces fixed private-data heterogeneity by a slow learning genotype. All learners will see the same environment, while genotype-dependent sample weights reshape the local gradient. The optional `sample_weights` argument in `coupled_sgd_step` is the insertion point for that dynamics.

The mathematical formulation is in `draft/main.tex`.
