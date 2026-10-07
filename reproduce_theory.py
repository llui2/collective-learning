"""Reproduce the released effective-theory experiment (paper Fig. 2a-b)."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


DEPTHS = (0, 1, 2)


def sigma_grid():
    a = np.logspace(-2.2, -1, 15)
    b = np.logspace(-1, 1, 30)
    c = np.logspace(1, 2.2, 15)
    return np.concatenate((a, b, c))


def dynamics(_, m, depth, gamma, sigma, disorder):
    coupling = sigma * (m.mean() - m)
    return (
        disorder * m**depth
        - m ** (2 * depth + 1)
        - gamma * m
        + coupling
    )


def effective_loss(m, depth, disorder):
    return m ** (2 * (depth + 1)) - 2 * disorder.mean() * m ** (depth + 1)


def run(args):
    rng = np.random.default_rng(args.seed)
    sigmas = sigma_grid()
    depths = DEPTHS

    if args.smoke:
        sigmas = sigmas[::15]
        args.runs = 2
        args.units = 40
        args.time = 3.0
        args.samples = 40

    shape = (len(depths), args.runs, len(sigmas))
    magnetization = np.zeros(shape)
    loss = np.zeros(shape)
    magnetization_ad = np.zeros(shape)
    loss_ad = np.zeros(shape)
    t_eval = np.linspace(0, args.time, args.samples)
    half = len(t_eval) // 2

    for d_idx, depth in enumerate(depths):
        for run_idx in range(args.runs):
            disorder = rng.normal(0.0, args.disorder_std, args.units)
            initial = rng.uniform(-args.bound + args.bias, args.bound + args.bias, args.units)
            state_ad = rng.uniform(-args.bound + args.bias, args.bound + args.bias, args.units)

            for s_idx, sigma in enumerate(sigmas):
                sol = solve_ivp(
                    dynamics,
                    (0, args.time),
                    initial,
                    args=(depth, args.gamma, sigma, disorder),
                    method="RK45",
                    t_eval=t_eval,
                )
                sol_ad = solve_ivp(
                    dynamics,
                    (0, args.time),
                    state_ad,
                    args=(depth, args.gamma, sigma, disorder),
                    method="RK45",
                    t_eval=t_eval,
                )
                state_ad = sol_ad.y[:, -1].copy()

                tail = sol.y[:, half:]
                tail_ad = sol_ad.y[:, half:]
                magnetization[d_idx, run_idx, s_idx] = abs(tail.mean())
                magnetization_ad[d_idx, run_idx, s_idx] = abs(tail_ad.mean())
                loss[d_idx, run_idx, s_idx] = effective_loss(tail, depth, disorder).mean()
                loss_ad[d_idx, run_idx, s_idx] = effective_loss(tail_ad, depth, disorder).mean()

    return sigmas, magnetization, loss, magnetization_ad, loss_ad


def save(args, data):
    sigmas, magnetization, loss, magnetization_ad, loss_ad = data
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        sigmas=sigmas,
        depths=np.array(DEPTHS),
        magnetization=magnetization,
        loss=loss,
        magnetization_adiabatic=magnetization_ad,
        loss_adiabatic=loss_ad,
    )

    fig = Path(args.figure)
    fig.parent.mkdir(parents=True, exist_ok=True)

    x = np.log10(sigmas)
    f, ax = plt.subplots(1, 2, figsize=(8.0, 3.2))
    for d_idx, depth in enumerate(DEPTHS):
        m = magnetization[d_idx].mean(axis=0)
        m_sem = magnetization[d_idx].std(axis=0) / np.sqrt(magnetization.shape[1])
        l = loss[d_idx].mean(axis=0)
        l0 = l[0]
        ax[0].plot(x, m, label=f"$D={depth}$")
        ax[0].fill_between(x, m - m_sem, m + m_sem, alpha=0.15)
        ax[1].plot(x, l / l0, label=f"$D={depth}$")

    ax[0].set_xlabel(r"$\log_{10}\sigma$")
    ax[0].set_ylabel(r"$|\langle m\rangle|$")
    ax[1].set_xlabel(r"$\log_{10}\sigma$")
    ax[1].set_ylabel(r"$\langle L\rangle/\langle L\rangle_{\sigma_{\min}}$")
    ax[0].legend(frameon=False)
    f.tight_layout()
    f.savefig(fig)
    print(f"wrote {out}")
    print(f"wrote {fig}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=200)
    p.add_argument("--units", type=int, default=200)
    p.add_argument("--gamma", type=float, default=1e-3)
    p.add_argument("--disorder-std", type=float, default=2.0)
    p.add_argument("--bound", type=float, default=2.0)
    p.add_argument("--bias", type=float, default=0.3)
    p.add_argument("--time", type=float, default=10.0)
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="results/theory_baseline.npz")
    p.add_argument("--figure", default="draft/figures/theory_baseline.pdf")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    save(args, run(args))
