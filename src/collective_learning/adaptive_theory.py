"""Effective adaptive-learning dynamics and symmetry breaking."""

import argparse
from pathlib import Path

import numpy as np


VERSION = 3


def simulate(args):
    sigmas = np.logspace(-2.0, 1.0, args.sigma_points)
    nus = np.asarray(args.mutation_rates, dtype=float)

    rng = np.random.default_rng(args.seed)
    shape = (len(nus), len(sigmas), args.runs, args.units, args.components)

    a = np.ones(shape) / args.components
    a += args.noise * rng.normal(size=shape)
    a = np.clip(a, 1e-12, None)
    a /= a.sum(axis=-1, keepdims=True)

    m_star = args.learning_rate / (
        args.learning_rate + args.relaxation * args.components
    )
    m = m_star + args.noise * rng.normal(size=shape)

    sigma = sigmas[None, :, None, None, None]
    nu = nus[:, None, None, None, None]

    steps = int(args.time / args.dt)
    tail_start = int(args.tail_fraction * steps)
    stride = max(1, int(args.sample_interval / args.dt))

    genotype_sum = np.zeros((len(nus), len(sigmas), args.runs))
    phenotype_sum = np.zeros_like(genotype_sum)
    samples = 0

    for step in range(steps):
        m_bar = m.mean(axis=3)
        m_bar_minus = (
            args.units * m_bar[..., None, :] - m
        ) / (args.units - 1)

        fitness = (
            0.5 * (1.0 - m_bar_minus) ** 2
            - 0.5 * (1.0 - m_bar[..., None, :]) ** 2
        )

        dm = (
            args.learning_rate * a * (1.0 - m)
            - args.relaxation * m
            + sigma * (m_bar[..., None, :] - m)
        )

        mean_fitness = (a * fitness).sum(axis=-1, keepdims=True)
        da = args.epsilon * (
            a * (fitness - mean_fitness)
            + nu * (1.0 / args.components - a)
        )

        m += args.dt * dm
        a += args.dt * da
        a = np.clip(a, 1e-12, None)
        a /= a.sum(axis=-1, keepdims=True)

        if step >= tail_start and (step - tail_start) % stride == 0:
            a_bar = a.mean(axis=3, keepdims=True)
            genotype_sum += np.mean(
                np.sum((a - a_bar) ** 2, axis=-1),
                axis=3,
            )

            m_bar_population = m.mean(axis=3, keepdims=True)
            phenotype_sum += np.mean(
                np.sum((m - m_bar_population) ** 2, axis=-1),
                axis=3,
            )
            samples += 1

    genotype = genotype_sum / samples
    phenotype = phenotype_sum / samples

    # Complete interaction graph: lambda_2 = N.
    lambda2 = float(args.units)
    base = args.learning_rate / args.components + args.relaxation
    gain = (
        args.learning_rate
        * (1.0 - m_star) ** 2
        / (args.components * (args.units - 1))
    )
    sigma_critical = (
        args.units / lambda2
        * (gain / nus - base)
    )

    return {
        "version": np.asarray(VERSION),
        "sigmas": sigmas,
        "mutation_rates": nus,
        "genotype_mean": genotype.mean(axis=2),
        "genotype_std": genotype.std(axis=2),
        "phenotype_mean": phenotype.mean(axis=2),
        "phenotype_std": phenotype.std(axis=2),
        "sigma_critical": sigma_critical,
        "m_star": np.asarray(m_star),
        "units": np.asarray(args.units),
        "components": np.asarray(args.components),
        "learning_rate": np.asarray(args.learning_rate),
        "relaxation": np.asarray(args.relaxation),
        "epsilon": np.asarray(args.epsilon),
        "dt": np.asarray(args.dt),
        "time": np.asarray(args.time),
        "runs": np.asarray(args.runs),
    }


def run(args):
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **simulate(args))
    print(f"wrote {out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--units", type=int, default=10)
    p.add_argument("--components", type=int, default=3)
    p.add_argument("--runs", type=int, default=8)
    p.add_argument("--sigma-points", type=int, default=20)
    p.add_argument(
        "--mutation-rates",
        type=float,
        nargs="+",
        default=(0.002, 0.004, 0.008),
    )
    p.add_argument("--learning-rate", type=float, default=1.2)
    p.add_argument("--relaxation", type=float, default=0.25)
    p.add_argument("--epsilon", type=float, default=0.2)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--time", type=float, default=12000.0)
    p.add_argument("--tail-fraction", type=float, default=0.8)
    p.add_argument("--sample-interval", type=float, default=1.0)
    p.add_argument("--noise", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output", default="results/adaptive_theory.npz")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
