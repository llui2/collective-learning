"""Adaptive extension of the collective-learning effective theory."""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from tqdm.auto import tqdm


VERSION = 7


def theory_parameters(args, nus):
    if args.drive <= args.relaxation:
        raise ValueError("drive must exceed relaxation for the positive D=1 symmetric state")

    m_star = np.sqrt(args.drive - args.relaxation)

    # D=1 Arola local dynamics:
    # f(m,a) = -m^3 + drive * K * a * m - relaxation * m.
    # At a=1/K, -df/dm = 2 m_*^2.
    restoring = 2.0 * m_star**2

    # Complete interaction graph: lambda_2 = N.
    gain = (
        args.drive
        * m_star
        * (1.0 - m_star)
        / (args.units - 1)
    )
    sigma_critical = gain / nus - restoring
    return m_star, restoring, sigma_critical


def sigma_grid(args):
    return np.logspace(-2.0, 1.0, args.sigma_points)


def rhs(_, state, runs, units, components, drive, relaxation, epsilon, nu, sigma):
    size = runs * units * components
    m = state[:size].reshape(runs, units, components)
    a = state[size:].reshape(runs, units, components)

    m_bar = m.mean(axis=1)
    m_bar_minus = (units * m_bar[:, None, :] - m) / (units - 1)

    fitness = (
        0.5 * (1.0 - m_bar_minus) ** 2
        - 0.5 * (1.0 - m_bar[:, None, :]) ** 2
    )

    dm = (
        -m**3
        + drive * components * a * m
        - relaxation * m
        + sigma * (m_bar[:, None, :] - m)
    )

    mean_fitness = (a * fitness).sum(axis=-1, keepdims=True)
    da = epsilon * (
        a * (fitness - mean_fitness)
        + nu * (1.0 / components - a)
    )

    return np.concatenate((dm.ravel(), da.ravel()))


def observables(solution, runs, units, components):
    size = runs * units * components
    samples = solution.shape[1]

    m = solution[:size].T.reshape(samples, runs, units, components)
    a = solution[size:].T.reshape(samples, runs, units, components)

    a_bar = a.mean(axis=2, keepdims=True)
    genotype = np.mean(
        np.sum((a - a_bar) ** 2, axis=-1),
        axis=2,
    )

    m_bar = m.mean(axis=2, keepdims=True)
    phenotype = np.mean(
        np.sum((m - m_bar) ** 2, axis=-1),
        axis=2,
    )

    # The objective used in the marginal-contribution fitness.
    collective_loss = 0.5 * np.sum(
        (1.0 - m_bar[:, :, 0, :]) ** 2, axis=-1
    )

    # Sum of learner-level allocation/competence covariances over components.
    # Averaging the fast equation eliminates the diffusive term, leaving
    # + drive*K * covariance as an explicit contribution to mean growth.
    covariance = np.sum(
        np.mean((a - a_bar) * (m - m_bar), axis=2), axis=-1
    )

    return (
        genotype.mean(axis=0),
        phenotype.mean(axis=0),
        collective_loss.mean(axis=0),
        covariance.mean(axis=0),
    )


def run_point(task):
    nu_idx, sigma_idx, nu, sigma, config = task

    runs = config["runs"]
    units = config["units"]
    components = config["components"]

    seed = np.random.SeedSequence(
        [config["seed"], nu_idx, sigma_idx]
    )
    rng = np.random.default_rng(seed)

    a = np.ones((runs, units, components)) / components
    a += config["noise"] * rng.normal(size=a.shape)
    a = np.clip(a, 1e-12, None)
    a /= a.sum(axis=-1, keepdims=True)

    m = config["m_star"] + config["noise"] * rng.normal(size=a.shape)
    state = np.concatenate((m.ravel(), a.ravel()))

    t_start = 0.0
    t_stop = config["time"]
    last_solution = None
    residual = np.inf

    while True:
        tail_start = t_start + config["tail_fraction"] * (t_stop - t_start)
        t_eval = np.linspace(tail_start, t_stop, config["tail_samples"])

        sol = solve_ivp(
            rhs,
            (t_start, t_stop),
            state,
            args=(
                runs,
                units,
                components,
                config["drive"],
                config["relaxation"],
                config["epsilon"],
                nu,
                sigma,
            ),
            method="DOP853",
            rtol=config["rtol"],
            atol=config["atol"],
            t_eval=t_eval,
        )
        if not sol.success:
            raise RuntimeError(sol.message)

        last_solution = sol.y
        state = sol.y[:, -1]

        derivative = rhs(
            t_stop,
            state,
            runs,
            units,
            components,
            config["drive"],
            config["relaxation"],
            config["epsilon"],
            nu,
            sigma,
        )
        residual = float(np.max(np.abs(derivative)))

        if residual <= config["steady_tol"] or t_stop >= config["max_time"]:
            break

        t_start = t_stop
        t_stop = min(t_stop + config["extension"], config["max_time"])

    genotype, phenotype, collective_loss, covariance = observables(
        last_solution,
        runs,
        units,
        components,
    )

    size = runs * units * components
    final_a = state[size:].reshape(runs, units, components)
    simplex_error = float(
        np.max(np.abs(final_a.sum(axis=-1) - 1.0))
    )

    return (
        nu_idx,
        sigma_idx,
        genotype,
        phenotype,
        collective_loss,
        covariance,
        residual,
        simplex_error,
        t_stop,
    )


def simulate(args):
    nus = np.asarray(args.mutation_rates, dtype=float)
    m_star, restoring, sigma_critical = theory_parameters(args, nus)
    sigmas = sigma_grid(args)

    config = {
        "runs": args.runs,
        "units": args.units,
        "components": args.components,
        "drive": args.drive,
        "relaxation": args.relaxation,
        "epsilon": args.epsilon,
        "noise": args.noise,
        "seed": args.seed,
        "m_star": m_star,
        "time": args.time,
        "max_time": args.max_time,
        "extension": args.extension,
        "tail_fraction": args.tail_fraction,
        "tail_samples": args.tail_samples,
        "rtol": args.rtol,
        "atol": args.atol,
        "steady_tol": args.steady_tol,
    }

    tasks = [
        (nu_idx, sigma_idx, float(nu), float(sigma), config)
        for nu_idx, nu in enumerate(nus)
        for sigma_idx, sigma in enumerate(sigmas)
    ]

    shape = (len(nus), len(sigmas), args.runs)
    genotype = np.zeros(shape)
    phenotype = np.zeros(shape)
    collective_loss = np.zeros(shape)
    covariance = np.zeros(shape)
    residual = np.zeros((len(nus), len(sigmas)))
    simplex_error = np.zeros_like(residual)
    final_time = np.zeros_like(residual)

    jobs = min(args.jobs, len(tasks))
    print(
        f"adaptive theory: {len(sigmas)} sigma values, "
        f"{args.runs} realizations, {jobs} workers",
        flush=True,
    )

    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(run_point, task) for task in tasks]
        with tqdm(
            total=len(futures),
            desc="adaptive theory",
            unit="point",
            dynamic_ncols=True,
        ) as progress:
            for future in as_completed(futures):
                (
                    nu_idx,
                    sigma_idx,
                    g,
                    s,
                    loss,
                    cov,
                    res,
                    simplex,
                    t_stop,
                ) = future.result()

                genotype[nu_idx, sigma_idx] = g
                phenotype[nu_idx, sigma_idx] = s
                collective_loss[nu_idx, sigma_idx] = loss
                covariance[nu_idx, sigma_idx] = cov
                residual[nu_idx, sigma_idx] = res
                simplex_error[nu_idx, sigma_idx] = simplex
                final_time[nu_idx, sigma_idx] = t_stop
                progress.update(1)

    homogeneous_loss = 0.5 * args.components * (1.0 - m_star) ** 2
    collective_gain = homogeneous_loss - collective_loss

    return {
        "version": np.asarray(VERSION),
        "sigmas": sigmas,
        "mutation_rates": nus,
        "genotype_mean": genotype.mean(axis=2),
        "genotype_std": genotype.std(axis=2),
        "phenotype_mean": phenotype.mean(axis=2),
        "phenotype_std": phenotype.std(axis=2),
        "collective_loss_mean": collective_loss.mean(axis=2),
        "collective_loss_std": collective_loss.std(axis=2),
        "collective_gain_mean": collective_gain.mean(axis=2),
        "collective_gain_std": collective_gain.std(axis=2),
        "covariance_mean": covariance.mean(axis=2),
        "covariance_std": covariance.std(axis=2),
        "homogeneous_loss": np.asarray(homogeneous_loss),
        "sigma_critical": sigma_critical,
        "residual": residual,
        "simplex_error": simplex_error,
        "final_time": final_time,
        "m_star": np.asarray(m_star),
        "restoring": np.asarray(restoring),
        "units": np.asarray(args.units),
        "components": np.asarray(args.components),
        "drive": np.asarray(args.drive),
        "relaxation": np.asarray(args.relaxation),
        "epsilon": np.asarray(args.epsilon),
        "runs": np.asarray(args.runs),
        "rtol": np.asarray(args.rtol),
        "atol": np.asarray(args.atol),
    }


def run(args):
    result = simulate(args)

    print(
        "max residual:",
        f"{result['residual'].max():.2e}",
        "| max simplex error:",
        f"{result['simplex_error'].max():.2e}",
        flush=True,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **result)
    print(f"wrote {out}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--units", type=int, default=10)
    p.add_argument("--components", type=int, default=3)
    p.add_argument("--runs", type=int, default=32)
    p.add_argument("--sigma-points", type=int, default=36)
    p.add_argument(
        "--mutation-rates",
        type=float,
        nargs="+",
        default=(0.002, 0.006, 0.02),
    )
    p.add_argument("--drive", type=float, default=0.35)
    p.add_argument("--relaxation", type=float, default=0.25)
    p.add_argument("--epsilon", type=float, default=0.2)
    p.add_argument("--time", type=float, default=12000.0)
    p.add_argument("--max-time", type=float, default=30000.0)
    p.add_argument("--extension", type=float, default=6000.0)
    p.add_argument("--tail-fraction", type=float, default=0.8)
    p.add_argument("--tail-samples", type=int, default=25)
    p.add_argument("--noise", type=float, default=0.03)
    p.add_argument("--rtol", type=float, default=1e-8)
    p.add_argument("--atol", type=float, default=1e-10)
    p.add_argument("--steady-tol", type=float, default=1e-7)
    p.add_argument(
        "--jobs",
        type=int,
        default=min(28, os.cpu_count() or 1),
    )
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output", default="results/adaptive_theory.npz")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
