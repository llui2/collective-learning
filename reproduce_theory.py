"""Reproduce the released effective-theory experiment."""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp


DEPTHS = (0, 1, 2)


def sigma_grid():
    return np.concatenate(
        (
            np.logspace(-2.2, -1, 15),
            np.logspace(-1, 1, 30),
            np.logspace(1, 2.2, 15),
        )
    )


def dynamics(_, m, depth, gamma, sigma, disorder):
    return (
        disorder * m**depth
        - m ** (2 * depth + 1)
        - gamma * m
        + sigma * (m.mean() - m)
    )


def effective_loss(m, depth, disorder):
    return m ** (2 * (depth + 1)) - 2 * disorder.mean() * m ** (depth + 1)


def run_one(task):
    (
        d_idx,
        depth,
        run_idx,
        sigmas,
        gamma,
        time,
        samples,
        disorder,
        initial,
        state_ad,
    ) = task

    t_eval = np.linspace(0, time, samples)
    half = samples // 2
    magnetization = np.zeros(len(sigmas))
    loss = np.zeros(len(sigmas))
    magnetization_ad = np.zeros(len(sigmas))
    loss_ad = np.zeros(len(sigmas))

    for s_idx, sigma in enumerate(sigmas):
        sol = solve_ivp(
            dynamics,
            (0, time),
            initial,
            args=(depth, gamma, sigma, disorder),
            method="RK45",
            t_eval=t_eval,
        )
        sol_ad = solve_ivp(
            dynamics,
            (0, time),
            state_ad,
            args=(depth, gamma, sigma, disorder),
            method="RK45",
            t_eval=t_eval,
        )
        state_ad = sol_ad.y[:, -1].copy()

        tail = sol.y[:, half:]
        tail_ad = sol_ad.y[:, half:]
        magnetization[s_idx] = abs(tail.mean())
        magnetization_ad[s_idx] = abs(tail_ad.mean())
        loss[s_idx] = effective_loss(tail, depth, disorder).mean()
        loss_ad[s_idx] = effective_loss(tail_ad, depth, disorder).mean()

    return d_idx, run_idx, magnetization, loss, magnetization_ad, loss_ad


def run(args):
    sigmas = sigma_grid()
    if args.smoke:
        sigmas = sigmas[::15]
        args.runs = 2
        args.units = 40
        args.time = 3.0
        args.samples = 40

    rng = np.random.default_rng(args.seed)
    tasks = []
    for d_idx, depth in enumerate(DEPTHS):
        for run_idx in range(args.runs):
            disorder = rng.normal(0.0, args.disorder_std, args.units)
            initial = rng.uniform(
                -args.bound + args.bias,
                args.bound + args.bias,
                args.units,
            )
            state_ad = rng.uniform(
                -args.bound + args.bias,
                args.bound + args.bias,
                args.units,
            )
            tasks.append(
                (
                    d_idx,
                    depth,
                    run_idx,
                    sigmas,
                    args.gamma,
                    args.time,
                    args.samples,
                    disorder,
                    initial,
                    state_ad,
                )
            )

    shape = (len(DEPTHS), args.runs, len(sigmas))
    magnetization = np.zeros(shape)
    loss = np.zeros(shape)
    magnetization_ad = np.zeros(shape)
    loss_ad = np.zeros(shape)

    jobs = min(args.jobs, len(tasks))
    print(
        f"effective theory: {len(tasks)} runs, {len(sigmas)} sigma values, "
        f"{jobs} workers",
        flush=True,
    )

    completed = 0
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(run_one, task) for task in tasks]
        for future in as_completed(futures):
            d_idx, run_idx, m, l, m_ad, l_ad = future.result()
            magnetization[d_idx, run_idx] = m
            loss[d_idx, run_idx] = l
            magnetization_ad[d_idx, run_idx] = m_ad
            loss_ad[d_idx, run_idx] = l_ad
            completed += 1

            if completed == 1 or completed % args.progress_every == 0 or completed == len(tasks):
                print(
                    f"effective theory: {completed}/{len(tasks)} runs complete",
                    flush=True,
                )

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
    print(f"wrote {out}", flush=True)


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
    p.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument("--progress-every", type=int, default=10)
    p.add_argument("--output", default="results/theory_baseline.npz")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
