"""Two-token-population, two-expert soft MoE: exact gradient-flow pilot.

This reduced model is NOT a sparse top-k transformer. Each expert is a
one-parameter linear map of a token feature z, and a learned router sees
the population label c=+/-1. Competing target slopes +/-1 force a
representational choice; no specialization reward is imposed.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    """Stable logistic used for the two conditional routing probabilities."""
    x = np.asarray(x, dtype=float)
    return np.exp(-np.logaddexp(0.0, -x))


def routing(state, temperature):
    """State is (mean expert weight, contrast, router contrast, bias)."""
    _, _, q, b = state
    classes = np.array([-1.0, 1.0])
    return sigmoid((b + q * classes) / temperature)


def observable(state, temperature):
    m, d, _, _ = state
    p = routing(state, temperature)
    c = np.array([-1.0, 1.0])
    predictions = m + (2 * p - 1) * d
    return {
        "loss": float(0.25 * np.square(predictions - c).sum()),
        "specialization": float(p[1] - p[0]),
        "load_imbalance": float(p.sum() - 1),
        "expert_contrast": float(2 * d),
        "alignment": float((p[1] - p[0]) * 2 * d),
    }


def energy_and_gradient(state, temperature, expert_decay, router_decay, balance):
    """Expected squared loss, load penalty, and exact parameter gradients.

    The gradient is with respect to (m,d,q,b); the expert update uses
    half this gradient, as m,d are coordinates of two independent w_i.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    m, d, q, b = state
    c = np.array([-1.0, 1.0])
    p = routing(state, temperature)
    g = 2 * p - 1
    error = m + g * d - c
    load = float(p.mean())
    gp = p * (1 - p) / temperature
    main = 0.25 * np.square(error).sum()
    regularization = expert_decay * (m * m + d * d) + 0.5 * router_decay * (q * q + b * b)
    penalty = 0.5 * balance * (load - 0.5) ** 2

    gm = 0.5 * error.sum() + 2 * expert_decay * m
    gd = 0.5 * np.dot(error, g) + 2 * expert_decay * d
    gq = d * np.dot(error, gp * c) + balance * (load - 0.5) * np.dot(gp, c) / 2 + router_decay * q
    gb = d * np.dot(error, gp) + balance * (load - 0.5) * gp.mean() + router_decay * b
    return float(main + regularization + penalty), np.array([gm, gd, gq, gb])


def dynamics(state, temperature, expert_rate, router_rate, expert_decay, router_decay, balance):
    _, gradient = energy_and_gradient(
        state, temperature, expert_decay, router_decay, balance
    )
    # w1=m+d, w2=m-d: Euclidean GD on (w1,w2) induces factor 1/2.
    return -np.array([
        expert_rate / 2, expert_rate / 2,
        router_rate, router_rate,
    ]) * gradient


def jacobian(temperature, expert_rate, router_rate, expert_decay, router_decay, balance):
    """Exact linearization of gradient flow at (m,d,q,b)=(0,0,0,0)."""
    t = temperature
    return np.array([
        [-expert_rate * (0.5 + expert_decay), 0, 0, 0],
        [0, -expert_rate * expert_decay, expert_rate / (4 * t), 0],
        [0, router_rate / (2 * t), -router_rate * router_decay, 0],
        [0, 0, 0, -router_rate * (router_decay + balance / (16 * t * t))],
    ])


def eigenvalues(temperature, expert_rate, router_rate, expert_decay, router_decay, balance):
    """Real growth rates; maximum is the stability diagnostic."""
    return np.linalg.eigvals(jacobian(
        temperature, expert_rate, router_rate, expert_decay,
        router_decay, balance
    )).real


def trial(temperature, seed, args):
    rng = np.random.default_rng(seed)
    # Nonzero perturbations are essential: exact symmetry is stationary.
    state = args.noise * rng.standard_normal(4)
    initial = state.copy()
    history = []
    for step in range(args.steps + 1):
        if step % args.record_every == 0 or step == args.steps:
            history.append({"step": step, "time": step * args.dt,
                            **observable(state, temperature)})
        if step == args.steps:
            break
        state = state + args.dt * dynamics(
            state, temperature, args.expert_rate, args.router_rate,
            args.expert_decay, args.router_decay, args.balance
        )
        if not np.isfinite(state).all():
            raise FloatingPointError("Divergent gradient flow; decrease --dt")
    return {
        "temperature": temperature, "seed": seed,
        "initial": initial.tolist(), "final": state.tolist(),
        "history": history,
    }


def plot(data, path):
    """One order-parameter trajectory panel, one linear growth-rate panel."""
    args = data["config"]
    tau = sorted(set(t["temperature"] for t in data["trials"]))
    fig, axes = plt.subplots(2, 1, figsize=(5.5, 5.6))
    for t in tau:
        sims = [s for s in data["trials"] if s["temperature"] == t]
        times = np.array([h["time"] for h in sims[0]["history"]])
        values = np.array([
            [abs(h["specialization"]) for h in s["history"]]
            for s in sims
        ])
        mean, sd = values.mean(axis=0), values.std(axis=0)
        line, = axes[0].plot(
            times, mean, marker="o", markersize=3,
            markevery=max(1, len(times) // 12),
            label=rf"$T={t:g}$"
        )
        axes[0].fill_between(
            times, np.maximum(mean - sd, 0), mean + sd,
            color=line.get_color(), alpha=0.15,
        )
    axes[0].set_ylabel(r"Conditional routing contrast $|S|$")
    axes[0].legend(frameon=False, ncol=3, fontsize=8)

    grid = np.linspace(min(tau), max(tau), 120)
    growth = [max(eigenvalues(
        t, args["expert_rate"], args["router_rate"], args["expert_decay"],
        args["router_decay"], args["balance"]
    )) for t in grid]
    axes[1].plot(grid, growth)
    axes[1].axhline(0, color="0.4", linestyle=":", linewidth=1)
    if args["expert_decay"] > 0 and args["router_decay"] > 0:
        critical = 1 / np.sqrt(8 * args["expert_decay"] * args["router_decay"])
        if min(tau) < critical < max(tau):
            axes[1].axvline(critical, color="0.5", linestyle="--", linewidth=1)
    for label, ax in zip(("(a)", "(b)"), axes):
        ax.text(-0.16, 1.02, label, transform=ax.transAxes, va="bottom")
    axes[1].set_xlabel(r"Routing temperature $T$")
    axes[1].set_ylabel("Largest linear growth rate")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run(args):
    if args.smoke:
        args.temperatures, args.seeds = "1,6", "0,1"
        args.steps, args.record_every = 60, 10
        if args.output == "results/moe_reduced.json":
            args.output = "results/moe_smoke.json"
    ts = [float(x) for x in args.temperatures.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    if (not ts or not seeds or len(set(ts)) != len(ts)
            or len(set(seeds)) != len(seeds)
            or any(not np.isfinite(t) or t <= 0 for t in ts)
            or args.steps < 1 or args.record_every < 1 or args.dt <= 0
            or args.noise < 0 or args.expert_rate <= 0 or args.router_rate <= 0
            or args.expert_decay < 0 or args.router_decay < 0 or args.balance < 0):
        raise ValueError("Invalid simulation settings")
    data = {
        "config": dict(vars(args)),
        "model": "two-population soft MoE with balanced input frequencies",
        "trials": [trial(t, seed, args) for t in ts for seed in seeds],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2))
    plot(data, output.with_suffix(".pdf"))
    return data


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--temperatures", default="1,3,6")
    p.add_argument("--seeds", default=",".join(str(i) for i in range(12)))
    p.add_argument("--steps", type=int, default=2400)
    p.add_argument("--record-every", type=int, default=20)
    p.add_argument("--dt", type=float, default=0.2)
    p.add_argument("--noise", type=float, default=0.025)
    p.add_argument("--expert-rate", type=float, default=0.25)
    p.add_argument("--router-rate", type=float, default=0.25)
    p.add_argument("--expert-decay", type=float, default=0.08)
    p.add_argument("--router-decay", type=float, default=0.08)
    p.add_argument("--balance", type=float, default=0.2)
    p.add_argument("--output", default="results/moe_reduced.json")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
