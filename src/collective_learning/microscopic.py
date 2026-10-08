"""Microscopic test of whether specialization accelerates collective learning.

Only the local learning allocation differs between the paired populations.
Their independent initial networks, examples and update budgets are identical.
Coupling uses the original simultaneous parameter-diffusion update.
"""

import argparse
import copy
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from .core import coupled_sgd_step


def squared_loss(prediction, target):
    return 0.5 * (prediction - target).square().sum(dim=1)


def task_data(samples, generator, device):
    """Independent input directions with a jointly learnable target y=x."""
    z = torch.randn((2, samples), generator=generator)
    x = torch.zeros((2, samples, 2))
    x[0, :, 0], x[1, :, 1] = z[0], z[1]
    return [(x[k].to(device), x[k].to(device)) for k in range(2)]


def training_stream(steps, batch_size, generator, device):
    z = torch.randn((steps, 2, batch_size), generator=generator)
    x = torch.zeros((steps, 2, batch_size, 2))
    x[:, 0, :, 0], x[:, 1, :, 1] = z[:, 0], z[:, 1]
    return [
        (x[t].reshape(-1, 2).to(device), x[t].reshape(-1, 2).to(device))
        for t in range(steps)
    ]


def allocation_weights(allocations, batch_size, device):
    """Total local loss weight is unchanged by allocation."""
    return [
        torch.cat((
            torch.full((batch_size,), 2 * float(a), device=device),
            torch.full((batch_size,), 2 * (1 - float(a)), device=device),
        ))
        for a in allocations
    ]


def make_population(units, depth, width, scale, seed, device):
    """Depth is the number of hidden linear layers (0, 1 or 2 by default)."""
    torch.manual_seed(seed)
    population = []
    sizes = [2] + [width] * depth + [2]
    for _ in range(units):
        model = nn.Sequential(*[
            nn.Linear(a, b, bias=False)
            for a, b in zip(sizes[:-1], sizes[1:])
        ])
        with torch.no_grad():
            for p in model.parameters():
                p.normal_(0.0, scale)
        population.append(model.to(device))
    return population


def advance(models, allocations, stream, batch_size, rate, coupling, decay):
    weights = allocation_weights(
        allocations, batch_size, next(models[0].parameters()).device
    )
    for batch in stream:
        coupled_sgd_step(
            models, [batch] * len(models),
            learning_rate=rate, coupling=coupling, weight_decay=decay,
            sample_weights=weights, loss_fn=squared_loss,
        )


@torch.no_grad()
def cross_loss(models, data):
    """Each row is a learner, each column is an independently tested task."""
    return np.array([
        [squared_loss(model(x), y).mean().item() for x, y in data]
        for model in models
    ])


@torch.no_grad()
def functional_diversity(models, data):
    values = []
    for x, _ in data:
        y = torch.stack([model(x) for model in models])
        values.append((y - y.mean(dim=0)).square().sum(dim=-1).mean().item())
    return float(np.mean(values))


def measure(models, data, allocations):
    matrix = cross_loss(models, data)
    result = {
        "individual_loss": float(matrix.mean()),
        "functional_diversity": functional_diversity(models, data),
        "cross_loss": matrix.tolist(),
    }
    # Exactly 0 or 1 marks a learner trained solely on one task.
    if np.all((allocations == 0) | (allocations == 1)):
        own = [0 if a == 1 else 1 for a in allocations]
        result["own_task_loss"] = float(np.mean([
            matrix[i, j] for i, j in enumerate(own)
        ]))
        result["off_task_loss"] = float(np.mean([
            matrix[i, 1 - j] for i, j in enumerate(own)
        ]))
    return result


def checkpoints_for(steps):
    return sorted(set([0, steps] + np.geomspace(1, steps, 12).astype(int).tolist()))


def trial(args, depth, scale, coupling, seed):
    # Reinitialize and regenerate the same random stream for every parameter
    # setting at this seed. The two strategies are paired at every update.
    initial = make_population(
        args.units, depth, args.width, scale, seed, args.device
    )
    models = {
        "uniform": copy.deepcopy(initial),
        "specialists": copy.deepcopy(initial),
    }
    allocations = {
        "uniform": np.full(args.units, 0.5),
        "specialists": np.array([float(i % 2) for i in range(args.units)]),
    }
    data = task_data(
        args.evaluation, torch.Generator().manual_seed(seed + 300), args.device
    )
    stream = training_stream(
        args.steps, args.batch, torch.Generator().manual_seed(seed + 100),
        args.device,
    )
    checkpoints = checkpoints_for(args.steps)
    history = []
    previous = 0
    for step in checkpoints:
        part = stream[previous:step]
        if part:
            for name in ("uniform", "specialists"):
                advance(
                    models[name], allocations[name], part,
                    args.batch, args.rate, coupling, args.decay,
                )
        recorded = {
            "step": step,
            **{name: measure(models[name], data, allocations[name])
               for name in models},
        }
        recorded["advantage"] = (
            recorded["uniform"]["individual_loss"]
            - recorded["specialists"]["individual_loss"]
        )
        history.append(recorded)
        previous = step

    return {
        "depth": int(depth), "init_scale": float(scale),
        "coupling": float(coupling), "seed": int(seed),
        "history": history,
    }


def plot(trials, path):
    """Two diagnostic panels for a representative depth and weight scale."""
    depths = sorted({t["depth"] for t in trials})
    scales = sorted({t["init_scale"] for t in trials})
    couplings = sorted({t["coupling"] for t in trials})
    depth, scale = depths[len(depths) // 2], scales[len(scales) // 2]
    matching = [
        t for t in trials
        if t["depth"] == depth and t["init_scale"] == scale
    ]
    selected = couplings[len(couplings) // 2]
    selected_trials = [t for t in matching if t["coupling"] == selected]
    steps = [record["step"] for record in selected_trials[0]["history"]]

    fig, axes = plt.subplots(2, 1, figsize=(5.5, 5.8))
    for name in ("uniform", "specialists"):
        curves = np.array([
            [h[name]["individual_loss"] for h in t["history"]]
            for t in selected_trials
        ])
        avg, std = curves.mean(axis=0), curves.std(axis=0)
        line, = axes[0].plot(steps, avg, marker="o", markersize=3, label=name)
        axes[0].fill_between(
            steps, np.maximum(0, avg - std), avg + std,
            color=line.get_color(), alpha=0.15
        )
    axes[0].set_ylabel("Mean individual test loss")
    axes[0].legend(frameon=False)
    axes[0].set_title(
        rf"$D={depth}$, initial scale $={scale:g}$, $\sigma={selected:g}$",
        fontsize=10,
    )

    for coupling in couplings:
        group = [t for t in matching if t["coupling"] == coupling]
        curves = np.array([
            [h["advantage"] for h in t["history"]]
            for t in group
        ])
        avg, std = curves.mean(axis=0), curves.std(axis=0)
        line, = axes[1].plot(
            steps, avg, marker="o", markersize=3,
            label=rf"$\sigma={coupling:g}$"
        )
        axes[1].fill_between(
            steps, avg - std, avg + std,
            color=line.get_color(), alpha=0.15
        )
    axes[1].axhline(0, color="0.4", linestyle=":", linewidth=1)
    axes[1].set_xlabel("SGD updates")
    axes[1].set_ylabel(r"$E_{\mathrm{uniform}}-E_{\mathrm{specialists}}$")
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def numbers(value, cast):
    parts = [cast(part) for part in value.split(",")]
    if not parts or len(set(parts)) != len(parts):
        raise ValueError("Grid entries must be nonempty and distinct")
    return parts


def run(args):
    if args.smoke:
        args.depths, args.scales, args.couplings, args.seeds = (
            "0,1", "0.1", "0,0.6", "0"
        )
        args.steps, args.batch, args.evaluation = 6, 4, 16
        if args.output == "results/microscopic_timescales.json":
            args.output = "results/microscopic_smoke.json"

    depths = numbers(args.depths, int)
    scales = numbers(args.scales, float)
    couplings = numbers(args.couplings, float)
    seeds = numbers(args.seeds, int)
    if (args.units < 2 or args.units % 2 or args.width < 2
            or args.steps < 1 or args.batch < 1 or args.evaluation < 1
            or args.rate <= 0 or args.decay < 0
            or any(d < 0 for d in depths)
            or any(not np.isfinite(s) or s <= 0 for s in scales)
            or any(not np.isfinite(c) or c < 0 for c in couplings)):
        raise ValueError("Invalid training budget, depth, weight scale or coupling")

    args.device = torch.device(args.device)
    if args.device.type == "cpu":
        torch.set_num_threads(1)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    config = dict(vars(args), device=str(args.device))
    trials = []
    total = len(depths) * len(scales) * len(couplings) * len(seeds)
    print(f"Microscopic scan: {total} paired trials, {args.steps} updates each", flush=True)

    for depth in depths:
        for scale in scales:
            for coupling in couplings:
                for seed in seeds:
                    result = trial(args, depth, scale, coupling, seed)
                    trials.append(result)
                    final = result["history"][-1]
                    print(
                        f"{len(trials)}/{total} D={depth} init={scale:g} "
                        f"sigma={coupling:g} seed={seed} "
                        f"uniform={final['uniform']['individual_loss']:.5f} "
                        f"specialists={final['specialists']['individual_loss']:.5f} "
                        f"delta={final['advantage']:+.5f}", flush=True
                    )
                    output.write_text(json.dumps({
                        "config": config, "complete": False, "trials": trials
                    }, indent=2))

    output.write_text(json.dumps({
        "config": config, "complete": True, "trials": trials
    }, indent=2))
    plot(trials, output.with_suffix(".pdf"))
    print(f"Saved {output} and {output.with_suffix('.pdf')}")
    return trials


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depths", default="0,1,2", help="Hidden linear layers")
    parser.add_argument("--scales", default="0.03,0.1,0.3")
    parser.add_argument("--couplings", default="0,0.2,1,3")
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--units", type=int, default=4)
    parser.add_argument("--width", type=int, default=2)
    parser.add_argument("--batch", type=int, default=32, help="Samples per task per step")
    parser.add_argument("--rate", type=float, default=0.05)
    parser.add_argument("--decay", type=float, default=0.001)
    parser.add_argument("--evaluation", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="results/microscopic_timescales.json")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
