"""Microscopic test of whether specialization accelerates collective learning.

Only the local learning allocation differs between the paired populations.
Their independent initial networks, examples and update budgets are identical.
Coupling uses the original simultaneous parameter-diffusion update.
"""

import argparse
import copy
import json
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
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


def checkpoints_for(steps, every=10):
    """Uniformly spaced measurements; threshold times are interval-censored."""
    return sorted(set(range(0, steps + 1, every)) | {steps})


def first_crossing(history, strategy, threshold):
    """First measured step with individual test loss at or below threshold."""
    return next(
        (h["step"] for h in history
         if h[strategy]["individual_loss"] <= threshold), None
    )


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
    checkpoints = checkpoints_for(args.steps, args.check_every)
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

    hits = {
        str(threshold): {
            name: first_crossing(history, name, threshold)
            for name in ("uniform", "specialists")
        }
        for threshold in numbers(args.thresholds, float)
    }
    return {
        "depth": int(depth), "init_scale": float(scale),
        "coupling": float(coupling), "seed": int(seed),
        "first_crossing": hits, "history": history,
    }


def summarize(trials, args, complete):
    """A compact, shareable summary. None denotes right-censored failure."""
    thresholds = numbers(args.thresholds, float)
    groups = {}
    for trial_data in trials:
        key = (trial_data["depth"], trial_data["init_scale"],
               trial_data["coupling"])
        groups.setdefault(key, []).append(trial_data)

    entries = []
    for (depth, scale, coupling), matched in sorted(groups.items()):
        steps = [h["step"] for h in matched[0]["history"]]
        means = {}
        for strategy in ("uniform", "specialists"):
            matrix = np.array([
                [h[strategy]["individual_loss"] for h in t["history"]]
                for t in matched
            ])
            means[strategy] = {
                "mean": matrix.mean(axis=0).tolist(),
                "std": matrix.std(axis=0).tolist(),
            }
        outcomes = {}
        for threshold in thresholds:
            label = str(threshold)
            observed = {
                strategy: [t["first_crossing"][label][strategy] for t in matched]
                for strategy in ("uniform", "specialists")
            }
            # T^H=min(T,H): for censored runs use H, not a fictional hit.
            differences = [
                (u if u is not None else args.steps)
                - (s if s is not None else args.steps)
                for u, s in zip(observed["uniform"], observed["specialists"])
            ]
            both = [
                u - s for u, s in zip(observed["uniform"], observed["specialists"])
                if u is not None and s is not None
            ]
            outcomes[label] = {
                "hit_uniform": sum(t is not None for t in observed["uniform"]),
                "hit_specialists": sum(t is not None for t in observed["specialists"]),
                "both_hit": len(both),
                "specialists_faster_when_both_hit": sum(x > 0 for x in both),
                "restricted_speedup_mean": float(np.mean(differences)),
                "restricted_speedup_std": float(np.std(differences)),
            }
        entries.append({
            "depth": depth, "init_scale": scale, "coupling": coupling,
            "seeds": [t["seed"] for t in matched],
            "checkpoints": steps, "individual_loss": means,
            "thresholds": outcomes,
            "final_off_task_loss_mean": float(np.mean([
                t["history"][-1]["specialists"]["off_task_loss"]
                for t in matched
            ])),
        })
    per_trial = [
        {
            "depth": t["depth"], "init_scale": t["init_scale"],
            "coupling": t["coupling"], "seed": t["seed"],
            "first_crossing": t["first_crossing"],
            "final_individual_loss": {
                name: t["history"][-1][name]["individual_loss"]
                for name in ("uniform", "specialists")
            },
            "final_specialist_off_task_loss": (
                t["history"][-1]["specialists"]["off_task_loss"]
            ),
        }
        for t in trials
    ]
    return {
        "config": dict(vars(args), device=str(args.device)),
        "complete": complete, "trials": per_trial, "groups": entries,
        "interpretation": {
            "crossing": "first recorded mean individual test loss <= threshold",
            "censoring": "null means not reached by final observed update",
            "resolution": "first crossing is observed on the checkpoint grid",
            "restricted_speedup": "min(T_uniform,H)-min(T_specialists,H), H=steps",
            "positive": "specialists reach the threshold earlier, within horizon",
        },
    }


def plot(trials, args, path):
    """Learning curves and predeclared restricted crossing-time comparison."""
    depths = sorted({t["depth"] for t in trials})
    scales = sorted({t["init_scale"] for t in trials})
    couplings = sorted({t["coupling"] for t in trials})
    depth = 1 if 1 in depths else depths[0]
    scale = scales[0]
    selected = 1.0 if 1.0 in couplings else couplings[len(couplings) // 2]
    focus = [t for t in trials if t["depth"] == depth
             and t["init_scale"] == scale and t["coupling"] == selected]
    steps = [h["step"] for h in focus[0]["history"]]
    threshold = numbers(args.thresholds, float)[0]
    fig, axes = plt.subplots(2, 1, figsize=(6, 5.5))

    for name in ("uniform", "specialists"):
        curves = np.array([
            [h[name]["individual_loss"] for h in t["history"]]
            for t in focus
        ])
        avg, std = curves.mean(axis=0), curves.std(axis=0)
        line, = axes[0].plot(steps, avg, label=name)
        axes[0].fill_between(
            steps, np.maximum(0, avg - std), avg + std,
            color=line.get_color(), alpha=0.16
        )
    axes[0].axhline(threshold, color="0.5", linestyle=":", linewidth=1)
    axes[0].set_ylabel("Mean individual test loss")
    axes[0].set_title(
        rf"$D={depth}$, initial scale $={scale:g}$, $\sigma={selected:g}$",
        fontsize=10,
    )
    axes[0].legend(frameon=False)

    for init_scale in scales:
        points = []
        spreads = []
        for coupling in couplings:
            group = [
                t for t in trials if t["depth"] == depth
                and t["init_scale"] == init_scale and t["coupling"] == coupling
            ]
            label = str(threshold)
            differences = [
                (t["first_crossing"][label]["uniform"]
                 if t["first_crossing"][label]["uniform"] is not None
                 else args.steps)
                - (t["first_crossing"][label]["specialists"]
                   if t["first_crossing"][label]["specialists"] is not None
                   else args.steps)
                for t in group
            ]
            points.append(np.mean(differences))
            spreads.append(np.std(differences))
        line, = axes[1].plot(
            couplings, points, marker="o",
            label=rf"initial scale $={init_scale:g}$",
        )
        points, spreads = np.array(points), np.array(spreads)
        axes[1].fill_between(
            couplings, points - spreads, points + spreads,
            alpha=0.16, color=line.get_color()
        )
    axes[1].axhline(0, color="0.5", linestyle=":", linewidth=1)
    axes[1].set_xlabel(r"Parameter coupling $\sigma$")
    axes[1].set_ylabel(r"Restricted $T_{\mathrm{uniform}}-T_{\mathrm{specialists}}$")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _worker(job):
    args, depth, scale, coupling, seed = job
    if args.device.type == "cpu":
        torch.set_num_threads(1)
    return trial(args, depth, scale, coupling, seed)


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
            or args.jobs < 1
            or args.rate <= 0 or args.decay < 0
            or any(d < 0 for d in depths)
            or any(not np.isfinite(s) or s <= 0 for s in scales)
            or any(not np.isfinite(c) or c < 0 for c in couplings)):
        raise ValueError("Invalid training budget, depth, weight scale or coupling")

    args.device = torch.device(args.device)
    if args.jobs > 1 and args.device.type != "cpu":
        raise ValueError("Multiple workers are supported for CPU runs only")
    if args.device.type == "cpu":
        torch.set_num_threads(1)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    config = dict(vars(args), device=str(args.device))
    trials = []
    total = len(depths) * len(scales) * len(couplings) * len(seeds)
    print(f"Microscopic scan: {total} paired trials, {args.steps} updates each", flush=True)

    jobs = [
        (args, depth, scale, coupling, seed)
        for depth in depths for scale in scales
        for coupling in couplings for seed in seeds
    ]
    executor = (
        ProcessPoolExecutor(max_workers=args.jobs)
        if args.jobs > 1 else nullcontext()
    )
    with executor as pool:
        results = pool.map(_worker, jobs) if pool else map(_worker, jobs)
        for result in results:
            trials.append(result)
            final = result["history"][-1]
            print(
                f"{len(trials)}/{total} D={result['depth']} "
                f"init={result['init_scale']:g} "
                f"sigma={result['coupling']:g} seed={result['seed']} "
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
    parser.add_argument("--jobs", type=int, default=1, help="Parallel CPU trials")
    parser.add_argument("--output", default="results/microscopic_timescales.json")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
