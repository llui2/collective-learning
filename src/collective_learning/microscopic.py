"""Finite-time collective learning with evolving task allocations.

Each deep-linear learner has enough width to learn both independent tasks.
Training effort is finite, and coupling is parameter diffusion as in the
original neural baseline. Evolution selects on mean individual validation
loss; ensemble prediction is reported only as a separate diagnostic.
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
    """Two orthogonal input classes, with common teacher y=x."""
    z = torch.randn((2, samples), generator=generator)
    x = torch.zeros((2, samples, 2))
    x[0, :, 0], x[1, :, 1] = z[0], z[1]
    return [(x[k].to(device), x[k].to(device)) for k in range(2)]


def training_stream(steps, batch, generator, device):
    """Shared examples make the allocation/coupling comparisons paired."""
    z = torch.randn((steps, 2, batch), generator=generator)
    x = torch.zeros((steps, 2, batch, 2))
    x[:, 0, :, 0], x[:, 1, :, 1] = z[:, 0], z[:, 1]
    return [(x[t].reshape(-1, 2).to(device), x[t].reshape(-1, 2).to(device))
            for t in range(steps)]


def allocation_weights(a, batch, device):
    """Mean sample weight is one for every learner, for any allocation."""
    return [
        torch.cat((torch.full((batch,), 2 * float(p), device=device),
                   torch.full((batch,), 2 * (1 - float(p)), device=device)))
        for p in a
    ]


def advance(models, allocations, stream, args):
    weights = allocation_weights(
        allocations, args.batch, next(models[0].parameters()).device
    )
    for batch in stream:
        coupled_sgd_step(
            models, [batch] * len(models), learning_rate=args.rate,
            coupling=args.coupling, weight_decay=args.decay,
            sample_weights=weights, loss_fn=squared_loss,
        )
    return models


@torch.no_grad()
def cross_loss(models, data):
    return np.asarray([
        [squared_loss(model(x), y).mean().item() for x, y in data]
        for model in models
    ])


@torch.no_grad()
def ensemble_metrics(models, data):
    ensemble_errors = []
    functional_variances = []
    for x, y in data:
        outputs = torch.stack([model(x) for model in models])
        mean = outputs.mean(dim=0)
        ensemble_errors.append(squared_loss(mean, y).mean().item())
        functional_variances.append(
            (outputs - mean).square().sum(dim=-1).mean().item()
        )
    return float(np.mean(ensemble_errors)), float(np.mean(functional_variances))


def measure(models, allocations, data, specialized=False):
    errors = cross_loss(models, data)
    ensemble, function_diversity = ensemble_metrics(models, data)
    result = {
        "individual_loss": float(errors.mean()),
        "cross_loss": errors.tolist(),
        "ensemble_loss": ensemble,
        "allocation_diversity": float(np.var(allocations)),
        "functional_diversity": function_diversity,
        "allocations": allocations.tolist(),
    }
    if specialized:
        private = [0 if a == 1 else 1 for a in allocations]
        result["own_task_loss"] = float(np.mean(
            [errors[i, private[i]] for i in range(len(models))]
        ))
        result["off_task_loss"] = float(np.mean(
            [errors[i, 1 - private[i]] for i in range(len(models))]
        ))
    return result


def population_loss(models, data):
    """Selection objective: population-mean individual validation loss."""
    return float(cross_loss(models, data).mean())


def trial(args, seed, coupling):
    # The identical initialization and streams are reused across conditions.
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    batches = torch.Generator().manual_seed(seed + 100)
    validation = task_data(
        args.evaluation, torch.Generator().manual_seed(seed + 200), args.device
    )
    test = task_data(
        args.evaluation, torch.Generator().manual_seed(seed + 300), args.device
    )
    initial = [
        nn.Sequential(
            nn.Linear(2, args.width, bias=False),
            nn.Linear(args.width, 2, bias=False),
        ).to(args.device)
        for _ in range(args.units)
    ]
    a0 = np.clip(0.5 + 0.03 * rng.standard_normal(args.units), 0.0, 1.0)
    allocations = {
        "uniform": np.full(args.units, 0.5),
        "specialists": np.array([float(i % 2) for i in range(args.units)]),
        "frozen": a0.copy(),
        "evolving": a0.copy(),
    }
    models = {name: copy.deepcopy(initial) for name in allocations}
    args.coupling = float(coupling)

    def snapshot(round_idx, data):
        return {
            "round": round_idx,
            **{
                name: measure(
                    models[name], allocations[name], data,
                    specialized=(name == "specialists")
                )
                for name in models
            },
        }

    history = [snapshot(0, validation)]
    accepted = 0
    for k in range(1, args.rounds + 1):
        stream = training_stream(args.window, args.batch, batches, args.device)
        for name in ("uniform", "specialists", "frozen"):
            advance(models[name], allocations[name], stream, args)

        current = allocations["evolving"]
        candidate = current.copy()
        idx = int(rng.integers(args.units))
        candidate[idx] = np.clip(
            candidate[idx] + rng.normal(0, args.mutation), 0.0, 1.0
        )
        base = advance(copy.deepcopy(models["evolving"]), current, stream, args)
        proposed = advance(copy.deepcopy(models["evolving"]), candidate, stream, args)
        # Both counterfactuals start from the same weights and see the same data.
        if population_loss(proposed, validation) < population_loss(base, validation) - 1e-10:
            allocations["evolving"] = candidate
            models["evolving"] = proposed
            accepted += 1
        else:
            models["evolving"] = base
        history.append(snapshot(k, validation))

    return {
        "seed": int(seed),
        "coupling": float(coupling),
        "accepted_mutations": accepted,
        "history": history,
        "test": snapshot(args.rounds, test),
    }


def figure(trials, path):
    couplings = sorted(set(t["coupling"] for t in trials))
    fig, axes = plt.subplots(2, 1, figsize=(5.5, 5.5), sharex=True)
    strategies = ("uniform", "specialists", "frozen", "evolving")
    for name in strategies:
        groups = [[t["test"][name]["individual_loss"] for t in trials
                   if t["coupling"] == s] for s in couplings]
        mean = np.array([np.mean(v) for v in groups])
        std = np.array([np.std(v) for v in groups])
        line, = axes[0].plot(couplings, mean, marker="o", label=name)
        axes[0].fill_between(
            couplings, np.maximum(mean - std, 0), mean + std,
            alpha=0.15, color=line.get_color()
        )
    axes[0].set_ylabel("Mean individual test loss")
    axes[0].legend(frameon=False, ncol=2, fontsize=8)

    for key, label, style in (
        ("own_task_loss", "studied task", "-"),
        ("off_task_loss", "unstudied task", "--"),
    ):
        groups = [[t["test"]["specialists"][key] for t in trials
                   if t["coupling"] == s] for s in couplings]
        mean = np.array([np.mean(v) for v in groups])
        std = np.array([np.std(v) for v in groups])
        line, = axes[1].plot(
            couplings, mean, linestyle=style, marker="o", label=label
        )
        axes[1].fill_between(
            couplings, np.maximum(mean - std, 0), mean + std,
            alpha=0.15, color=line.get_color()
        )
    axes[1].set_xlabel(r"Parameter coupling $\sigma$")
    axes[1].set_ylabel("Fixed specialists: test loss")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run(args):
    if args.smoke:
        args.couplings, args.seeds = "0,0.6", "0"
        args.rounds, args.window, args.batch, args.evaluation = 2, 2, 8, 24
        if args.output == "results/microscopic_sweep.json":
            args.output = "results/microscopic_smoke.json"

    couplings = [float(x) for x in args.couplings.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    if (not couplings or not seeds or len(set(couplings)) != len(couplings)
            or len(set(seeds)) != len(seeds)
            or any(not np.isfinite(s) or s < 0 for s in couplings)):
        raise ValueError("Couplings must be distinct, finite and nonnegative; seeds distinct")
    if (args.units < 2 or args.units % 2 or args.width < 2
            or args.rounds < 1 or args.window < 1 or args.batch < 1
            or args.evaluation < 1 or args.rate <= 0 or args.decay < 0
            or args.mutation < 0):
        raise ValueError("Need even units >= 2, width >= 2, positive budget/rate")

    args.device = torch.device(args.device)
    output = Path(args.output)
    summary_path = output.with_name(output.stem + "_summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    trials = []

    def save(complete):
        config = dict(vars(args), device=str(args.device))
        config.pop("coupling", None)  # The sweep contains several values.
        output.write_text(json.dumps(
            {"config": config, "complete": complete, "trials": trials}, indent=2
        ))
        summary_path.write_text(json.dumps({
            "config": config,
            "complete": complete,
            "trials": [
                {
                    "seed": t["seed"], "coupling": t["coupling"],
                    "accepted_mutations": t["accepted_mutations"],
                    "test": {
                        name: {key: value for key, value in t["test"][name].items()
                               if key != "cross_loss"}
                        for name in ("uniform", "specialists", "frozen", "evolving")
                    },
                }
                for t in trials
            ],
        }, indent=2))
    for seed in seeds:
        for coupling in couplings:
            result = trial(args, seed, coupling)
            trials.append(result)
            tested = result["test"]
            print(
                f"seed={seed} sigma={coupling:g} "
                f"uniform={tested['uniform']['individual_loss']:.4f} "
                f"specialists={tested['specialists']['individual_loss']:.4f} "
                f"evolving={tested['evolving']['individual_loss']:.4f} "
                f"unstudied={tested['specialists']['off_task_loss']:.4f} "
                f"accepted={result['accepted_mutations']}/{args.rounds}",
                flush=True,
            )
            save(complete=False)

    save(complete=True)
    figure(trials, output.with_suffix(".pdf"))
    print(f"Saved {output}, {summary_path} and {output.with_suffix('.pdf')}")
    return trials


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--couplings", default="0,0.1,0.3,0.6,1,2,4")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--units", type=int, default=4)
    parser.add_argument("--width", type=int, default=2)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--rate", type=float, default=0.04)
    parser.add_argument("--decay", type=float, default=0.01)
    parser.add_argument("--mutation", type=float, default=0.15)
    parser.add_argument("--evaluation", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="results/microscopic_sweep.json")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
