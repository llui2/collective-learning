"""Two-task deep-linear learners with evolving microscopic learning allocation.

Strategies compete by their effect on future population validation error.
No differentiation reward, effective closure, or presumed collective gain is used.
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
    return 0.5 * (prediction[:, 0] - target).square()


def task_data(samples, generator, device):
    """Two orthogonal input classes with the same target y=x[0]+x[1]."""
    z = torch.randn((2, samples), generator=generator)
    x = torch.zeros((2, samples, 2))
    x[0, :, 0] = z[0]
    x[1, :, 1] = z[1]
    return [(x[k].to(device), z[k].to(device)) for k in range(2)]


def training_stream(steps, batch_size, generator, device):
    z = torch.randn((steps, 2, batch_size), generator=generator)
    x = torch.zeros((steps, 2, batch_size, 2))
    x[:, 0, :, 0] = z[:, 0]
    x[:, 1, :, 1] = z[:, 1]
    return [
        (x[t].reshape(2 * batch_size, 2).to(device),
         z[t].reshape(2 * batch_size).to(device))
        for t in range(steps)
    ]


def allocation_weights(a, batch_size, device):
    """Each learner has total batch weight 2*batch_size, independent of a."""
    return [
        torch.cat((
            torch.full((batch_size,), 2 * float(p), device=device),
            torch.full((batch_size,), 2 * (1 - float(p)), device=device),
        ))
        for p in a
    ]


def advance(models, a, stream, args):
    weights = allocation_weights(a, args.batch, next(models[0].parameters()).device)
    for batch in stream:
        coupled_sgd_step(
            models, [batch] * len(models),
            learning_rate=args.rate, coupling=args.coupling,
            weight_decay=args.decay, sample_weights=weights,
            loss_fn=squared_loss,
        )
    return models


@torch.no_grad()
def cross_loss(models, data):
    """One row per learner and one column per task, as in Arola--Lacasa."""
    return np.array([
        [float(squared_loss(model(x), y).mean()) for x, y in data]
        for model in models
    ])


def record(round_idx, models, allocations, data):
    entry = {"round": round_idx}
    for name in models:
        a = allocations[name]
        matrix = cross_loss(models[name], data)
        entry[name] = {
            "loss": float(matrix.mean()),
            "cross_loss": matrix.tolist(),
            "a": a.tolist(),
            "diversity": float(2 * np.var(a)),
        }
    return entry


def figure(history, out):
    t = [entry["round"] for entry in history]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for name in ("uniform", "frozen", "evolving"):
        axes[0].plot(t, [entry[name]["loss"] for entry in history], label=name)
    axes[0].set_xlabel("Selection rounds")
    axes[0].set_ylabel("Mean validation loss")
    axes[0].legend(frameon=False)

    allocations = np.array([entry["evolving"]["a"] for entry in history])
    for i in range(allocations.shape[1]):
        axes[1].plot(t, allocations[:, i], label=fr"$i={i+1}$")
    axes[1].axhline(0.5, color="0.6", linewidth=0.8, linestyle=":")
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("Selection rounds")
    axes[1].set_ylabel(r"Task-1 allocation $a_i$")
    axes[1].legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def run(args):
    if args.units < 2 or args.width < 1 or args.batch < 1 or args.window < 1:
        raise ValueError("units >= 2, width, batch and window must be positive")
    if args.rounds < 1 or args.rate <= 0 or args.mutation < 0 or args.coupling < 0:
        raise ValueError("rounds and rate must be positive; coupling and mutation nonnegative")

    if args.smoke:
        args.rounds, args.window, args.batch = 2, 2, 8
        args.evaluation = 32
        if args.output == "results/microscopic.json":
            args.output = "results/microscopic_smoke.json"

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    batches = torch.Generator().manual_seed(args.seed + 100)
    validation = task_data(args.evaluation, torch.Generator().manual_seed(args.seed + 200), device)
    test = task_data(args.evaluation, torch.Generator().manual_seed(args.seed + 300), device)

    initial_models = [
        nn.Sequential(
            nn.Linear(2, args.width, bias=False),
            nn.Linear(args.width, 1, bias=False),
        ).to(device)
        for _ in range(args.units)
    ]
    initial_a = np.clip(0.5 + 0.03 * rng.standard_normal(args.units), 0.05, 0.95)
    allocations = {
        "uniform": np.full(args.units, 0.5),
        "frozen": initial_a.copy(),
        "evolving": initial_a.copy(),
    }
    models = {name: copy.deepcopy(initial_models) for name in allocations}
    history = [record(0, models, allocations, validation)]
    accepted = 0

    for round_idx in range(1, args.rounds + 1):
        stream = training_stream(args.window, args.batch, batches, device)
        for name in ("uniform", "frozen"):
            advance(models[name], allocations[name], stream, args)

        a = allocations["evolving"]
        proposal = a.copy()
        i = int(rng.integers(args.units))
        proposal[i] = np.clip(proposal[i] + rng.normal(0, args.mutation), 0.02, 0.98)

        unchanged = advance(copy.deepcopy(models["evolving"]), a, stream, args)
        mutated = advance(copy.deepcopy(models["evolving"]), proposal, stream, args)
        if cross_loss(mutated, validation).mean() < cross_loss(unchanged, validation).mean() - 1e-10:
            models["evolving"] = mutated
            allocations["evolving"] = proposal
            accepted += 1
        else:
            models["evolving"] = unchanged

        history.append(record(round_idx, models, allocations, validation))

    result = {
        "config": vars(args),
        "accepted_mutations": accepted,
        "history": history,
        "test": record(args.rounds, models, allocations, test),
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    figure(history, path.with_suffix(".pdf"))

    print(f"Accepted mutations: {accepted}/{args.rounds}")
    for name in allocations:
        summary = result["test"][name]
        print(
            f"{name:8s} test_loss={summary['loss']:.5f} "
            f"diversity={summary['diversity']:.5f} "
            f"allocations={np.round(allocations[name], 3).tolist()}"
        )
    print(f"Saved {path} and {path.with_suffix('.pdf')}")
    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--coupling", type=float, default=0.6)
    p.add_argument("--rounds", type=int, default=60)
    p.add_argument("--window", type=int, default=20, help="SGD steps per selection round")
    p.add_argument("--units", type=int, default=4)
    p.add_argument("--width", type=int, default=2)
    p.add_argument("--batch", type=int, default=32, help="examples per task per SGD step")
    p.add_argument("--rate", type=float, default=0.08, help="SGD learning rate")
    p.add_argument("--decay", type=float, default=0.001)
    p.add_argument("--mutation", type=float, default=0.15)
    p.add_argument("--evaluation", type=int, default=512, help="examples per task")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", default="results/microscopic.json")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
