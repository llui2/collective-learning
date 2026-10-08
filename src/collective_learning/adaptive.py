"""Mutation-selection of task allocations in coupled deep-linear learners.

A global validation objective chooses between two equal-data, equal-SGD
counterfactual rollouts. This is an oracle-assisted evolutionary pilot,
not decentralized adaptation. Neither heterogeneity nor specialization
appears in the selection objective.
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

from .microscopic import (
    advance, cross_loss, first_crossing, make_population, measure,
    numbers, task_data, training_stream,
)


STRATEGIES = ("uniform", "specialists", "frozen", "drift", "evolving")


def propose(allocations, index, change, lower=0.02, upper=0.98):
    """Change one study allocation, keeping fixed total learning effort."""
    proposed = allocations.copy()
    proposed[index] = np.clip(proposed[index] + change, lower, upper)
    return proposed


def population_loss(models, validation):
    """Mean individual generalization across BOTH tasks, not an ensemble loss."""
    return float(cross_loss(models, validation).mean())


def statistics(models, allocation, test):
    result = measure(models, test, allocation)
    result["allocations"] = allocation.tolist()
    result["polarization"] = float(2 * np.mean(np.abs(allocation - 0.5)))
    result["allocation_variance"] = float(np.var(allocation))
    result["task1_total_allocation"] = float(np.sum(allocation))
    return result


def trial(args, scale, coupling, window, seed):
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(1)
    initial = make_population(
        args.units, 1, args.width, scale, seed, args.device
    )
    models = {name: copy.deepcopy(initial) for name in STRATEGIES}
    rng = np.random.default_rng(seed + 500)
    allocations0 = np.clip(
        0.5 + 0.03 * rng.standard_normal(args.units), 0.02, 0.98
    )
    allocations = {
        "uniform": np.full(args.units, 0.5),
        "specialists": np.array([float(i % 2) for i in range(args.units)]),
        "frozen": allocations0.copy(),
        "drift": allocations0.copy(),
        "evolving": allocations0.copy(),
    }
    batches = torch.Generator().manual_seed(seed + 100)
    validation = task_data(
        args.evaluation, torch.Generator().manual_seed(seed + 200),
        args.device
    )
    test = task_data(
        args.evaluation, torch.Generator().manual_seed(seed + 300),
        args.device
    )

    def record(step):
        return {
            "step": step,
            **{name: statistics(models[name], allocations[name], test)
               for name in STRATEGIES},
        }

    history = [record(0)]
    accepted = 0
    proposals = 0
    for start in range(0, args.steps, window):
        n = min(window, args.steps - start)
        stream = training_stream(n, args.batch, batches, args.device)

        index = int(rng.integers(args.units))
        change = float(rng.normal(0.0, args.mutation))
        proposals += 1

        # Paired controls: same initial learners, minibatches and SGD steps.
        for name in ("uniform", "specialists", "frozen"):
            advance(
                models[name], allocations[name], stream,
                args.batch, args.rate, coupling, args.decay,
            )

        # Neutral drift sees the same mutation proposals but selection is random.
        if rng.random() < 0.5:
            allocations["drift"] = propose(
                allocations["drift"], index, change
            )
        advance(
            models["drift"], allocations["drift"], stream,
            args.batch, args.rate, coupling, args.decay,
        )

        base = copy.deepcopy(models["evolving"])
        mutant = copy.deepcopy(models["evolving"])
        candidate = propose(allocations["evolving"], index, change)
        advance(
            base, allocations["evolving"], stream,
            args.batch, args.rate, coupling, args.decay,
        )
        advance(
            mutant, candidate, stream,
            args.batch, args.rate, coupling, args.decay,
        )
        # The training horizon for the two counterfactuals is identical.
        # Validation is independent of the training batches and test set.
        if population_loss(mutant, validation) < population_loss(base, validation):
            models["evolving"] = mutant
            allocations["evolving"] = candidate
            accepted += 1
        else:
            models["evolving"] = base

        history.append(record(start + n))

    hits = {
        str(t): {
            name: first_crossing(history, name, t)
            for name in STRATEGIES
        } for t in numbers(args.thresholds, float)
    }
    return {
        "scale": float(scale), "coupling": float(coupling),
        "window": int(window), "seed": int(seed),
        "accepted": accepted, "proposals": proposals,
        "first_crossing": hits, "history": history,
    }


def compact(trials, args, complete):
    """Shareable outcomes and mean trajectories without full cross-loss matrices."""
    result = []
    for tr in trials:
        last = tr["history"][-1]
        result.append({
            "scale": tr["scale"], "coupling": tr["coupling"],
            "window": tr["window"], "seed": tr["seed"],
            "accepted": tr["accepted"], "proposals": tr["proposals"],
            "first_crossing": tr["first_crossing"],
            "final": {
                name: {key: last[name][key] for key in (
                    "individual_loss", "functional_diversity", "allocations",
                    "polarization", "allocation_variance",
                    "task1_total_allocation",
                )}
                for name in STRATEGIES
            },
            "history": [{
                "step": h["step"],
                **{name: {
                    "individual_loss": h[name]["individual_loss"],
                    "polarization": h[name]["polarization"],
                } for name in STRATEGIES},
            } for h in tr["history"]],
        })
    return {
        "config": dict(vars(args), device=str(args.device)),
        "complete": complete,
        "trials": result,
        "selection": "lower next-window population-mean validation loss",
        "censoring": "null first_crossing means loss threshold not reached by horizon",
        "limitation": "global heldout labels and extra counterfactual computation",
    }


def plot(trials, args, out):
    """Fixed comparison: sigma=1 (if supplied), first scale, first window."""
    couplings = numbers(args.couplings, float)
    scales = numbers(args.scales, float)
    windows = numbers(args.windows, int)
    sigma = 1.0 if 1.0 in couplings else couplings[0]
    focus = [
        t for t in trials if t["scale"] == scales[0]
        and t["coupling"] == sigma and t["window"] == windows[0]
    ]
    step = [h["step"] for h in focus[0]["history"]]
    fig, axes = plt.subplots(2, 1, figsize=(5.8, 5.8), sharex=True)
    for name in STRATEGIES:
        for ax, variable in zip(axes, ("individual_loss", "polarization")):
            curves = np.array([
                [h[name][variable] for h in t["history"]] for t in focus
            ])
            mean = curves.mean(axis=0)
            sd = curves.std(axis=0)
            line, = ax.plot(step, mean, label=name)
            ax.fill_between(
                step, mean - sd, mean + sd, alpha=0.13,
                color=line.get_color()
            )
    axes[0].axhline(
        numbers(args.thresholds, float)[0], color="0.5",
        linestyle=":", linewidth=0.9,
    )
    axes[0].set_ylabel("Mean individual test loss")
    axes[1].set_ylabel("Allocation polarization")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_xlabel("SGD steps")
    axes[0].set_title(
        rf"$\sigma={sigma:g}$, initial scale $={scales[0]:g}$, "
        rf"selection window $={windows[0]}$", fontsize=10,
    )
    axes[0].legend(frameon=False, ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _worker(job):
    return trial(*job)


def run(args):
    if args.smoke:
        args.scales, args.couplings, args.windows, args.seeds = (
            "0.03", "0,1", "2", "0"
        )
        args.steps, args.batch, args.evaluation = 8, 4, 12
        if args.output == "results/microscopic_adaptive.json":
            args.output = "results/microscopic_adaptive_smoke.json"

    scales = numbers(args.scales, float)
    couplings = numbers(args.couplings, float)
    windows = numbers(args.windows, int)
    seeds = numbers(args.seeds, int)
    thresholds = numbers(args.thresholds, float)
    if (args.units < 2 or args.units % 2 or args.width < 2
            or args.steps < 1 or args.batch < 1 or args.evaluation < 1
            or args.jobs < 1 or args.rate <= 0 or args.decay < 0
            or args.mutation < 0 or not np.isfinite(args.mutation)
            or any(not np.isfinite(x) or x <= 0 for x in scales)
            or any(not np.isfinite(x) or x < 0 for x in couplings)
            or any(x < 1 for x in windows)
            or any(not np.isfinite(x) or x <= 0 for x in thresholds)):
        raise ValueError("Invalid allocation experiment parameters")

    args.device = torch.device(args.device)
    if args.jobs > 1 and args.device.type != "cpu":
        raise ValueError("Parallel jobs require CPU")
    if args.device.type == "cpu":
        torch.set_num_threads(1)
    output = Path(args.output)
    summary = output.with_name(output.stem + "_summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    jobs = [
        (args, scale, coupling, window, seed)
        for scale in scales for coupling in couplings
        for window in windows for seed in seeds
    ]
    print(f"Adaptive allocation: {len(jobs)} trials", flush=True)
    completed = []
    pool = (
        ProcessPoolExecutor(max_workers=args.jobs)
        if args.jobs > 1 else nullcontext()
    )
    with pool as executor:
        results = executor.map(_worker, jobs) if executor else map(_worker, jobs)
        for t in results:
            completed.append(t)
            final = t["history"][-1]
            print(
                f"{len(completed)}/{len(jobs)} sigma={t['coupling']:g} "
                f"init={t['scale']:g} window={t['window']} seed={t['seed']} "
                f"uniform={final['uniform']['individual_loss']:.4f} "
                f"specialists={final['specialists']['individual_loss']:.4f} "
                f"evolving={final['evolving']['individual_loss']:.4f} "
                f"polarization={final['evolving']['polarization']:.3f} "
                f"accepted={t['accepted']}/{t['proposals']}", flush=True
            )
            # Incremental outputs survive an interrupted long run.
            output.write_text(json.dumps({
                "config": dict(vars(args), device=str(args.device)),
                "complete": False, "trials": completed,
            }, indent=2))
            summary.write_text(json.dumps(compact(completed, args, False), indent=2))

    output.write_text(json.dumps({
        "config": dict(vars(args), device=str(args.device)),
        "complete": True, "trials": completed,
    }, indent=2))
    summary.write_text(json.dumps(compact(completed, args, True), indent=2))
    plot(completed, args, output.with_suffix(".pdf"))
    print(f"Saved {output}, {summary} and {output.with_suffix('.pdf')}")
    return completed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scales", default="0.03")
    p.add_argument("--couplings", default="0,0.2,1,3")
    p.add_argument("--windows", default="20,60", help="SGD steps per evolutionary proposal")
    p.add_argument("--seeds", default=",".join(map(str, range(12))))
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--units", type=int, default=4)
    p.add_argument("--width", type=int, default=2)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--rate", type=float, default=0.05)
    p.add_argument("--decay", type=float, default=0.001)
    p.add_argument("--mutation", type=float, default=0.2)
    p.add_argument("--evaluation", type=int, default=256)
    p.add_argument("--thresholds", default="0.2,0.1")
    p.add_argument("--device", default="cpu")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--output", default="results/microscopic_adaptive.json")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
