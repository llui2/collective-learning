"""Generate the main coupling-sweep figure from matched-seed runs."""

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "draft" / "figures" / "fig1.pdf"
CSV_OUT = ROOT / "draft" / "figures" / "fig1.csv"

LAMBDAS = [0.0, 0.25, 0.5, 1.0, 2.0]
SEEDS = [0, 1, 2, 3, 4]
MIN_STEP = 200


def result_path(lambda_coupling, seed):
    label = f"{lambda_coupling:g}"
    return RESULTS / f"sweep_lambda{label}_seed{seed}.json"


def load_run(lambda_coupling, seed):
    path = result_path(lambda_coupling, seed)
    if not path.exists():
        raise FileNotFoundError(path)

    data = json.loads(path.read_text())
    config = data["config"]

    if float(config["lambda_coupling"]) != lambda_coupling:
        raise ValueError(f"lambda mismatch in {path}")
    if int(config["seed"]) != seed:
        raise ValueError(f"seed mismatch in {path}")

    return {row["step"]: row for row in data["history"]}


runs = {
    (lambda_coupling, seed): load_run(lambda_coupling, seed)
    for lambda_coupling in LAMBDAS
    for seed in SEEDS
}

common_steps = None
for history in runs.values():
    steps = {step for step in history if step >= MIN_STEP}
    common_steps = steps if common_steps is None else common_steps & steps

steps = sorted(common_steps)
if not steps:
    raise RuntimeError("No common evaluation steps found across the sweep.")

final_step = steps[-1]

OUT.parent.mkdir(parents=True, exist_ok=True)

with CSV_OUT.open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(
        [
            "lambda",
            "seed",
            "step",
            "loss",
            "baseline_loss",
            "delta_loss",
            "entropy",
            "kl",
            "coupling_norm",
            "load_cv",
        ]
    )

    for lambda_coupling in LAMBDAS:
        for seed in SEEDS:
            baseline = runs[(0.0, seed)]
            history = runs[(lambda_coupling, seed)]
            for step in steps:
                row = history[step]
                baseline_loss = baseline[step]["loss"]
                writer.writerow(
                    [
                        lambda_coupling,
                        seed,
                        step,
                        row["loss"],
                        baseline_loss,
                        row["loss"] - baseline_loss,
                        row["entropy"],
                        row["kl"],
                        row["coupling_norm"],
                        row["load_cv"],
                    ]
                )

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

ax = axes[0]
for lambda_coupling in LAMBDAS:
    values = np.array(
        [
            [runs[(lambda_coupling, seed)][step]["loss"] for step in steps]
            for seed in SEEDS
        ]
    )
    mean = values.mean(axis=0)
    sem = values.std(axis=0, ddof=1) / math.sqrt(len(SEEDS))

    line = ax.plot(
        steps,
        mean,
        lw=1.5,
        label=rf"$\lambda={lambda_coupling:g}$",
    )[0]
    ax.fill_between(
        steps,
        mean - sem,
        mean + sem,
        alpha=0.14,
        color=line.get_color(),
        linewidth=0,
    )

ax.set_xlabel("training step")
ax.set_ylabel("validation loss")
ax.legend(frameon=False, ncol=2, fontsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.text(-0.16, 1.04, "(a)", transform=ax.transAxes, fontsize=11)

ax = axes[1]
means = []
sems = []

for lambda_coupling in LAMBDAS:
    deltas = np.array(
        [
            runs[(lambda_coupling, seed)][final_step]["loss"]
            - runs[(0.0, seed)][final_step]["loss"]
            for seed in SEEDS
        ]
    )
    means.append(deltas.mean())
    sems.append(deltas.std(ddof=1) / math.sqrt(len(SEEDS)))

    ax.scatter(
        np.full(len(SEEDS), lambda_coupling),
        deltas,
        s=14,
        color="0.65",
        alpha=0.7,
        zorder=2,
    )

ax.axhline(0.0, lw=0.8, ls="--", color="0.35")
ax.errorbar(
    LAMBDAS,
    means,
    yerr=sems,
    marker="o",
    lw=1.4,
    capsize=2.5,
    color="black",
    zorder=3,
)
ax.set_xlabel(r"coupling $\lambda$")
ax.set_ylabel(r"final $\Delta\mathcal{L}$")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.text(-0.16, 1.04, "(b)", transform=ax.transAxes, fontsize=11)

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
plt.close(fig)

print(f"final matched step: {final_step}")
print(f"wrote {OUT.relative_to(ROOT)}")
print(f"wrote {CSV_OUT.relative_to(ROOT)}")
