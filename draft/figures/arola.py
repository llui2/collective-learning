"""Allocation polarization and learning benefit in the adaptive pilot."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from style import apply_style, panel_label

apply_style()

ROOT = Path(__file__).resolve().parents[2]
summary = json.loads((ROOT / "results/microscopic_adaptive_summary.json").read_text())
trials = [
    trial for trial in summary["trials"]
    if trial["scale"] == 0.03
    and trial["coupling"] == 1
    and trial["window"] == 20
]
if not summary["complete"] or len(trials) != 12:
    raise ValueError("Expected 12 complete paired allocation trials")

steps = np.array([row["step"] for row in trials[0]["history"]])
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))

for strategy, label, color in (
    ("specialists", "fixed specialists", "C0"),
    ("evolving", "selected mutations", "C1"),
    ("drift", "neutral drift", "0.5"),
):
    polarization = np.array([
        [row[strategy]["polarization"] for row in trial["history"]]
        for trial in trials
    ])
    advantage = np.array([
        [row["uniform"]["individual_loss"] - row[strategy]["individual_loss"]
         for row in trial["history"]]
        for trial in trials
    ])
    for ax, values in zip(axes, (polarization, advantage)):
        mean, std = values.mean(axis=0), values.std(axis=0)
        line, = ax.plot(steps, mean, color=color, lw=1.45, label=label)
        ax.plot(steps[::3], mean[::3], linestyle="", marker="o",
                markersize=2.4, color=line.get_color())
        if strategy != "specialists":
            ax.fill_between(steps, mean - std, mean + std,
                            color=color, alpha=0.12, linewidth=0)

axes[0].set_ylabel("Allocation polarization")
axes[0].set_ylim(-0.05, 1.07)
axes[1].set_ylabel("Improvement in test loss")
axes[1].axhline(0, color="0.35", linestyle=":", lw=0.85)
axes[1].set_ylim(-0.08, 0.18)

for ax, label in zip(axes, ("(a)", "(b)")):
    ax.set_xlim(0, 400)
    ax.set_xlabel("Training steps")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)
    panel_label(ax, label)

axes[0].legend(loc="center left", frameon=False, fontsize=8)
fig.subplots_adjust(left=0.12, right=0.99, bottom=0.23, top=0.92, wspace=0.36)
fig.savefig(Path(__file__).with_suffix(".pdf"), bbox_inches="tight")
plt.close(fig)
