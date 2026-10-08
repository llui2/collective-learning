"""Paired learning advantage for imposed and adaptive task allocation."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from style import apply_style, panel_label

apply_style()

ROOT = Path(__file__).resolve().parents[2]
fixed = json.loads((ROOT / "results/microscopic_speed_summary.json").read_text())
adaptive = json.loads((ROOT / "results/microscopic_adaptive_summary.json").read_text())

group = next(g for g in fixed["groups"]
             if g["depth"] == 1 and g["init_scale"] == 0.03 and g["coupling"] == 1)
trials = [t for t in adaptive["trials"]
          if t["scale"] == 0.03 and t["coupling"] == 1 and t["window"] == 20]
if len(group["seeds"]) != 12 or len(trials) != 12:
    raise ValueError("Expected 12 paired trials for each comparison")

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharex=True, sharey=True)

steps = np.asarray(group["checkpoints"])
difference = (np.asarray(group["individual_loss"]["uniform"]["mean"])
              - np.asarray(group["individual_loss"]["specialists"]["mean"]))
axes[0].plot(steps, difference, color="C0", linewidth=1.6)
axes[0].set_title("Fixed allocations", fontsize=10)

steps = np.array([h["step"] for h in trials[0]["history"]])
for strategy, label, color in (
    ("specialists", "fixed specialists", "C0"),
    ("evolving", "selected mutations", "C1"),
    ("drift", "neutral drift", "0.55"),
):
    values = np.array([
        [h["uniform"]["individual_loss"] - h[strategy]["individual_loss"]
         for h in trial["history"]]
        for trial in trials
    ])
    axes[1].plot(steps, values.mean(axis=0), color=color,
                 linewidth=1.6, label=label)
axes[1].set_title("Evolving allocations", fontsize=10)
axes[1].legend(frameon=False, loc="upper right", fontsize=8)

for ax, letter in zip(axes, ("(a)", "(b)")):
    ax.axhline(0, color="0.35", linestyle=":", linewidth=0.85)
    ax.set_xlim(0, 400)
    ax.set_ylim(-0.065, 0.18)
    ax.set_xlabel("Training steps")
    ax.spines[["top", "right"]].set_visible(False)
    panel_label(ax, letter)
axes[0].set_ylabel("Loss improvement over generalists")

fig.subplots_adjust(left=0.12, right=0.99, bottom=0.22, top=0.85, wspace=0.14)
fig.savefig(Path(__file__).with_suffix(".pdf"), bbox_inches="tight")
plt.close(fig)
