"""Two-panel record of the completed Arola-based specialization tests."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from style import apply_style, panel_label

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_suffix(".pdf")
apply_style()

fixed = json.loads((ROOT / "results" / "microscopic_speed_summary.json").read_text())
adaptive = json.loads((ROOT / "results" / "microscopic_adaptive_summary.json").read_text())
if not fixed.get("complete") or not adaptive.get("complete"):
    raise RuntimeError("Arola note requires complete committed result summaries")

group = next(
    g for g in fixed["groups"]
    if g["depth"] == 1 and g["init_scale"] == 0.03 and g["coupling"] == 1
)
trials = [
    t for t in adaptive["trials"]
    if t["scale"] == 0.03 and t["coupling"] == 1 and t["window"] == 20
]
if len(group["seeds"]) != 12 or len(trials) != 12:
    raise RuntimeError("Expected the 12-seed fixed and adaptive comparisons")

fig, axes = plt.subplots(2, 1, figsize=(5.6, 5.4))

top_steps = np.asarray(group["checkpoints"])
for name, label in (("uniform", "uniform"), ("specialists", "fixed specialists")):
    summary = group["individual_loss"][name]
    mean = np.asarray(summary["mean"])
    std = np.asarray(summary["std"])
    line, = axes[0].plot(top_steps, mean, label=label, lw=1.4)
    axes[0].plot(top_steps[::6], mean[::6], linestyle="", marker="o",
                 markersize=2.5, color=line.get_color())
    axes[0].fill_between(top_steps, np.maximum(0, mean - std), mean + std,
                         color=line.get_color(), alpha=0.16, linewidth=0)

bottom_steps = np.asarray([h["step"] for h in trials[0]["history"]])
for name, label in (
    ("uniform", "uniform"),
    ("specialists", "fixed specialists"),
    ("evolving", "evolving"),
    ("drift", "neutral drift"),
):
    curves = np.array([
        [h[name]["individual_loss"] for h in t["history"]] for t in trials
    ])
    mean, std = curves.mean(axis=0), curves.std(axis=0)
    line, = axes[1].plot(
        bottom_steps, mean, label=label, marker="o", markersize=2.5,
        markevery=3, lw=1.3
    )
    axes[1].fill_between(
        bottom_steps, np.maximum(0, mean - std), mean + std,
        color=line.get_color(), alpha=0.13, linewidth=0
    )

for ax, name in zip(axes, ("(a)", "(b)")):
    panel_label(ax, name)
    ax.axhline(0.2, color="0.5", ls=":", lw=0.9)
    ax.set_ylabel("Mean individual test loss")
    ax.set_ylim(bottom=0, top=0.58)
    ax.grid(False)
    ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=2)

axes[0].set_xlabel("SGD updates")
axes[1].set_xlabel("SGD updates")
fig.subplots_adjust(left=0.18, right=0.99, top=0.96, bottom=0.10, hspace=0.42)
fig.savefig(OUT, bbox_inches="tight")
plt.close(fig)
