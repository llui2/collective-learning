"""Differentiation and routing entropy in the neural performance experiment."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from style import DEPTH_COLORS, DEPTH_MARKERS, MARKERSIZE, apply_style, panel_label


apply_style()
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_suffix(".pdf")

data = []
for path in sorted((ROOT / "results").glob("neural_performance_*.json")):
    item = json.loads(path.read_text())
    if item.get("complete") and item.get("version") == 1:
        data.append(item)

if not data:
    OUT.unlink(missing_ok=True)
    raise SystemExit(0)

steps = max(item["config"]["steps"] for item in data)
data = [item for item in data if item["config"]["steps"] == steps]
points = defaultdict(list)
for item in data:
    cfg = item["config"]
    if cfg["control"] != "adaptive":
        continue
    last = item["validation"][-1]
    points[(float(cfg["strategy_rate"]), float(cfg["sigma"]))].append(
        (last["R"], last["routing_entropy"])
    )

if not points:
    OUT.unlink(missing_ok=True)
    raise SystemExit(0)

fig, ax = plt.subplots(2, 1, figsize=(3.8, 4.7), sharex=True)
rates = sorted({k[0] for k in points})
for k, rate in enumerate(rates):
    sigmas = sorted(s for r, s in points if r == rate)
    x = np.log10(sigmas)
    mean = np.array([
        np.asarray(points[(rate, s)]).mean(axis=0) for s in sigmas
    ])
    std = np.array([
        np.asarray(points[(rate, s)]).std(axis=0) for s in sigmas
    ])
    color = DEPTH_COLORS[k % len(DEPTH_COLORS)]
    marker = DEPTH_MARKERS[k % len(DEPTH_MARKERS)]
    for j in range(2):
        ax[j].plot(
            x, mean[:, j], color=color, marker=marker,
            linewidth=1.0, markersize=0.82 * MARKERSIZE,
            label=fr"$r={rate:g}$" if j == 0 else None,
        )
        if any(len(points[(rate, s)]) > 1 for s in sigmas):
            ax[j].fill_between(
                x, mean[:, j] - std[:, j], mean[:, j] + std[:, j],
                color=color, alpha=0.22, linewidth=0,
            )
for a in ax:
    a.grid(False)
    a.tick_params(top=False, right=False)
ax[0].set_ylabel(r"$R$")
ax[1].set_ylabel(r"$H$")
ax[1].set_xlabel(r"$\log_{10}\sigma$")
ax[0].set_ylim(bottom=0)
ax[1].set_ylim(0, 1.03)
ax[0].legend(
    loc="lower center", bbox_to_anchor=(0.5, 1.02),
    ncol=len(rates), frameon=False, columnspacing=0.8,
    handletextpad=0.3,
)
panel_label(ax[0], "(a)")
panel_label(ax[1], "(b)")
fig.subplots_adjust(left=0.18, right=0.98, bottom=0.11, top=0.91, hspace=0.25)
fig.savefig(OUT, bbox_inches="tight")
plt.close(fig)
print(f"fig9: {len(rates)} rates, {steps} SGD steps")
