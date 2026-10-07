"""Adaptive neural dynamics in a common MNIST environment."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from style import DEPTH_COLORS, MARKERSIZE, apply_style, panel_label


apply_style()

ROOT = Path(__file__).resolve().parents[2]


def completed(paths):
    keep = []
    for path in paths:
        data = json.loads(path.read_text())
        if data.get("complete") is True:
            keep.append((path, data))
    return keep


full = completed(sorted((ROOT / "results").glob("adaptive_neural_seed*.json")))
quick = completed(
    sorted((ROOT / "results").glob("adaptive_neural_quick_seed*.json"))
)
files = full if full else quick
if not files:
    raise SystemExit("no completed adaptive neural result files")

rows = []
for _, data in files:
    rows.extend(data["results"])

sigmas = sorted({row["sigma"] for row in rows})
x = np.log10(sigmas)

g_mean, g_std = [], []
s_mean, s_std = [], []

for sigma in sigmas:
    subset = [
        row for row in rows if np.isclose(row["sigma"], sigma)
    ]
    g = np.asarray([row["G"] for row in subset], dtype=float)
    s = np.asarray([row["S"] for row in subset], dtype=float)
    g_mean.append(g.mean())
    g_std.append(g.std())
    s_mean.append(s.mean())
    s_std.append(s.std())

g_mean = np.asarray(g_mean)
g_std = np.asarray(g_std)
s_mean = np.asarray(s_mean)
s_std = np.asarray(s_std)

fig, ax = plt.subplots(2, 1, figsize=(3.8, 4.8), sharex=True)
color = DEPTH_COLORS[2]

ax[0].plot(
    x,
    g_mean,
    color=color,
    marker="o",
    markersize=0.82 * MARKERSIZE,
    linewidth=1.0,
)
ax[0].fill_between(
    x,
    np.maximum(0.0, g_mean - g_std),
    g_mean + g_std,
    color=color,
    alpha=0.22,
    linewidth=0,
)

ax[1].plot(
    x,
    s_mean,
    color=color,
    marker="o",
    markersize=0.82 * MARKERSIZE,
    linewidth=1.0,
)
ax[1].fill_between(
    x,
    np.maximum(0.0, s_mean - s_std),
    s_mean + s_std,
    color=color,
    alpha=0.22,
    linewidth=0,
)

ax[0].set_ylabel(r"$G$")
ax[1].set_xlabel(r"$\log_{10}\sigma$")
ax[1].set_ylabel(r"$S$")

for axis in ax:
    axis.set_xlim(x.min(), x.max())
    axis.set_ylim(bottom=0)
    axis.grid(False)
    axis.tick_params(top=False, right=False)

panel_label(ax[0], "(a)")
panel_label(ax[1], "(b)")

fig.subplots_adjust(
    left=0.18,
    right=0.98,
    bottom=0.11,
    top=0.97,
    hspace=0.25,
)
fig.savefig(Path(__file__).with_suffix(".pdf"), bbox_inches="tight")
plt.close(fig)
