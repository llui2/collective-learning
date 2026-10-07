"""Sample-level adaptive neural dynamics."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from style import DEPTH_COLORS, DEPTH_MARKERS, MARKERSIZE, apply_style, panel_label


apply_style()

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_suffix(".pdf")
VERSION = 2


def completed(paths):
    keep = []
    for path in paths:
        data = json.loads(path.read_text())
        config = data.get("config", {})
        if (
            data.get("complete") is True
            and int(config.get("version", -1)) == VERSION
        ):
            keep.append((path, data))
    return keep


full = completed(
    sorted((ROOT / "results").glob("adaptive_neural_seed*.json"))
)
quick = completed(
    sorted((ROOT / "results").glob("adaptive_neural_quick_seed*.json"))
)
files = full if full else quick

if not files:
    OUT.unlink(missing_ok=True)
    raise SystemExit(0)

rows = []
for _, data in files:
    rows.extend(data["results"])

sigmas = np.array(sorted({row["sigma"] for row in rows}), dtype=float)
x_sigma = np.log10(sigmas)

fig, ax = plt.subplots(2, 1, figsize=(3.8, 4.8))

selected = [0, len(sigmas) // 2, len(sigmas) - 1]
for k, idx in enumerate(selected):
    sigma = sigmas[idx]
    subset = [
        row for row in rows if np.isclose(row["sigma"], sigma)
    ]
    steps = np.asarray(subset[0]["history"]["step"], dtype=float)
    histories = np.asarray(
        [row["history"]["G"] for row in subset],
        dtype=float,
    )
    mean = histories.mean(axis=0)
    std = histories.std(axis=0)

    color = DEPTH_COLORS[k]
    ax[0].plot(
        steps / 1000.0,
        mean,
        color=color,
        linewidth=1.2,
        label=fr"$\sigma={sigma:g}$",
    )
    if len(subset) > 1:
        ax[0].fill_between(
            steps / 1000.0,
            np.maximum(0.0, mean - std),
            mean + std,
            color=color,
            alpha=0.20,
            linewidth=0,
        )

g_mean = []
g_std = []
for sigma in sigmas:
    subset = [
        row for row in rows if np.isclose(row["sigma"], sigma)
    ]
    values = np.asarray([row["G"] for row in subset], dtype=float)
    g_mean.append(values.mean())
    g_std.append(values.std())

g_mean = np.asarray(g_mean)
g_std = np.asarray(g_std)

ax[1].plot(
    x_sigma,
    g_mean,
    color=DEPTH_COLORS[2],
    marker=DEPTH_MARKERS[2],
    markersize=0.82 * MARKERSIZE,
    linewidth=1.0,
)
if len(files) > 1:
    ax[1].fill_between(
        x_sigma,
        np.maximum(0.0, g_mean - g_std),
        g_mean + g_std,
        color=DEPTH_COLORS[2],
        alpha=0.20,
        linewidth=0,
    )

ax[0].set_xlabel(r"SGD steps ($\times 10^3$)")
ax[0].set_ylabel(r"$G(t)$")
ax[0].legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=3,
    frameon=False,
    columnspacing=0.8,
    handletextpad=0.3,
)

ax[1].set_xlabel(r"$\log_{10}\sigma$")
ax[1].set_ylabel(r"$G$")

for axis in ax:
    axis.set_ylim(bottom=0)
    axis.grid(False)
    axis.tick_params(top=False, right=False)

ax[1].set_xlim(x_sigma.min(), x_sigma.max())

panel_label(ax[0], "(a)")
panel_label(ax[1], "(b)")

fig.subplots_adjust(
    left=0.18,
    right=0.98,
    bottom=0.11,
    top=0.91,
    hspace=0.34,
)
fig.savefig(OUT, bbox_inches="tight")
plt.close(fig)
