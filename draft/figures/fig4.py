"""Time-resolved specialization in the sample-level neural dynamics."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from style import DEPTH_COLORS, apply_style, panel_label


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
pilot = completed(
    sorted((ROOT / "results").glob("adaptive_neural_pilot_seed*.json"))
)
files = full if full else (quick if quick else pilot)

if not files:
    OUT.unlink(missing_ok=True)
    raise SystemExit(0)

rows = []
for _, data in files:
    rows.extend(data["results"])

sigmas = np.array(sorted({row["sigma"] for row in rows}), dtype=float)
selected = [0, len(sigmas) // 2, len(sigmas) - 1]

fig, ax = plt.subplots(2, 1, figsize=(3.8, 4.8), sharex=True)

for k, idx in enumerate(selected):
    sigma = sigmas[idx]
    subset = [
        row for row in rows if np.isclose(row["sigma"], sigma)
    ]
    steps = np.asarray(subset[0]["history"]["step"], dtype=float)

    r_values = np.asarray(
        [row["history"]["R"] for row in subset],
        dtype=float,
    )
    s_values = np.asarray(
        [row["history"]["S_sample"] for row in subset],
        dtype=float,
    )

    r_mean = r_values.mean(axis=0)
    r_std = r_values.std(axis=0)
    s_mean = s_values.mean(axis=0)
    s_std = s_values.std(axis=0)

    color = DEPTH_COLORS[k]
    label = fr"$\sigma={sigma:g}$"

    ax[0].plot(
        steps / 1000.0,
        r_mean,
        color=color,
        linewidth=1.2,
        label=label,
    )
    ax[1].plot(
        steps / 1000.0,
        s_mean,
        color=color,
        linewidth=1.2,
    )

    if len(subset) > 1:
        ax[0].fill_between(
            steps / 1000.0,
            np.maximum(0.0, r_mean - r_std),
            r_mean + r_std,
            color=color,
            alpha=0.20,
            linewidth=0,
        )
        ax[1].fill_between(
            steps / 1000.0,
            np.maximum(0.0, s_mean - s_std),
            s_mean + s_std,
            color=color,
            alpha=0.20,
            linewidth=0,
        )

ax[0].set_ylabel(r"$R(t)$")
ax[1].set_xlabel(r"SGD steps ($\times 10^3$)")
ax[1].set_ylabel(r"$S_x(t)$")

ax[0].legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=3,
    frameon=False,
    columnspacing=0.8,
    handletextpad=0.3,
)

for axis in ax:
    axis.set_ylim(bottom=0)
    axis.grid(False)
    axis.tick_params(top=False, right=False)

panel_label(ax[0], "(a)")
panel_label(ax[1], "(b)")

fig.subplots_adjust(
    left=0.18,
    right=0.98,
    bottom=0.11,
    top=0.91,
    hspace=0.25,
)
fig.savefig(OUT, bbox_inches="tight")
plt.close(fig)
