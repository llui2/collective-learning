"""Finite-time coupling dependence in the microscopic neural experiment."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from style import DEPTH_COLORS, DEPTH_MARKERS, MARKERSIZE, apply_style, panel_label


apply_style()
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_suffix(".pdf")

files = sorted((ROOT / "results").glob("adaptive_neural_quick_seed*.json"))
runs = []
for path in files:
    data = json.loads(path.read_text())
    if data.get("complete") and data.get("config", {}).get("version") == 2:
        runs.append(data)

if not runs:
    OUT.unlink(missing_ok=True)
    raise SystemExit(0)

steps = {int(run["config"]["steps"]) for run in runs}
if len(steps) != 1:
    raise ValueError("quick neural sweeps must use the same observation time")

sigmas = np.array([row["sigma"] for row in runs[0]["results"]], dtype=float)
values = np.zeros((len(runs), len(sigmas), 2))
for k, run in enumerate(runs):
    rows = run["results"]
    current = np.array([row["sigma"] for row in rows], dtype=float)
    if current.shape != sigmas.shape or not np.allclose(current, sigmas):
        raise ValueError("coupling grids differ between neural seeds")
    if any(
        int(row.get("steps_completed", run["config"]["steps"]))
        != run["config"]["steps"]
        for row in rows
    ):
        raise ValueError("incomplete SGD trajectory")
    values[k, :, 0] = [row["R_test"] for row in rows]
    values[k, :, 1] = [row["S_sample_test"] for row in rows]

x = np.log10(sigmas)
fig, ax = plt.subplots(2, 1, figsize=(3.8, 4.6), sharex=True)
color = DEPTH_COLORS[2]

for idx, symbol in enumerate((r"$R$", r"$S_x$")):
    mean = values[:, :, idx].mean(axis=0)
    std = values[:, :, idx].std(axis=0)
    ax[idx].plot(
        x, mean, color=color, marker=DEPTH_MARKERS[2],
        markersize=0.82 * MARKERSIZE, linewidth=1.1
    )
    if len(runs) > 1:
        ax[idx].fill_between(
            x, np.maximum(0.0, mean - std), mean + std,
            color=color, alpha=0.20, linewidth=0
        )
    ax[idx].set_ylabel(symbol)
    ax[idx].set_ylim(bottom=0)
    ax[idx].set_xlim(x.min(), x.max())
    ax[idx].grid(False)
    ax[idx].tick_params(top=False, right=False)

ax[1].set_xlabel(r"$\log_{10}\sigma$")
ax[1].set_xticks([-3, -2, -1, 0, 1])
panel_label(ax[0], "(a)")
panel_label(ax[1], "(b)")
fig.subplots_adjust(left=0.18, right=0.98, bottom=0.11, top=0.97, hspace=0.25)
fig.savefig(OUT, bbox_inches="tight")
plt.close(fig)

print(
    f"fig5: {len(runs)} seeds, {len(sigmas)} couplings, "
    f"{next(iter(steps))} SGD steps"
)
