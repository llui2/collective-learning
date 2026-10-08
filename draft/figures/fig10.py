"""Time-resolved collective gain and differentiation at intermediate coupling."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from style import DEPTH_COLORS, apply_style, panel_label


apply_style()
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_suffix(".pdf")
TARGET_SIGMA = 0.1

items = []
for path in sorted((ROOT / "results").glob("neural_performance_*.json")):
    value = json.loads(path.read_text())
    if value.get("complete") and value.get("version") == 1:
        items.append(value)

available = [
    item for item in items
    if np.isclose(item["config"]["sigma"], TARGET_SIGMA)
]
if not available:
    OUT.unlink(missing_ok=True)
    raise SystemExit(0)

steps = max(item["config"]["steps"] for item in available)
available = [
    item for item in available if item["config"]["steps"] == steps
]
uniform = {
    item["config"]["seed"]: item
    for item in available if item["config"]["control"] == "uniform"
}
points = defaultdict(list)
for item in available:
    cfg = item["config"]
    if cfg["control"] != "adaptive" or cfg["seed"] not in uniform:
        continue
    ref = uniform[cfg["seed"]]["validation"]
    new = item["validation"]
    if [a["step"] for a in ref] != [a["step"] for a in new]:
        raise ValueError("time grids of matched controls differ")

    group = (float(cfg["strategy_rate"]), float(cfg["exploration"]))
    points[group].append((
        np.array([row["step"] for row in new]),
        np.array([
            a["collective_loss"] - b["collective_loss"]
            for a, b in zip(ref, new)
        ]),
        np.array([row["R"] for row in new]),
    ))

if not points:
    OUT.unlink(missing_ok=True)
    raise SystemExit(0)

rates = sorted({r for r, _ in points})
explorations = sorted({e for _, e in points})
fig, ax = plt.subplots(2, 1, figsize=(3.8, 4.8), sharex=True)
for rate, exploration in sorted(points):
    records = points[(rate, exploration)]
    time = records[0][0] / 1000.0
    if any(not np.array_equal(row[0], records[0][0]) for row in records):
        raise ValueError("time grids differ across seeds")
    color = DEPTH_COLORS[rates.index(rate) % len(DEPTH_COLORS)]
    ls = "-" if exploration == explorations[0] else "--"
    for j in range(2):
        arr = np.array([row[j + 1] for row in records])
        mean, std = arr.mean(axis=0), arr.std(axis=0)
        ax[j].plot(
            time, mean, color=color, linestyle=ls, linewidth=1.1,
            label=fr"$r={rate:g},\,\nu={exploration:g}$" if j == 0 else None,
        )
        if len(records) > 1:
            ax[j].fill_between(
                time, mean - std, mean + std, alpha=0.18,
                color=color, linewidth=0,
            )

ax[0].axhline(0, color="0.5", linestyle=":", linewidth=0.8)
ax[0].set_ylabel(r"$L_{\mathrm{uniform}}-L_{\mathrm{adaptive}}$")
ax[1].set_ylabel(r"$R(t)$")
ax[1].set_xlabel(r"SGD steps ($\times 10^3$)")
ax[1].set_ylim(bottom=0)
ax[0].legend(
    loc="lower center", bbox_to_anchor=(0.5, 1.02),
    ncol=len(rates), frameon=False, columnspacing=0.5,
    handletextpad=0.3,
)

for a in ax:
    a.grid(False)
    a.tick_params(top=False, right=False)
panel_label(ax[0], "(a)")
panel_label(ax[1], "(b)")
fig.subplots_adjust(left=0.26, right=0.98, bottom=0.11, top=0.84, hspace=0.26)
fig.savefig(OUT, bbox_inches="tight")
plt.close(fig)
print(f"fig10: sigma={TARGET_SIGMA}, {len(points)} strategies, {steps} steps")
