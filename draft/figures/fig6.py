"""Paired long-time adaptive versus frozen-strategy neural trajectories."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from style import DEPTH_COLORS, apply_style, panel_label


apply_style()
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_suffix(".pdf")


def completed(path, frozen):
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    config = data.get("config", {})
    if not (
        data.get("complete")
        and config.get("version") == 2
        and config.get("frozen_strategy") is frozen
        and len(data.get("results", [])) == 1
    ):
        return None
    return data["results"][0]


pairs = []
for adaptive_path in sorted(
    (ROOT / "results").glob("adaptive_neural_long_seed*.json")
):
    suffix = adaptive_path.stem.replace("adaptive_neural_long_seed", "")
    frozen_path = (
        ROOT / "results" / f"adaptive_neural_frozen_seed{suffix}.json"
    )
    adaptive = completed(adaptive_path, False)
    frozen = completed(frozen_path, True)
    if adaptive is None or frozen is None:
        continue
    if not np.isclose(adaptive["sigma"], frozen["sigma"]):
        raise ValueError(f"mismatched coupling for seed {suffix}")
    if adaptive["steps_completed"] != frozen["steps_completed"]:
        raise ValueError(f"mismatched comparison length for seed {suffix}")
    pairs.append((adaptive, frozen))

if not pairs:
    OUT.unlink(missing_ok=True)
    raise SystemExit(0)

# Average only over time points observed in every paired realization.
n = min(
    len(row["history"]["step"])
    for pair in pairs
    for row in pair
)
step = np.asarray(pairs[0][0]["history"]["step"][:n], dtype=float)
for pair in pairs:
    for row in pair:
        current = np.asarray(row["history"]["step"][:n], dtype=float)
        if not np.array_equal(step, current):
            raise ValueError("paired runs have different observation times")

fig, ax = plt.subplots(2, 1, figsize=(3.8, 4.8), sharex=True)
conditions = (
    ("Adaptive", 0, DEPTH_COLORS[2], "-"),
    ("Frozen", 1, "0.45", "--"),
)

for name, j, color, linestyle in conditions:
    for index, key in enumerate(("R", "S_sample")):
        traces = np.asarray(
            [pair[j]["history"][key][:n] for pair in pairs],
            dtype=float,
        )
        mean = traces.mean(axis=0)
        std = traces.std(axis=0)
        ax[index].plot(
            step / 1000.0, mean, color=color, linestyle=linestyle,
            linewidth=1.25, label=name if index == 0 else None
        )
        if len(pairs) > 1:
            ax[index].fill_between(
                step / 1000.0,
                np.maximum(0.0, mean - std),
                mean + std,
                color=color,
                alpha=0.15,
                linewidth=0,
            )

ax[0].set_ylabel(r"$R(t)$")
ax[1].set_ylabel(r"$S_x(t)$")
ax[1].set_xlabel(r"SGD steps ($\times 10^3$)")
ax[0].legend(
    loc="lower center", bbox_to_anchor=(0.5, 1.02),
    ncol=2, frameon=False, columnspacing=1.5,
)

for axis in ax:
    axis.set_xlim(0, step[-1] / 1000.0)
    axis.set_ylim(bottom=0)
    axis.grid(False)
    axis.tick_params(top=False, right=False)

panel_label(ax[0], "(a)")
panel_label(ax[1], "(b)")
fig.subplots_adjust(
    left=0.18, right=0.98, bottom=0.11, top=0.91, hspace=0.25
)
fig.savefig(OUT, bbox_inches="tight")
plt.close(fig)

statuses = [pair[0].get("stop_reason") for pair in pairs]
print(
    f"fig6: {len(pairs)} paired seeds, sigma={pairs[0][0]['sigma']:g}, "
    f"common horizon={int(step[-1])}, stop reasons={statuses}"
)
