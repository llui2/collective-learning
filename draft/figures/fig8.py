"""Neural collective-loss and accuracy gains relative to matched uniform learning."""

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

# Never pool evaluations made at different optimization horizons.
steps = max(item["config"]["steps"] for item in data)
data = [item for item in data if item["config"]["steps"] == steps]
controls = {
    (item["config"]["seed"], item["config"]["sigma"], item["config"]["control"]): item
    for item in data if item["config"]["control"] != "adaptive"
}
points = defaultdict(list)

for item in data:
    cfg = item["config"]
    if cfg["control"] != "adaptive":
        continue
    key = (cfg["seed"], cfg["sigma"], "uniform")
    if key not in controls:
        continue
    uniform = controls[key]["validation"][-1]
    adaptive = item["validation"][-1]
    rate = float(cfg["strategy_rate"])
    sigma = float(cfg["sigma"])
    points[(rate, float(cfg["exploration"]), sigma)].append((
        uniform["collective_loss"] - adaptive["collective_loss"],
        100.0 * (adaptive["ensemble_accuracy"] - uniform["ensemble_accuracy"]),
    ))

if not points:
    OUT.unlink(missing_ok=True)
    raise SystemExit(0)

conditions = sorted({(k[0], k[1]) for k in points})
rates = sorted({r for r, _ in conditions})
explorations = sorted({e for _, e in conditions})
fig, ax = plt.subplots(2, 1, figsize=(3.8, 4.7), sharex=True)

for rate, exploration in conditions:
    sigmas = sorted(sigma for r, e, sigma in points if r == rate and e == exploration)
    x = np.log10(sigmas)
    values = np.array([
        np.asarray(points[(rate, exploration, sigma)]).mean(axis=0)
        for sigma in sigmas
    ])
    deviations = np.array([
        np.asarray(points[(rate, exploration, sigma)]).std(axis=0)
        for sigma in sigmas
    ])
    k = rates.index(rate)
    color = DEPTH_COLORS[k % len(DEPTH_COLORS)]
    marker = DEPTH_MARKERS[k % len(DEPTH_MARKERS)]
    linestyle = "-" if exploration == explorations[0] else "--"

    for j in range(2):
        ax[j].plot(
            x, values[:, j], color=color, marker=marker,
            linewidth=1.0, linestyle=linestyle,
            markersize=0.82 * MARKERSIZE,
            label=fr"$r={rate:g},\,\nu={exploration:g}$" if j == 0 else None,
        )
        if any(len(points[(rate, exploration, sigma)]) > 1 for sigma in sigmas):
            ax[j].fill_between(
                x, values[:, j] - deviations[:, j],
                values[:, j] + deviations[:, j],
                color=color, alpha=0.22, linewidth=0,
            )

for a in ax:
    a.axhline(0, color="0.5", linewidth=0.8, linestyle="--")
    a.tick_params(top=False, right=False)
    a.grid(False)

ax[0].set_ylabel(r"$L_{\mathrm{uniform}}-L_{\mathrm{adaptive}}$")
ax[1].set_ylabel(r"$A_{\mathrm{adaptive}}-A_{\mathrm{uniform}}$ (pp)")
ax[1].set_xlabel(r"$\log_{10}\sigma$")
ax[0].legend(
    loc="lower center", bbox_to_anchor=(0.5, 1.02),
    ncol=len(rates), frameon=False, columnspacing=0.6,
    handletextpad=0.3,
)
panel_label(ax[0], "(a)")
panel_label(ax[1], "(b)")
fig.subplots_adjust(left=0.26, right=0.98, bottom=0.11, top=0.84, hspace=0.26)
fig.savefig(OUT, bbox_inches="tight")
plt.close(fig)
print(f"fig8: {len(rates)} adaptation rates, {steps} SGD steps, validation results")
