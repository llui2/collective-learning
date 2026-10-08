"""Performance gain and learning-allocation covariance in the effective theory."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from style import DEPTH_COLORS, DEPTH_MARKERS, MARKERSIZE, apply_style, panel_label


apply_style()

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_suffix(".pdf")
data = np.load(ROOT / "results" / "adaptive_theory.npz")

if int(data["version"]) != 7:
    OUT.unlink(missing_ok=True)
    raise SystemExit("performance figure requires adaptive theory version 7")

x = np.log10(data["sigmas"])
nus = data["mutation_rates"]

fig, ax = plt.subplots(2, 1, figsize=(3.8, 4.8), sharex=True)

for k, nu in enumerate(nus):
    color = DEPTH_COLORS[k]
    marker = DEPTH_MARKERS[k]

    for axis, value, uncertainty in (
        (ax[0], data["collective_gain_mean"][k], data["collective_gain_std"][k]),
        (ax[1], data["covariance_mean"][k], data["covariance_std"][k]),
    ):
        axis.plot(
            x, value, color=color, marker=marker, linestyle="",
            markersize=0.82 * MARKERSIZE,
            label=fr"$\nu={nu:g}$" if axis is ax[0] else None,
        )
        axis.fill_between(
            x, value - uncertainty, value + uncertainty,
            color=color, alpha=0.22, linewidth=0,
        )

ax[0].axhline(0, color="0.45", linestyle="--", linewidth=0.8)
ax[1].axhline(0, color="0.45", linestyle="--", linewidth=0.8)

ax[0].set_ylabel(r"$L_0-L$")
ax[1].set_ylabel(r"$\sum_\mu\operatorname{Cov}(a_\mu,m_\mu)$")
ax[1].set_xlabel(r"$\log_{10}\sigma$")

ax[0].legend(
    loc="lower center", bbox_to_anchor=(0.5, 1.02),
    ncol=len(nus), frameon=False, columnspacing=0.9, handletextpad=0.3,
)

for axis in ax:
    axis.set_xlim(x.min(), x.max())
    axis.set_xticks([-2, -1, 0, 1])
    axis.grid(False)
    axis.tick_params(top=False, right=False)

panel_label(ax[0], "(a)")
panel_label(ax[1], "(b)")

fig.subplots_adjust(left=0.21, right=0.98, bottom=0.11, top=0.91, hspace=0.26)
fig.savefig(OUT, bbox_inches="tight")
plt.close(fig)
