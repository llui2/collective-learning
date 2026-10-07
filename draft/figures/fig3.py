"""Symmetry breaking in the adaptive effective theory."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from style import DEPTH_COLORS, DEPTH_MARKERS, MARKERSIZE, apply_style, panel_label


apply_style()

ROOT = Path(__file__).resolve().parents[2]
data = np.load(ROOT / "results" / "adaptive_theory.npz")
OUT = Path(__file__).with_suffix(".pdf")

sigmas = data["sigmas"]
nus = data["mutation_rates"]
g_mean = data["genotype_mean"]
g_std = data["genotype_std"]
s_mean = data["phenotype_mean"]
s_std = data["phenotype_std"]
sigma_critical = data["sigma_critical"]

x = np.log10(sigmas)

fig, ax = plt.subplots(1, 2, figsize=(7.3, 2.75))

for k, nu in enumerate(nus):
    color = DEPTH_COLORS[k]
    marker = DEPTH_MARKERS[k]

    ax[0].plot(
        x,
        g_mean[k],
        color=color,
        marker=marker,
        linestyle="",
        markersize=0.82 * MARKERSIZE,
        label=fr"$\nu={nu:g}$",
    )
    ax[0].fill_between(
        x,
        np.maximum(0.0, g_mean[k] - g_std[k]),
        g_mean[k] + g_std[k],
        color=color,
        alpha=0.22,
        linewidth=0,
    )

    ax[1].plot(
        x,
        s_mean[k],
        color=color,
        marker=marker,
        linestyle="",
        markersize=0.82 * MARKERSIZE,
    )
    ax[1].fill_between(
        x,
        np.maximum(0.0, s_mean[k] - s_std[k]),
        s_mean[k] + s_std[k],
        color=color,
        alpha=0.22,
        linewidth=0,
    )

    if sigma_critical[k] > sigmas.min():
        xc = np.log10(sigma_critical[k])
        if xc <= x.max():
            ax[0].axvline(xc, color=color, linestyle=":", linewidth=0.9, alpha=0.75)
            ax[1].axvline(xc, color=color, linestyle=":", linewidth=0.9, alpha=0.75)

ax[0].set_xlabel(r"$\log_{10}\sigma$")
ax[0].set_ylabel(r"$G$")

ax[1].set_xlabel(r"$\log_{10}\sigma$")
ax[1].set_ylabel(r"$S$")

handles, labels = ax[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=len(labels),
    frameon=False,
    bbox_to_anchor=(0.5, 1.01),
    columnspacing=0.9,
    handletextpad=0.3,
)

for axis in ax:
    axis.set_xlim(x.min(), x.max())
    axis.set_ylim(bottom=0)
    axis.set_xticks([-2, -1, 0, 1])
    axis.grid(False)
    axis.tick_params(top=False, right=False)

panel_label(ax[0], "(a)")
panel_label(ax[1], "(b)")

fig.subplots_adjust(left=0.11, right=0.99, bottom=0.22, top=0.84, wspace=0.28)
fig.savefig(OUT, bbox_inches="tight")
plt.close(fig)
