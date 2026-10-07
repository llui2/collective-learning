"""Adaptive effective theory: genotype and phenotype specialization."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from style import DEPTH_COLORS, DEPTH_MARKERS, MARKERSIZE, apply_style, panel_label


apply_style()

ROOT = Path(__file__).resolve().parents[2]
data = np.load(ROOT / "results" / "adaptive_theory.npz")
OUT = Path(__file__).with_suffix(".pdf")

sigmas = data["sigmas"]
nus = data["mutation_rates"]
genotype_mean = data["genotype_mean"]
genotype_std = data["genotype_std"]
phenotype_mean = data["phenotype_mean"]
phenotype_std = data["phenotype_std"]
sigma_critical = data["sigma_critical"]

x = np.log10(sigmas)

fig, ax = plt.subplots(1, 2, figsize=(7.3, 2.75))

for k, nu in enumerate(nus):
    color = DEPTH_COLORS[k]
    marker = DEPTH_MARKERS[k]
    label = fr"$\nu={nu:g}$"

    ax[0].plot(
        x,
        genotype_mean[k],
        color=color,
        marker=marker,
        linestyle="",
        markersize=MARKERSIZE,
        label=label,
    )
    ax[0].fill_between(
        x,
        np.maximum(0.0, genotype_mean[k] - genotype_std[k]),
        genotype_mean[k] + genotype_std[k],
        color=color,
        alpha=0.22,
        linewidth=0,
    )

    ax[1].plot(
        x,
        phenotype_mean[k],
        color=color,
        marker=marker,
        linestyle="",
        markersize=MARKERSIZE,
    )
    ax[1].fill_between(
        x,
        np.maximum(0.0, phenotype_mean[k] - phenotype_std[k]),
        phenotype_mean[k] + phenotype_std[k],
        color=color,
        alpha=0.22,
        linewidth=0,
    )

    if sigma_critical[k] > 0:
        xc = np.log10(sigma_critical[k])
        ax[0].axvline(xc, color=color, linestyle=":", linewidth=1.0)
        ax[1].axvline(xc, color=color, linestyle=":", linewidth=1.0)

ax[0].set_xlabel(r"$\log_{10}\sigma$")
ax[0].set_ylabel(r"$G$")
ax[0].legend(loc="upper right")

ax[1].set_xlabel(r"$\log_{10}\sigma$")
ax[1].set_ylabel(r"$S$")

threshold = Line2D([0], [0], color="black", linestyle=":", linewidth=1.0, label=r"$\sigma_c$")
handles, labels = ax[0].get_legend_handles_labels()
ax[0].legend(handles + [threshold], labels + [r"$\sigma_c$"], loc="upper right")

for axis in ax:
    axis.set_xlim(x.min(), x.max())
    axis.set_xticks([-2, -1, 0, 1])
    axis.grid(False)
    axis.tick_params(top=False, right=False)

panel_label(ax[0], "(a)")
panel_label(ax[1], "(b)")

fig.subplots_adjust(left=0.11, right=0.99, bottom=0.22, top=0.96, wspace=0.30)
fig.savefig(OUT, bbox_inches="tight")
plt.close(fig)
