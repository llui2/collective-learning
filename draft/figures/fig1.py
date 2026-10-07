"""Arola--Lacasa effective-theory baseline."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from style import DEPTH_COLORS, LINESTYLE, MARKER, MARKERSIZE, apply_style, panel_label


apply_style()

ROOT = Path(__file__).resolve().parents[2]
data = np.load(ROOT / "results" / "theory_baseline.npz")
OUT = Path(__file__).with_suffix(".pdf")

sigmas = data["sigmas"]
depths = data["depths"]
magnetization = data["magnetization"]
loss = data["loss"]
x = np.log10(sigmas)

fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.65))

for k, depth in enumerate(depths):
    color = DEPTH_COLORS[k]
    m = magnetization[k].mean(axis=0)
    m_sem = magnetization[k].std(axis=0) / np.sqrt(magnetization.shape[1])
    l = loss[k].mean(axis=0)
    l_sem = loss[k].std(axis=0) / np.sqrt(loss.shape[1])
    l0 = l[0]

    ax[0].plot(
        x,
        m,
        color=color,
        linestyle=LINESTYLE,
        marker=MARKER,
        markersize=MARKERSIZE,
        markevery=3,
        label=fr"$D={int(depth)}$",
    )
    ax[0].fill_between(
        x,
        m - m_sem,
        m + m_sem,
        color=color,
        alpha=0.10,
        linewidth=0,
    )

    ax[1].plot(
        x,
        l / l0,
        color=color,
        linestyle=LINESTYLE,
        marker=MARKER,
        markersize=MARKERSIZE,
        markevery=3,
    )
    ax[1].fill_between(
        x,
        (l - l_sem) / l0,
        (l + l_sem) / l0,
        color=color,
        alpha=0.10,
        linewidth=0,
    )

ax[0].set_xlabel(r"$\log_{10}\sigma$")
ax[0].set_ylabel(r"$|\langle m\rangle|$")
ax[0].legend(loc="best")

ax[1].set_xlabel(r"$\log_{10}\sigma$")
ax[1].set_ylabel(r"$\langle L\rangle / \langle L\rangle_{\sigma_{\min}}$")

for axis in ax:
    axis.grid(False)
    axis.tick_params(top=False, right=False)

panel_label(ax[0], "(a)")
panel_label(ax[1], "(b)")

fig.subplots_adjust(left=0.11, right=0.99, bottom=0.22, top=0.96, wspace=0.32)
fig.savefig(OUT, bbox_inches="tight")
plt.close(fig)
