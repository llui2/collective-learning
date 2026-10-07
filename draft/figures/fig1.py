"""Arola--Lacasa effective-theory baseline."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from style import (
    ADIABATIC_LINESTYLE,
    DEPTH_COLORS,
    DEPTH_MARKERS,
    MARKERSIZE,
    apply_style,
    panel_label,
)


apply_style()

ROOT = Path(__file__).resolve().parents[2]
data = np.load(ROOT / "results" / "theory_baseline.npz")
OUT = Path(__file__).with_suffix(".pdf")

sigmas = data["sigmas"]
depths = data["depths"]
magnetization = data["magnetization"]
loss = data["loss"]
magnetization_ad = data["magnetization_adiabatic"]
loss_ad = data["loss_adiabatic"]
x = np.log10(sigmas)

fig, ax = plt.subplots(1, 2, figsize=(7.3, 2.75))

for k, depth in enumerate(depths):
    color = DEPTH_COLORS[k]
    marker = DEPTH_MARKERS[k]

    m = magnetization[k].mean(axis=0)
    m_std = magnetization[k].std(axis=0)
    m_ad = magnetization_ad[k].mean(axis=0)

    l = loss[k].mean(axis=0)
    l_std = loss[k].std(axis=0)
    l_ad = loss_ad[k].mean(axis=0)

    l0 = l[0]
    l_ad0 = l_ad[0]

    ax[0].plot(
        x,
        m,
        color=color,
        marker=marker,
        linestyle="",
        markersize=MARKERSIZE,
        label=fr"$D={int(depth)}$",
    )
    ax[0].fill_between(
        x,
        m - m_std,
        m + m_std,
        color=color,
        alpha=0.22,
        linewidth=0,
    )
    ax[0].plot(
        x,
        m_ad,
        color=color,
        linestyle=ADIABATIC_LINESTYLE,
        marker=None,
        linewidth=1.1,
    )

    l_norm = l / l0
    l_std_norm = l_std / l0
    l_ad_norm = l_ad / l_ad0

    ax[1].plot(
        x,
        l_norm,
        color=color,
        marker=marker,
        linestyle="",
        markersize=MARKERSIZE,
    )
    ax[1].fill_between(
        x,
        l_norm - l_std_norm,
        l_norm + l_std_norm,
        color=color,
        alpha=0.22,
        linewidth=0,
    )
    ax[1].plot(
        x,
        l_ad_norm,
        color=color,
        linestyle=ADIABATIC_LINESTYLE,
        marker=None,
        linewidth=1.1,
    )

ax[0].set_xlabel(r"$\log_{10}\sigma$")
ax[0].set_ylabel(r"$|\langle m\rangle|$")
ax[0].legend(loc="best")

ax[1].set_xlabel(r"$\log_{10}\sigma$")
ax[1].set_ylabel(r"$\langle L\rangle / \langle L\rangle_{\sigma_{\min}}$")

axins = ax[1].inset_axes([0.68, 0.64, 0.27, 0.28])
for k, depth in enumerate(depths):
    color = DEPTH_COLORS[k]
    marker = DEPTH_MARKERS[k]
    l = loss[k].mean(axis=0)
    l_std = loss[k].std(axis=0)
    l_norm = l / l[0]
    l_std_norm = l_std / l[0]

    axins.plot(
        x,
        1.0 - l_norm,
        color=color,
        marker=marker,
        linestyle="",
        markersize=3.0,
    )
    axins.fill_between(
        x,
        1.0 - l_norm - l_std_norm,
        1.0 - l_norm + l_std_norm,
        color=color,
        alpha=0.22,
        linewidth=0,
    )

axins.set_xlim(-1.5, 1.5)
axins.set_xlabel(r"$\log_{10}\sigma$", fontsize=7)
axins.set_ylabel(r"$1-\langle \hat L\rangle$", fontsize=7)
axins.tick_params(labelsize=6, top=False, right=False)

for axis in ax:
    axis.set_xlim(x.min(), x.max())
    axis.grid(False)
    axis.tick_params(top=False, right=False)

panel_label(ax[0], "(a)")
panel_label(ax[1], "(b)")

fig.subplots_adjust(left=0.11, right=0.99, bottom=0.22, top=0.96, wspace=0.32)
fig.savefig(OUT, bbox_inches="tight")
plt.close(fig)
