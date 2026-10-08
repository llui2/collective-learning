"""Shared typography for the two baseline figures."""

import matplotlib as mpl

DEPTH_COLORS = ("deepskyblue", "blue", "darkblue")
DEPTH_MARKERS = ("x", "^", "o")
ADIABATIC_LINESTYLE = "-."
MARKERSIZE = 4.5


def apply_style():
    mpl.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 9,
            "axes.linewidth": 0.9,
            "lines.linewidth": 1.35,
            "legend.frameon": False,
        }
    )


def panel_label(ax, label):
    ax.text(-0.18, 1.03, label, transform=ax.transAxes, fontsize=10, va="bottom")
