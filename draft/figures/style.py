"""Shared figure style derived from the Arola--Lacasa plotting scripts."""

import matplotlib as mpl

DEPTH_COLORS = ("lightblue", "deepskyblue", "blue", "darkblue")
LINESTYLE = "-."
MARKER = "."
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
