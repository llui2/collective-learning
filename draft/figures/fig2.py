"""Arola--Lacasa MNIST baseline aggregated over available seeds."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from style import DEPTH_COLORS, LINESTYLE, MARKER, MARKERSIZE, apply_style, panel_label


apply_style()

ROOT = Path(__file__).resolve().parents[2]
def completed(paths):
    keep = []
    for path in paths:
        data = json.loads(path.read_text())
        if data.get("complete") is True:
            keep.append((path, data))
    return keep


full = completed(sorted((ROOT / "results").glob("mnist_seed*.json")))
quick = completed(sorted((ROOT / "results").glob("mnist_quick_seed*.json")))
files = full if full else quick
OUT = Path(__file__).with_suffix(".pdf")

rows = []
for _, data in files:
    rows.extend(data["results"])

depths = sorted({row["depth"] for row in rows})
sigmas = sorted({row["sigma_released_plot"] for row in rows})

fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.65))

for k, depth in enumerate(depths):
    color = DEPTH_COLORS[k]
    m_mean, m_sem, l_mean, l_sem = [], [], [], []

    for sigma in sigmas:
        subset = [
            row
            for row in rows
            if row["depth"] == depth
            and np.isclose(row["sigma_released_plot"], sigma)
        ]
        m = np.array([row["magnetization"] for row in subset])
        l = np.array([row["loss"] for row in subset])

        m_mean.append(m.mean())
        m_sem.append(m.std() / np.sqrt(len(m)))
        l_mean.append(l.mean())
        l_sem.append(l.std() / np.sqrt(len(l)))

    m_mean = np.array(m_mean)
    m_sem = np.array(m_sem)
    l_mean = np.array(l_mean)
    l_sem = np.array(l_sem)
    x = np.log10(sigmas)

    marker = ("x", "^", "o")[k]
    ax[0].plot(
        x,
        m_mean,
        color=color,
        linestyle="",
        marker=marker,
        markersize=MARKERSIZE,
        label=fr"$D={depth}$",
    )
    ax[0].fill_between(
        x,
        m_mean - m_sem,
        m_mean + m_sem,
        color=color,
        alpha=0.10,
        linewidth=0,
    )

    ax[1].plot(
        x,
        l_mean / l_mean[0],
        color=color,
        linestyle="",
        marker=marker,
        markersize=MARKERSIZE,
    )
    ax[1].fill_between(
        x,
        (l_mean - l_sem) / l_mean[0],
        (l_mean + l_sem) / l_mean[0],
        color=color,
        alpha=0.10,
        linewidth=0,
    )

ax[0].set_xlabel(r"$\log_{10}\sigma$")
ax[0].set_ylabel(r"$\langle m\rangle$")
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
