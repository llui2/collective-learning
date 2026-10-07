"""Arola--Lacasa MNIST baseline aggregated over available seeds."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FILES = sorted((ROOT / "results").glob("mnist_seed*.json"))
OUT = Path(__file__).with_suffix(".pdf")

rows = []
for path in FILES:
    rows.extend(json.loads(path.read_text())["results"])

depths = sorted({row["depth"] for row in rows})
sigmas = sorted({row["sigma_released_plot"] for row in rows})

fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
for depth in depths:
    m_mean, m_sem, l_mean, l_sem = [], [], [], []
    for sigma in sigmas:
        subset = [r for r in rows if r["depth"] == depth and np.isclose(r["sigma_released_plot"], sigma)]
        m = np.array([r["magnetization"] for r in subset])
        l = np.array([r["loss"] for r in subset])
        m_mean.append(m.mean()); m_sem.append(m.std()/np.sqrt(len(m)))
        l_mean.append(l.mean()); l_sem.append(l.std()/np.sqrt(len(l)))

    m_mean = np.array(m_mean); m_sem = np.array(m_sem)
    l_mean = np.array(l_mean); l_sem = np.array(l_sem)
    x = np.log10(sigmas)
    ax[0].plot(x, m_mean, label=fr"$D={depth}$")
    ax[0].fill_between(x, m_mean-m_sem, m_mean+m_sem, alpha=0.18)
    ax[1].plot(x, l_mean/l_mean[0])
    ax[1].fill_between(x, (l_mean-l_sem)/l_mean[0], (l_mean+l_sem)/l_mean[0], alpha=0.18)

ax[0].set_xlabel(r"$\log_{10}\sigma$")
ax[0].set_ylabel(r"$\langle m\rangle$")
ax[0].legend(frameon=False)
ax[1].set_xlabel(r"$\log_{10}\sigma$")
ax[1].set_ylabel(r"$\langle L\rangle/\langle L\rangle_{\sigma_{\min}}$")
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
