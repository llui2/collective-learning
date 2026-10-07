"""Arola--Lacasa effective-theory baseline."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
data = np.load(ROOT / "results" / "theory_baseline.npz")
OUT = Path(__file__).with_suffix(".pdf")

sigmas = data["sigmas"]
depths = data["depths"]
magnetization = data["magnetization"]
loss = data["loss"]
x = np.log10(sigmas)

fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
for k, depth in enumerate(depths):
    m = magnetization[k].mean(axis=0)
    m_sem = magnetization[k].std(axis=0) / np.sqrt(magnetization.shape[1])
    l = loss[k].mean(axis=0)
    l_sem = loss[k].std(axis=0) / np.sqrt(loss.shape[1])
    l0 = l[0]
    ax[0].plot(x, m, label=fr"$D={int(depth)}$")
    ax[0].fill_between(x, m - m_sem, m + m_sem, alpha=0.18)
    ax[1].plot(x, l / l0)
    ax[1].fill_between(x, (l-l_sem)/l0, (l+l_sem)/l0, alpha=0.18)

ax[0].set_xlabel(r"$\log_{10}\sigma$")
ax[0].set_ylabel(r"$|\langle m\rangle|$")
ax[0].legend(frameon=False)
ax[1].set_xlabel(r"$\log_{10}\sigma$")
ax[1].set_ylabel(r"$\langle L\rangle/\langle L\rangle_{\sigma_{\min}}$")
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
