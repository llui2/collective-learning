"""Generate the main performance comparison from matched MoE runs."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "draft" / "figures" / "fig1.pdf"
CSV_OUT = RESULTS / "comparison.csv"


def load_history(path):
    data = json.loads(path.read_text())
    history = data["history"]
    return {row["step"]: row for row in history}


baseline = load_history(RESULTS / "lambda0.json")
coupled = load_history(RESULTS / "lambda1.json")

steps = sorted(set(baseline) & set(coupled))
if not steps:
    raise RuntimeError("No matched evaluation steps found in the two runs.")

loss0 = [baseline[step]["loss"] for step in steps]
loss1 = [coupled[step]["loss"] for step in steps]
delta = [b - a for a, b in zip(loss0, loss1)]

RESULTS.mkdir(parents=True, exist_ok=True)
with CSV_OUT.open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["step", "baseline_loss", "coupled_loss", "delta_loss"])
    writer.writerows(zip(steps, loss0, loss1, delta))

OUT.parent.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7))

ax = axes[0]
ax.plot(steps, loss0, lw=1.6, label=r"$\lambda = 0$")
ax.plot(steps, loss1, lw=1.6, label=r"$\lambda = 1$")
ax.set_xlabel("training step")
ax.set_ylabel("validation loss")
ax.legend(frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.text(-0.17, 1.04, "(a)", transform=ax.transAxes, fontsize=11)

ax = axes[1]
ax.axhline(0.0, lw=0.8, ls="--")
ax.plot(steps, delta, lw=1.6)
ax.set_xlabel("training step")
ax.set_ylabel(r"$\mathcal{L}_{\lambda=1}-\mathcal{L}_{\lambda=0}$")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.text(-0.17, 1.04, "(b)", transform=ax.transAxes, fontsize=11)

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
plt.close(fig)

print(f"wrote {OUT.relative_to(ROOT)}")
print(f"wrote {CSV_OUT.relative_to(ROOT)}")
