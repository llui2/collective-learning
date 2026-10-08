"""Build the two-panel MoE figure from the reduced gradient-flow model."""

from pathlib import Path
from argparse import Namespace
from collective_learning.moe import run

if __name__ == "__main__":
    args = Namespace(
        temperatures="1,3,6", seeds="0,1,2,3,4,5,6,7",
        steps=1600, record_every=20, dt=0.2, noise=0.025,
        expert_rate=0.25, router_rate=0.25,
        expert_decay=0.08, router_decay=0.08, balance=0.2,
        output="results/moe_figure.json", smoke=False,
    )
    run(args)
    Path("draft/figures/moe.pdf").write_bytes(
        Path("results/moe_figure.pdf").read_bytes()
    )
