"""A minimal trainable MoE feed-forward layer with token-level top-k routing.

Not a full transformer: token representations are fixed and the
reference implementation evaluates all experts before masking them.
For top-1, the selected expert retains its softmax gate (Switch-style).
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class TokenMoE(nn.Module):
    def __init__(self, width=8, temperature=1.0, topk=1, capacity_factor=1.25):
        super().__init__()
        if topk not in (1, 2) or temperature <= 0 or capacity_factor <= 0:
            raise ValueError("Need topk in {1,2}, positive temperature/capacity")
        self.router = nn.Linear(2, 2)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(2, width), nn.ReLU(), nn.Linear(width, 1))
            for _ in range(2)
        ])
        self.topk = topk
        self.temperature = temperature
        self.capacity_factor = capacity_factor

    def forward(self, tokens):
        probabilities = F.softmax(self.router(tokens) / self.temperature, dim=-1)
        if self.topk == 2:
            assignment = torch.ones_like(probabilities)
        else:
            indices = probabilities.argmax(dim=-1)
            assignment = F.one_hot(indices, num_classes=2).to(probabilities.dtype)
            capacity = math.ceil(self.capacity_factor * len(tokens) / 2)
            for i in range(2):
                locations = torch.nonzero(indices == i, as_tuple=True)[0]
                assignment[locations[capacity:], i] = 0.0
        values = torch.stack([expert(tokens) for expert in self.experts], dim=1)
        output = ((probabilities * assignment).unsqueeze(-1) * values).sum(dim=1)
        return output, probabilities, assignment


def tokens(batch, generator, device):
    """Balanced classes c=+/-1, feature z~N(0,1), target c*z."""
    if batch % 2:
        raise ValueError("Batch size must be even")
    c = torch.cat([torch.ones(batch // 2), -torch.ones(batch // 2)])
    z = torch.randn(batch, generator=generator)
    index = torch.randperm(batch, generator=generator)
    x = torch.stack((c, z), dim=-1)[index].to(device)
    return x, (x[:, :1] * x[:, 1:2])


@torch.no_grad()
def metrics(model, x, y):
    model.eval()
    pred, p, assignment = model(x)
    mask_plus = x[:, 0] > 0
    contrast = (p[mask_plus, 0].mean() - p[~mask_plus, 0].mean()).item()
    load_bias = (2 * p[:, 0].mean() - 1).item()
    hard_bias = (assignment[:, 0].mean() - assignment[:, 1].mean()).item()
    loss = F.mse_loss(pred, y).item() / 2
    individual = torch.tensor([
        [F.mse_loss(expert(x[~mask_plus]), y[~mask_plus]).item() / 2,
         F.mse_loss(expert(x[mask_plus]), y[mask_plus]).item() / 2]
        for expert in model.experts
    ])
    # Positive when the routed experts are actually better on their
    # preferred token class, rather than just assigned different tokens.
    functional_alignment = contrast * (
        individual[0, 0] - individual[1, 0]
        + individual[1, 1] - individual[0, 1]
    ).item()
    return {
        "loss": loss,
        "routing_contrast": contrast,
        "soft_load_bias": load_bias,
        "hard_load_bias": hard_bias,
        "dropped_fraction": float(1 - assignment.sum(dim=-1).mean().item())
        if model.topk == 1 else 0.0,
        "functional_alignment": functional_alignment,
        "expert_cross_loss": individual.tolist(),
    }


def trial(args, topk, balance, seed):
    torch.manual_seed(seed)
    model = TokenMoE(
        width=args.width, temperature=args.temperature,
        topk=topk, capacity_factor=args.capacity,
    ).to(args.device)
    optimizer = torch.optim.SGD([
        {"params": model.experts.parameters(), "lr": args.expert_rate},
        {"params": model.router.parameters(), "lr": args.router_rate},
    ])
    rng = torch.Generator().manual_seed(seed + 100)
    test_x, test_y = tokens(
        args.evaluation, torch.Generator().manual_seed(seed + 200),
        args.device
    )
    history = []
    for step in range(args.steps + 1):
        if step % args.record_every == 0 or step == args.steps:
            history.append({"step": step, **metrics(model, test_x, test_y)})
        if step == args.steps:
            break
        model.train()
        x, y = tokens(args.batch, rng, args.device)
        prediction, p, _ = model(x)
        loss = 0.5 * F.mse_loss(prediction, y)
        loss = loss + balance * (p.mean(dim=0) - 0.5).square().sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return {"topk": topk, "balance": balance, "seed": seed, "history": history}


def plot(results, args, output):
    """Test loss and functional-alignment dynamics for each routing mode."""
    fig, axes = plt.subplots(2, 1, figsize=(5.5, 5.5), sharex=True)
    for topk in (1, 2):
        for balance in sorted(set(t["balance"] for t in results)):
            trials = [t for t in results if t["topk"] == topk
                      and t["balance"] == balance]
            if not trials:
                continue
            steps = [h["step"] for h in trials[0]["history"]]
            for ax, key in zip(axes, ("loss", "functional_alignment")):
                values = np.array([
                    [h[key] for h in t["history"]] for t in trials
                ])
                mean, sd = values.mean(axis=0), values.std(axis=0)
                line, = ax.plot(
                    steps, mean, marker="o", markevery=max(1,len(steps)//10),
                    markersize=3, label=rf"$k={topk}$, $\beta={balance:g}$"
                )
                ax.fill_between(
                    steps, mean - sd, mean + sd,
                    color=line.get_color(), alpha=0.13,
                )
    axes[0].set_ylabel("Individual MoE test loss")
    axes[1].set_ylabel("Routing--skill alignment")
    axes[1].set_xlabel("SGD updates")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def run(args):
    if args.smoke:
        args.topks, args.balances, args.seeds = "1,2", "0.2", "0"
        args.steps, args.batch, args.evaluation = 6, 16, 32
        args.record_every = 2
        if args.output == "results/moe_layer.json":
            args.output = "results/moe_layer_smoke.json"
    topks = [int(v) for v in args.topks.split(",")]
    balances = [float(v) for v in args.balances.split(",")]
    seeds = [int(v) for v in args.seeds.split(",")]
    if (not topks or not balances or not seeds or
            any(k not in (1, 2) for k in topks) or
            any(not np.isfinite(b) or b < 0 for b in balances) or
            args.width < 1 or args.steps < 1 or args.record_every < 1 or
            args.batch < 2 or args.batch % 2 or
            args.evaluation < 2 or args.evaluation % 2 or
            args.temperature <= 0 or args.capacity <= 0 or
            args.expert_rate <= 0 or args.router_rate <= 0):
        raise ValueError("Invalid token-level MoE configuration")
    args.device = torch.device(args.device)
    if args.device.type == "cpu":
        torch.set_num_threads(1)
    experiments = [
        trial(args, k, beta, seed)
        for k in topks for beta in balances for seed in seeds
    ]
    data = {
        "config": dict(vars(args), device=str(args.device)),
        "trials": experiments,
        "note": "Top-1 uses softmax-selected gate and optional token-capacity drop; "
                "top-2 is a dense differentiable reference, not sparse dispatch",
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2))
    plot(experiments, args, out.with_suffix(".pdf"))
    return data


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--topks", default="1,2")
    p.add_argument("--balances", default="0,0.2")
    p.add_argument("--seeds", default="0,1,2,3")
    p.add_argument("--width", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--capacity", type=float, default=1.25)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--record-every", type=int, default=20)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--evaluation", type=int, default=512)
    p.add_argument("--expert-rate", type=float, default=0.05)
    p.add_argument("--router-rate", type=float, default=0.05)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", default="results/moe_layer.json")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
