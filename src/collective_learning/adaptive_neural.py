"""Sample-level adaptive neural dynamics in a common MNIST environment.

Every learner receives the same minibatch. A slow strategy g_i scores each
sample through a fixed common representation phi(x), and a softmax over the
minibatch redistributes a fixed gradient budget. Environmental components are
used only afterwards as diagnostics for coarse-grained allocation and
competence; they do not enter the learning dynamics.
"""

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets
from tqdm.auto import tqdm

from .core import NeuralUnit, coupled_sgd_step, evaluate


VERSION = 2


def resolve_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_mnist(device):
    train = datasets.MNIST("data", train=True, download=True)
    test = datasets.MNIST("data", train=False, download=True)

    x_train = train.data.float().div_(255.0).unsqueeze(1).to(device)
    y_train = train.targets.to(device)
    x_test = test.data.float().div_(255.0).unsqueeze(1).to(device)
    y_test = test.targets.to(device)
    return x_train, y_train, x_test, y_test


@torch.no_grad()
def fit_common_representation(x_train, args, device):
    gen = torch.Generator(device=device).manual_seed(args.partition_seed)
    n_fit = min(args.partition_samples, x_train.shape[0])
    ids = torch.randperm(
        x_train.shape[0], generator=gen, device=device
    )[:n_fit]

    flat = x_train[ids].flatten(start_dim=1)
    mean = flat.mean(dim=0, keepdim=True)
    centered = flat - mean

    covariance = centered.T @ centered / max(1, n_fit - 1)
    _, eigenvectors = torch.linalg.eigh(covariance)
    basis = eigenvectors[:, -args.representation_dim :]

    z = centered @ basis
    scale = z.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    z = z / scale

    first = int(
        torch.randint(
            z.shape[0], (1,), generator=gen, device=device
        ).item()
    )
    centers = [z[first].clone()]
    min_distance = ((z - centers[0]) ** 2).sum(dim=1)

    for _ in range(1, args.components):
        idx = int(min_distance.argmax().item())
        center = z[idx].clone()
        centers.append(center)
        distance = ((z - center) ** 2).sum(dim=1)
        min_distance = torch.minimum(min_distance, distance)

    centers = torch.stack(centers)

    for _ in range(args.kmeans_iterations):
        labels = torch.cdist(z, centers).argmin(dim=1)
        updated = []
        for mu in range(args.components):
            mask = labels == mu
            updated.append(
                z[mask].mean(dim=0) if mask.any() else centers[mu]
            )
        new_centers = torch.stack(updated)
        if float((new_centers - centers).square().sum().sqrt()) < 1e-5:
            centers = new_centers
            break
        centers = new_centers

    return mean, basis, scale, centers


@torch.no_grad()
def transform(x, mean, basis, scale, chunk_size=8192):
    chunks = []
    for start in range(0, x.shape[0], chunk_size):
        flat = x[start : start + chunk_size].flatten(start_dim=1)
        chunks.append(((flat - mean) @ basis) / scale)
    return torch.cat(chunks)


@torch.no_grad()
def cluster_labels(phi, centers):
    return torch.cdist(phi, centers).argmin(dim=1)


def identical_ensemble(args, device, seed):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    base = NeuralUnit(depth=args.depth, width=args.width).to(device)
    return [copy.deepcopy(base) for _ in range(args.units)]


def sample_weights(g, phi, temperature):
    scores = g @ phi.T / temperature
    scores = scores - scores.max(dim=1, keepdim=True).values
    return phi.shape[0] * torch.softmax(scores, dim=1)


@torch.no_grad()
def sample_competence(ensemble, x, y):
    rows = []
    for model in ensemble:
        model.eval()
        loss = F.cross_entropy(model(x), y, reduction="none")
        rows.append(torch.exp(-loss))
    return torch.stack(rows)


@torch.no_grad()
def sample_fitness(competence):
    units = competence.shape[0]
    mean = competence.mean(dim=0, keepdim=True)
    mean_minus = (units * mean - competence) / (units - 1)
    return (
        0.5 * (1.0 - mean_minus).square()
        - 0.5 * (1.0 - mean).square()
    )


@torch.no_grad()
def update_strategy(g, phi, fitness, args):
    # The component rule dot g_mu = epsilon T F_mu is extended to a
    # continuous common feature space by projecting the sample-wise marginal
    # contribution field onto phi. Subtracting the sample mean is a softmax
    # gauge choice and removes feature-independent fitness offsets.
    centered = fitness - fitness.mean(dim=1, keepdim=True)
    drive = centered @ phi / phi.shape[0]

    g = g + args.epsilon * args.temperature * drive
    g = g - args.epsilon * args.exploration * g
    return g


@torch.no_grad()
def routing_allocation(g, phi, cluster, components, temperature):
    p = torch.softmax(g @ phi.T / temperature, dim=1)
    a = torch.zeros(
        (g.shape[0], components), device=g.device, dtype=p.dtype
    )
    a.scatter_add_(
        1,
        cluster.unsqueeze(0).expand(g.shape[0], -1),
        p,
    )
    return a


@torch.no_grad()
def component_competence(competence, cluster, components):
    out = torch.zeros(
        (competence.shape[0], components),
        device=competence.device,
        dtype=competence.dtype,
    )
    counts = torch.bincount(
        cluster, minlength=components
    ).to(competence.dtype).clamp_min(1.0)

    for i in range(competence.shape[0]):
        out[i].scatter_add_(0, cluster, competence[i])
    return out / counts.unsqueeze(0)


def differentiation(x):
    mean = x.mean(dim=0, keepdim=True)
    return float(((x - mean).square().sum(dim=1)).mean().item())


def run_sigma(
    sigma,
    args,
    x_train,
    y_train,
    x_test,
    y_test,
    phi_train,
    phi_test,
    cluster_test,
    device,
):
    ensemble = identical_ensemble(args, device, args.seed)

    gen = torch.Generator(device=device).manual_seed(args.seed + 100003)
    g = args.strategy_noise * torch.randn(
        (args.units, args.representation_dim),
        generator=gen,
        device=device,
    )

    batch_gen = torch.Generator(device=device).manual_seed(args.seed + 200003)
    probe_gen = torch.Generator(device=device).manual_seed(args.seed + 300003)

    adjacency = torch.ones((args.units, args.units), device=device)
    adjacency -= torch.eye(args.units, device=device)

    history_step = []
    history_g = []
    history_s = []
    history_norm = []
    budget_error = 0.0

    progress = tqdm(
        range(args.steps),
        desc=f"adaptive NN sigma={sigma:.3g}",
        unit="step",
        dynamic_ncols=True,
        leave=False,
    )

    for step in progress:
        ids = torch.randint(
            x_train.shape[0],
            (args.batch_size,),
            generator=batch_gen,
            device=device,
        )
        xb = x_train[ids]
        yb = y_train[ids]
        phib = phi_train[ids]

        weights = sample_weights(g, phib, args.temperature)
        budget_error = max(
            budget_error,
            float((weights.mean(dim=1) - 1.0).abs().max().item()),
        )

        coupled_sgd_step(
            ensemble,
            [(xb, yb)] * args.units,
            learning_rate=args.learning_rate,
            coupling=sigma,
            weight_decay=args.weight_decay,
            adjacency=adjacency,
            sample_weights=[weights[i] for i in range(args.units)],
        )

        if (step + 1) % args.strategy_every == 0:
            probe_ids = torch.randint(
                x_train.shape[0],
                (args.strategy_probe_size,),
                generator=probe_gen,
                device=device,
            )
            xp = x_train[probe_ids]
            yp = y_train[probe_ids]
            phip = phi_train[probe_ids]

            competence = sample_competence(ensemble, xp, yp)
            fitness = sample_fitness(competence)
            g = update_strategy(g, phip, fitness, args)

            cluster_probe = cluster_labels(phip, args.centers)
            a_probe = routing_allocation(
                g,
                phip,
                cluster_probe,
                args.components,
                args.temperature,
            )
            m_probe = component_competence(
                competence,
                cluster_probe,
                args.components,
            )

            history_step.append(step + 1)
            history_g.append(differentiation(a_probe))
            history_s.append(differentiation(m_probe))
            history_norm.append(float(g.norm(dim=1).mean().item()))

            progress.set_postfix(
                G=f"{history_g[-1]:.3g}",
                S=f"{history_s[-1]:.3g}",
                refresh=False,
            )

    with torch.no_grad():
        test_weights = torch.softmax(
            g @ phi_test.T / args.temperature,
            dim=1,
        )
        a_test = torch.zeros(
            (args.units, args.components),
            device=device,
            dtype=test_weights.dtype,
        )
        a_test.scatter_add_(
            1,
            cluster_test.unsqueeze(0).expand(args.units, -1),
            test_weights,
        )

    test_competence_sample = sample_competence(
        ensemble, x_test, y_test
    )
    test_competence = component_competence(
        test_competence_sample,
        cluster_test,
        args.components,
    )

    metrics = [evaluate(model, x_test, y_test) for model in ensemble]
    accuracy = float(np.mean([metric[0] for metric in metrics]))
    loss = float(np.mean([metric[1] for metric in metrics]))

    tail = max(1, int(args.tail_fraction * len(history_g)))

    return {
        "sigma": float(sigma),
        "G": differentiation(a_test),
        "G_tail": float(np.mean(history_g[-tail:])),
        "S": differentiation(test_competence),
        "S_probe_tail": float(np.mean(history_s[-tail:])),
        "accuracy": accuracy,
        "loss": loss,
        "budget_error": budget_error,
        "strategy_norm": float(g.norm(dim=1).mean().item()),
        "allocation": a_test.detach().cpu().tolist(),
        "competence": test_competence.detach().cpu().tolist(),
        "history": {
            "step": history_step,
            "G": history_g,
            "S": history_s,
            "strategy_norm": history_norm,
        },
    }


def run_sweep(args, x_train, y_train, x_test, y_test, phi_train, phi_test, cluster_test, device):
    if args.smoke:
        sigmas = np.asarray([0.03, 1.0, 30.0])
        args.steps = min(args.steps, 120)
        args.strategy_every = min(args.strategy_every, 20)
        args.strategy_probe_size = min(args.strategy_probe_size, 128)
    elif args.quick:
        sigmas = np.logspace(-2.0, 2.0, 9)
        args.steps = min(args.steps, 6000)
    else:
        sigmas = np.logspace(-2.0, 2.0, args.sigma_points)

    rows = []
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config.pop("centers", None)
    config.update(
        {
            "version": VERSION,
            "sample_level_routing": True,
            "clusters_enter_dynamics": False,
            "shared_minibatch": True,
            "identical_initial_networks": True,
        }
    )

    for k, sigma in enumerate(sigmas):
        print(
            f"\n== adaptive neural sigma={sigma:.6g} "
            f"({k + 1}/{len(sigmas)}) ==",
            flush=True,
        )
        row = run_sigma(
            float(sigma),
            args,
            x_train,
            y_train,
            x_test,
            y_test,
            phi_train,
            phi_test,
            cluster_test,
            device,
        )
        rows.append(row)
        out.write_text(
            json.dumps(
                {"config": config, "complete": False, "results": rows},
                indent=2,
            )
        )

    out.write_text(
        json.dumps(
            {"config": config, "complete": True, "results": rows},
            indent=2,
        )
    )
    print(f"wrote {out}", flush=True)


def run_pilot(args, x_train, y_train, x_test, y_test, phi_train, phi_test, cluster_test, device):
    temperatures = (0.5, 1.0, 2.0)
    epsilons = (5.0, 20.0, 80.0)
    original_temperature = args.temperature
    original_epsilon = args.epsilon
    original_steps = args.steps

    args.steps = min(args.steps, args.pilot_steps)
    rows = []

    for temperature in temperatures:
        for epsilon in epsilons:
            args.temperature = temperature
            args.epsilon = epsilon
            print(
                f"\n== pilot T={temperature:g}, epsilon={epsilon:g} ==",
                flush=True,
            )
            row = run_sigma(
                args.pilot_sigma,
                args,
                x_train,
                y_train,
                x_test,
                y_test,
                phi_train,
                phi_test,
                cluster_test,
                device,
            )
            history = row["history"]["G"]
            initial = max(history[0], 1e-12)
            row["temperature"] = temperature
            row["epsilon"] = epsilon
            row["G_growth"] = float(np.mean(history[-max(1, len(history)//5):]) / initial)
            rows.append(row)
            print(
                f"G={row['G']:.3e} growth={row['G_growth']:.3g} "
                f"S={row['S']:.3e}",
                flush=True,
            )

    args.temperature = original_temperature
    args.epsilon = original_epsilon
    args.steps = original_steps

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.pop("centers", None)
    config.update(
        {
            "version": VERSION,
            "pilot": True,
            "sample_level_routing": True,
            "clusters_enter_dynamics": False,
        }
    )
    out.write_text(
        json.dumps(
            {"config": config, "complete": True, "results": rows},
            indent=2,
        )
    )
    print(f"wrote {out}", flush=True)


def run(args):
    device = resolve_device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    x_train, y_train, x_test, y_test = load_mnist(device)

    print("building fixed common representation", flush=True)
    mean, basis, scale, centers = fit_common_representation(
        x_train, args, device
    )
    phi_train = transform(x_train, mean, basis, scale)
    phi_test = transform(x_test, mean, basis, scale)
    cluster_test = cluster_labels(phi_test, centers)
    args.centers = centers

    if args.pilot:
        run_pilot(
            args,
            x_train,
            y_train,
            x_test,
            y_test,
            phi_train,
            phi_test,
            cluster_test,
            device,
        )
    else:
        run_sweep(
            args,
            x_train,
            y_train,
            x_test,
            y_test,
            phi_train,
            phi_test,
            cluster_test,
            device,
        )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--units", type=int, default=10)
    p.add_argument("--components", type=int, default=3)
    p.add_argument("--depth", type=int, default=1)
    p.add_argument("--width", type=int, default=20)
    p.add_argument("--steps", type=int, default=15000)
    p.add_argument("--sigma-points", type=int, default=17)
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--strategy-every", type=int, default=20)
    p.add_argument("--strategy-probe-size", type=int, default=512)
    p.add_argument("--epsilon", type=float, default=20.0)
    p.add_argument("--exploration", type=float, default=0.002)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--strategy-noise", type=float, default=0.02)
    p.add_argument("--tail-fraction", type=float, default=0.2)
    p.add_argument("--learning-rate", type=float, default=0.005)
    p.add_argument("--weight-decay", type=float, default=0.001)
    p.add_argument("--representation-dim", type=int, default=8)
    p.add_argument("--partition-samples", type=int, default=6000)
    p.add_argument("--partition-seed", type=int, default=17)
    p.add_argument("--kmeans-iterations", type=int, default=30)
    p.add_argument("--pilot-sigma", type=float, default=0.03)
    p.add_argument("--pilot-steps", type=int, default=2500)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--output",
        default="results/adaptive_neural_seed0.json",
    )
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
