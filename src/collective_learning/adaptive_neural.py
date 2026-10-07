"""Adaptive neural dynamics in a common MNIST environment.

Every learner receives the same minibatch. A continuous adaptive strategy g_i
scores individual samples in a fixed shared representation phi(x),

    w_i(x_b) = B softmax_b[g_i . phi(x_b) / T],

so every learner has exactly the same total gradient budget. The strategy is
updated from sample-level marginal collective contribution using a natural
gradient in the softmax family. No coarse environmental partition enters the
learning or strategy dynamics.

A PCA representation and an unsupervised k-means partition are fixed before
training. PCA features enter the sample-level strategy. The k-means partition
is used only to measure coarse allocation and phenotype differentiation.
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
def fit_shared_representation(x_train, x_test, args, device):
    """Fit phi(x) without labels and return standardized PCA coordinates."""
    gen = torch.Generator(device=device).manual_seed(args.representation_seed)
    n_fit = min(args.representation_samples, x_train.shape[0])
    ids = torch.randperm(
        x_train.shape[0], generator=gen, device=device
    )[:n_fit]

    flat = x_train[ids].flatten(start_dim=1)
    mean = flat.mean(dim=0, keepdim=True)
    centered = flat - mean

    covariance = centered.T @ centered / max(1, n_fit - 1)
    _, eigenvectors = torch.linalg.eigh(covariance)
    basis = eigenvectors[:, -args.representation_dim :]

    fit_phi = centered @ basis
    scale = fit_phi.std(
        dim=0, unbiased=False, keepdim=True
    ).clamp_min(1e-6)

    def transform(x):
        chunks = []
        for start in range(0, x.shape[0], 8192):
            flat_chunk = x[start : start + 8192].flatten(start_dim=1)
            chunks.append(((flat_chunk - mean) @ basis) / scale)
        return torch.cat(chunks, dim=0)

    return transform(x_train), transform(x_test)


@torch.no_grad()
def fit_observation_partition(phi_train, phi_test, args, device):
    """Fit an unsupervised partition used only for coarse observables."""
    gen = torch.Generator(device=device).manual_seed(args.partition_seed)
    n_fit = min(args.partition_samples, phi_train.shape[0])
    ids = torch.randperm(
        phi_train.shape[0], generator=gen, device=device
    )[:n_fit]
    z = phi_train[ids]

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
        shift = (new_centers - centers).square().sum().sqrt()
        centers = new_centers
        if float(shift) < 1e-5:
            break

    train_cluster = torch.cdist(phi_train, centers).argmin(dim=1)
    test_cluster = torch.cdist(phi_test, centers).argmin(dim=1)
    return train_cluster, test_cluster


def cluster_summary(cluster, y, components):
    sizes = []
    class_distribution = []
    for mu in range(components):
        mask = cluster == mu
        count = int(mask.sum().item())
        sizes.append(count)
        counts = torch.bincount(y[mask], minlength=10).float()
        if count:
            counts /= count
        class_distribution.append(counts.cpu().tolist())
    return sizes, class_distribution


def identical_ensemble(args, device, seed):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    base = NeuralUnit(depth=args.depth, width=args.width).to(device)
    return [copy.deepcopy(base) for _ in range(args.units)]


@torch.no_grad()
def sample_competence(ensemble, x, y, batch_size=2048):
    """Return m_i(x)=exp[-ell_i(x)] for every learner and sample."""
    rows = []
    for model in ensemble:
        model.eval()
        parts = []
        for start in range(0, y.numel(), batch_size):
            xb = x[start : start + batch_size]
            yb = y[start : start + batch_size]
            loss = F.cross_entropy(model(xb), yb, reduction="none")
            parts.append(torch.exp(-loss))
        rows.append(torch.cat(parts))
    return torch.stack(rows)


def sample_fitness(competence):
    """Marginal collective contribution at the individual-sample level."""
    units = competence.shape[0]
    mean = competence.mean(dim=0, keepdim=True)
    mean_minus = (units * mean - competence) / (units - 1)
    return (
        0.5 * (1.0 - mean_minus).square()
        - 0.5 * (1.0 - mean).square()
    )


@torch.no_grad()
def strategy_weights(g, phi, temperature):
    scores = g @ phi.T / temperature
    return scores.shape[1] * torch.softmax(scores, dim=1)


@torch.no_grad()
def routing_probabilities(g, phi, temperature):
    return torch.softmax(g @ phi.T / temperature, dim=1)


@torch.no_grad()
def natural_strategy_step(
    g,
    phi,
    fitness,
    temperature,
    rate,
    exploration,
    ridge,
    max_step,
):
    """Natural-gradient adaptation of the sample-level softmax strategy."""
    probabilities = routing_probabilities(g, phi, temperature)
    dim = phi.shape[1]
    eye = torch.eye(dim, device=phi.device, dtype=phi.dtype)

    updated = g.clone()
    max_observed_step = 0.0

    for i in range(g.shape[0]):
        p = probabilities[i]
        mean_phi = p @ phi
        centered = phi - mean_phi

        f = fitness[i]
        mean_f = p @ f
        centered_f = f - mean_f

        covariance = centered.T @ (p[:, None] * centered)
        covariance = covariance + ridge * eye
        cross = (
            p[:, None]
            * centered
            * centered_f[:, None]
        ).sum(dim=0)

        direction = torch.linalg.solve(covariance, cross)
        delta = rate * (
            temperature * direction
            - exploration * g[i]
        )

        norm = float(torch.linalg.vector_norm(delta).item())
        if norm > max_step:
            delta *= max_step / norm
            norm = max_step

        max_observed_step = max(max_observed_step, norm)
        updated[i] += delta

    return updated, max_observed_step


def differentiation(x):
    mean = x.mean(dim=0, keepdim=True)
    return float(
        ((x - mean).square().sum(dim=1)).mean().item()
    )


@torch.no_grad()
def allocation_from_routing(
    g,
    phi,
    cluster,
    components,
    temperature,
):
    probabilities = routing_probabilities(g, phi, temperature)
    membership = F.one_hot(
        cluster, num_classes=components
    ).to(probabilities.dtype)
    return probabilities @ membership


@torch.no_grad()
def component_competence(competence, cluster, components):
    units = competence.shape[0]
    out = torch.zeros(
        (units, components),
        device=competence.device,
        dtype=competence.dtype,
    )

    for mu in range(components):
        mask = cluster == mu
        if mask.any():
            out[:, mu] = competence[:, mask].mean(dim=1)

    return out


@torch.no_grad()
def probe_observables(
    g,
    phi,
    cluster,
    competence,
    components,
    temperature,
):
    allocation = allocation_from_routing(
        g, phi, cluster, components, temperature
    )
    component_m = component_competence(
        competence, cluster, components
    )
    probabilities = routing_probabilities(
        g, phi, temperature
    )
    entropy = -(
        probabilities
        * probabilities.clamp_min(1e-12).log()
    ).sum(dim=1)
    entropy /= np.log(probabilities.shape[1])

    routing_center = probabilities @ phi
    mean_sample_competence = competence.mean(dim=0, keepdim=True)
    sample_phenotype = float(
        (competence - mean_sample_competence).square().mean().item()
    )

    return {
        "G": differentiation(allocation),
        "R": differentiation(routing_center),
        "S": differentiation(component_m),
        "S_sample": sample_phenotype,
        "entropy": float(entropy.mean().item()),
        "allocation": allocation,
        "component_competence": component_m,
    }


def fixed_probe_indices(size, count, seed, device):
    gen = torch.Generator(device=device).manual_seed(seed)
    return torch.randperm(
        size, generator=gen, device=device
    )[:count]


def sigma_grid(args):
    if args.smoke:
        return np.asarray([0.01, 0.3, 10.0])
    if args.quick:
        return np.logspace(-3.0, 1.5, 10)
    return np.logspace(-3.0, 1.5, args.sigma_points)


def run_sigma(
    sigma,
    args,
    x_train,
    y_train,
    phi_train,
    train_cluster,
    x_test,
    y_test,
    phi_test,
    test_cluster,
    probe_ids,
    device,
):
    ensemble = identical_ensemble(args, device, args.seed)

    strategy_gen = torch.Generator(device=device).manual_seed(
        args.seed + 100003
    )
    g = args.strategy_noise * torch.randn(
        (args.units, args.representation_dim),
        generator=strategy_gen,
        device=device,
    )

    batch_gen = torch.Generator(device=device).manual_seed(
        args.seed + 200003
    )
    adjacency = (
        torch.ones((args.units, args.units), device=device)
        - torch.eye(args.units, device=device)
    )

    x_probe = x_train[probe_ids]
    y_probe = y_train[probe_ids]
    phi_probe = phi_train[probe_ids]
    cluster_probe = train_cluster[probe_ids]

    history = {
        "step": [],
        "G": [],
        "R": [],
        "S": [],
        "S_sample": [],
        "entropy": [],
        "strategy_step": [],
    }
    budget_error = 0.0
    max_strategy_step = 0.0
    last_losses = []

    initial_competence = sample_competence(
        ensemble, x_probe, y_probe
    )
    initial_obs = probe_observables(
        g,
        phi_probe,
        cluster_probe,
        initial_competence,
        args.components,
        args.temperature,
    )
    history["step"].append(0)
    history["G"].append(initial_obs["G"])
    history["R"].append(initial_obs["R"])
    history["S"].append(initial_obs["S"])
    history["S_sample"].append(initial_obs["S_sample"])
    history["entropy"].append(initial_obs["entropy"])
    history["strategy_step"].append(0.0)

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
        phi_b = phi_train[ids]

        weights = strategy_weights(
            g, phi_b, args.temperature
        )
        budget_error = max(
            budget_error,
            float(
                (weights.mean(dim=1) - 1.0)
                .abs()
                .max()
                .item()
            ),
        )

        losses = coupled_sgd_step(
            ensemble,
            [(xb, yb)] * args.units,
            learning_rate=args.learning_rate,
            coupling=sigma,
            weight_decay=args.weight_decay,
            adjacency=adjacency,
            sample_weights=[
                weights[i] for i in range(args.units)
            ],
        )
        last_losses.append(float(losses.mean().item()))
        if len(last_losses) > 100:
            last_losses.pop(0)

        if (step + 1) % args.strategy_every == 0:
            competence = sample_competence(
                ensemble, x_probe, y_probe
            )
            fitness = sample_fitness(competence)

            g, strategy_step = natural_strategy_step(
                g,
                phi_probe,
                fitness,
                temperature=args.temperature,
                rate=args.strategy_rate,
                exploration=args.exploration,
                ridge=args.fisher_ridge,
                max_step=args.max_strategy_step,
            )
            max_strategy_step = max(
                max_strategy_step, strategy_step
            )

            obs = probe_observables(
                g,
                phi_probe,
                cluster_probe,
                competence,
                args.components,
                args.temperature,
            )
            history["step"].append(step + 1)
            history["G"].append(obs["G"])
            history["R"].append(obs["R"])
            history["S"].append(obs["S"])
            history["S_sample"].append(obs["S_sample"])
            history["entropy"].append(obs["entropy"])
            history["strategy_step"].append(
                strategy_step
            )

            progress.set_postfix(
                G=f"{obs['G']:.3g}",
                R=f"{obs['R']:.3g}",
                S=f"{obs['S_sample']:.3g}",
                H=f"{obs['entropy']:.3f}",
                loss=f"{np.mean(last_losses):.3g}",
                refresh=False,
            )

    test_competence = sample_competence(
        ensemble, x_test, y_test
    )
    final_obs = probe_observables(
        g,
        phi_test,
        test_cluster,
        test_competence,
        args.components,
        args.temperature,
    )

    metrics = [
        evaluate(model, x_test, y_test)
        for model in ensemble
    ]
    accuracy = float(
        np.mean([metric[0] for metric in metrics])
    )
    loss = float(
        np.mean([metric[1] for metric in metrics])
    )

    tail_count = max(
        1,
        int(
            args.tail_fraction
            * max(1, len(history["G"]) - 1)
        ),
    )
    tail_g = history["G"][-tail_count:]
    tail_s = history["S"][-tail_count:]

    return {
        "sigma": float(sigma),
        "G_initial": float(history["G"][0]),
        "G": float(np.mean(tail_g)),
        "S": float(np.mean(tail_s)),
        "G_test": final_obs["G"],
        "R_test": final_obs["R"],
        "S_test": final_obs["S"],
        "S_sample_test": final_obs["S_sample"],
        "routing_entropy_test": final_obs["entropy"],
        "accuracy": accuracy,
        "loss": loss,
        "budget_error": budget_error,
        "max_strategy_step": max_strategy_step,
        "allocation_test": (
            final_obs["allocation"].cpu().tolist()
        ),
        "competence_test": (
            final_obs["component_competence"]
            .cpu()
            .tolist()
        ),
        "strategy": g.cpu().tolist(),
        "history": history,
    }


def run(args):
    device = resolve_device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    if args.smoke:
        args.steps = 120
        args.strategy_every = 20
        args.representation_samples = min(
            args.representation_samples, 1500
        )
        args.partition_samples = min(
            args.partition_samples, 1500
        )
        args.probe_samples = min(
            args.probe_samples, 128
        )
        args.batch_size = min(args.batch_size, 64)
    elif args.quick:
        args.steps = min(args.steps, 8000)

    x_train, y_train, x_test, y_test = load_mnist(
        device
    )

    print(
        "building fixed shared representation",
        flush=True,
    )
    phi_train, phi_test = fit_shared_representation(
        x_train, x_test, args, device
    )

    print(
        "building observation-only partition",
        flush=True,
    )
    train_cluster, test_cluster = (
        fit_observation_partition(
            phi_train, phi_test, args, device
        )
    )
    train_sizes, class_distribution = cluster_summary(
        train_cluster, y_train, args.components
    )
    test_sizes, _ = cluster_summary(
        test_cluster, y_test, args.components
    )
    print(
        "observation component sizes:",
        train_sizes,
        flush=True,
    )

    probe_ids = fixed_probe_indices(
        x_train.shape[0],
        args.probe_samples,
        args.probe_seed,
        device,
    )

    sigmas = sigma_grid(args)
    rows = []
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config.update(
        {
            "version": VERSION,
            "train_component_sizes": train_sizes,
            "test_component_sizes": test_sizes,
            "cluster_class_distribution": (
                class_distribution
            ),
            "representation_uses_labels": False,
            "partition_uses_labels": False,
            "partition_enters_dynamics": False,
            "sample_level_routing": True,
            "shared_minibatch": True,
            "identical_initial_networks": True,
            "strategy_update": (
                "natural gradient of sample-level "
                "marginal collective contribution"
            ),
        }
    )

    for sigma_index, sigma in enumerate(sigmas):
        print(
            f"\n== adaptive neural sigma={sigma:.6g} "
            f"({sigma_index + 1}/{len(sigmas)}) ==",
            flush=True,
        )
        row = run_sigma(
            float(sigma),
            args,
            x_train,
            y_train,
            phi_train,
            train_cluster,
            x_test,
            y_test,
            phi_test,
            test_cluster,
            probe_ids,
            device,
        )
        rows.append(row)

        payload = {
            "config": config,
            "complete": False,
            "results": rows,
        }
        out.write_text(json.dumps(payload, indent=2))

    payload = {
        "config": config,
        "complete": True,
        "results": rows,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--units", type=int, default=10)
    p.add_argument("--components", type=int, default=3)
    p.add_argument("--depth", type=int, default=1)
    p.add_argument("--width", type=int, default=20)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--sigma-points", type=int, default=19)
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--strategy-every", type=int, default=25)
    p.add_argument("--strategy-rate", type=float, default=2.0)
    p.add_argument("--exploration", type=float, default=0.0005)
    p.add_argument("--strategy-noise", type=float, default=0.02)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--fisher-ridge", type=float, default=1e-3)
    p.add_argument("--max-strategy-step", type=float, default=0.1)
    p.add_argument("--tail-fraction", type=float, default=0.2)
    p.add_argument("--learning-rate", type=float, default=0.005)
    p.add_argument("--weight-decay", type=float, default=0.001)
    p.add_argument("--representation-dim", type=int, default=8)
    p.add_argument("--representation-samples", type=int, default=6000)
    p.add_argument("--representation-seed", type=int, default=13)
    p.add_argument("--partition-samples", type=int, default=6000)
    p.add_argument("--partition-seed", type=int, default=17)
    p.add_argument("--kmeans-iterations", type=int, default=30)
    p.add_argument("--probe-samples", type=int, default=512)
    p.add_argument("--probe-seed", type=int, default=23)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--output",
        default="results/adaptive_neural_seed0.json",
    )
    p.add_argument("--quick", action="store_true")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
