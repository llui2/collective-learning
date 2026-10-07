"""Adaptive neural dynamics in a common MNIST environment.

All learners receive the same balanced minibatch. A slow allocation a_i
redistributes a fixed gradient budget over an unsupervised partition of a
shared representation. Neural parameters follow the coupled SGD dynamics,
while the allocation follows the same marginal-contribution rule used in the
effective theory.

The environmental components are obtained without labels: PCA of MNIST
pixels followed by k-means. Labels are used only for the supervised learning
loss.
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


VERSION = 1


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
def assign_clusters(x, mean, basis, scale, centers, chunk_size=8192):
    labels = []
    for start in range(0, x.shape[0], chunk_size):
        flat = x[start : start + chunk_size].flatten(start_dim=1)
        z = ((flat - mean) @ basis) / scale
        distance = torch.cdist(z, centers)
        labels.append(distance.argmin(dim=1))
    return torch.cat(labels)


@torch.no_grad()
def fit_common_partition(x_train, x_test, args, device):
    gen = torch.Generator(device=device).manual_seed(args.partition_seed)
    n_fit = min(args.partition_samples, x_train.shape[0])
    ids = torch.randperm(x_train.shape[0], generator=gen, device=device)[:n_fit]

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
            if mask.any():
                updated.append(z[mask].mean(dim=0))
            else:
                updated.append(centers[mu])
        new_centers = torch.stack(updated)
        shift = (new_centers - centers).square().sum().sqrt()
        centers = new_centers
        if float(shift) < 1e-5:
            break

    train_cluster = assign_clusters(
        x_train, mean, basis, scale, centers
    )
    test_cluster = assign_clusters(
        x_test, mean, basis, scale, centers
    )

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


def split_environment(cluster, components, probe_per_component, seed, device):
    gen = torch.Generator(device=device).manual_seed(seed)
    train_pools = []
    probe = []

    for mu in range(components):
        ids = torch.where(cluster == mu)[0]
        if ids.numel() <= probe_per_component:
            raise ValueError(
                f"component {mu} has only {ids.numel()} samples; "
                f"need more than {probe_per_component}"
            )
        order = torch.randperm(ids.numel(), generator=gen, device=device)
        shuffled = ids[order]
        probe.append(shuffled[:probe_per_component])
        train_pools.append(shuffled[probe_per_component:])

    return train_pools, torch.cat(probe)


def balanced_batch(pools, per_component, generator):
    ids = []
    cluster = []
    device = pools[0].device

    for mu, pool in enumerate(pools):
        picks = torch.randint(
            pool.numel(),
            (per_component,),
            generator=generator,
            device=device,
        )
        ids.append(pool[picks])
        cluster.append(
            torch.full(
                (per_component,),
                mu,
                dtype=torch.long,
                device=device,
            )
        )

    ids = torch.cat(ids)
    cluster = torch.cat(cluster)
    order = torch.randperm(ids.numel(), generator=generator, device=device)
    return ids[order], cluster[order]


def identical_ensemble(args, device, seed):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    base = NeuralUnit(depth=args.depth, width=args.width).to(device)
    return [copy.deepcopy(base) for _ in range(args.units)]


@torch.no_grad()
def competence_matrix(ensemble, x, y, cluster, components, batch_size=2048):
    device = x.device
    sums = torch.zeros(
        (len(ensemble), components), device=device, dtype=torch.float64
    )
    counts = torch.zeros(components, device=device, dtype=torch.float64)

    for start in range(0, y.numel(), batch_size):
        xb = x[start : start + batch_size]
        yb = y[start : start + batch_size]
        cb = cluster[start : start + batch_size]

        ones = torch.ones(cb.numel(), device=device, dtype=torch.float64)
        counts.scatter_add_(0, cb, ones)

        for i, model in enumerate(ensemble):
            model.eval()
            loss = F.cross_entropy(model(xb), yb, reduction="none").double()
            sums[i].scatter_add_(0, cb, loss)

    mean_loss = sums / counts.clamp_min(1.0)
    return torch.exp(-mean_loss).float()


def differentiation(x):
    mean = x.mean(dim=0, keepdim=True)
    return float(((x - mean).square().sum(dim=1)).mean().item())


@torch.no_grad()
def update_strategy(a, competence, epsilon, exploration):
    units, components = a.shape
    mean = competence.mean(dim=0, keepdim=True)
    mean_minus = (units * mean - competence) / (units - 1)

    fitness = (
        0.5 * (1.0 - mean_minus).square()
        - 0.5 * (1.0 - mean).square()
    )
    mean_fitness = (a * fitness).sum(dim=1, keepdim=True)

    da = epsilon * (
        a * (fitness - mean_fitness)
        + exploration * (1.0 / components - a)
    )
    a = (a + da).clamp_min(1e-8)
    a /= a.sum(dim=1, keepdim=True)
    return a, fitness


def run_sigma(
    sigma,
    sigma_index,
    args,
    x_train,
    y_train,
    x_test,
    y_test,
    test_cluster,
    train_pools,
    probe_ids,
    device,
):
    run_seed = args.seed
    ensemble = identical_ensemble(args, device, run_seed)

    strategy_gen = torch.Generator(device=device).manual_seed(
        run_seed + 100003
    )
    a = torch.full(
        (args.units, args.components),
        1.0 / args.components,
        device=device,
    )
    a += args.strategy_noise * torch.randn(
        a.shape, generator=strategy_gen, device=device
    )
    a = a.clamp_min(1e-8)
    a /= a.sum(dim=1, keepdim=True)

    batch_gen = torch.Generator(device=device).manual_seed(run_seed + 200003)
    adjacency = torch.ones((args.units, args.units), device=device)
    adjacency -= torch.eye(args.units, device=device)

    probe_cluster = torch.cat(
        [
            torch.full(
                (args.probe_per_component,),
                mu,
                dtype=torch.long,
                device=device,
            )
            for mu in range(args.components)
        ]
    )
    x_probe = x_train[probe_ids]
    y_probe = y_train[probe_ids]

    g_history = []
    s_history = []
    budget_error = 0.0
    last_losses = []

    progress = tqdm(
        range(args.steps),
        desc=f"adaptive NN sigma={sigma:.3g}",
        unit="step",
        dynamic_ncols=True,
        leave=False,
    )

    for step in progress:
        batch_ids, batch_cluster = balanced_batch(
            train_pools, args.batch_per_component, batch_gen
        )
        xb = x_train[batch_ids]
        yb = y_train[batch_ids]

        # With an equal number of samples from every component and
        # g_i = T log a_i, Eq. (weights) reduces exactly to w=K a.
        weights = args.components * a[:, batch_cluster]
        budget_error = max(
            budget_error,
            float((weights.mean(dim=1) - 1.0).abs().max().item()),
        )

        losses = coupled_sgd_step(
            ensemble,
            [(xb, yb)] * args.units,
            learning_rate=args.learning_rate,
            coupling=sigma,
            weight_decay=args.weight_decay,
            adjacency=adjacency,
            sample_weights=[weights[i] for i in range(args.units)],
        )
        last_losses.append(float(losses.mean().item()))
        if len(last_losses) > 100:
            last_losses.pop(0)

        if (step + 1) % args.strategy_every == 0:
            competence = competence_matrix(
                ensemble,
                x_probe,
                y_probe,
                probe_cluster,
                args.components,
                batch_size=x_probe.shape[0],
            )
            g_history.append(differentiation(a))
            s_history.append(differentiation(competence))
            a, _ = update_strategy(
                a,
                competence,
                epsilon=args.epsilon,
                exploration=args.exploration,
            )

            progress.set_postfix(
                G=f"{g_history[-1]:.3g}",
                S=f"{s_history[-1]:.3g}",
                loss=f"{np.mean(last_losses):.3g}",
                refresh=False,
            )

    test_competence = competence_matrix(
        ensemble,
        x_test,
        y_test,
        test_cluster,
        args.components,
    )

    metrics = [evaluate(model, x_test, y_test) for model in ensemble]
    accuracy = float(np.mean([metric[0] for metric in metrics]))
    loss = float(np.mean([metric[1] for metric in metrics]))

    tail = max(1, int(args.tail_fraction * len(g_history)))
    g_tail = float(np.mean(g_history[-tail:]))
    s_probe_tail = float(np.mean(s_history[-tail:]))

    return {
        "sigma": float(sigma),
        "G": g_tail,
        "S": differentiation(test_competence),
        "S_probe": s_probe_tail,
        "accuracy": accuracy,
        "loss": loss,
        "budget_error": budget_error,
        "allocation": a.detach().cpu().tolist(),
        "competence": test_competence.detach().cpu().tolist(),
    }


def sigma_grid(args):
    if args.smoke:
        return np.asarray([0.03, 1.0, 30.0])
    if args.quick:
        return np.logspace(-2.0, 2.0, 9)
    return np.logspace(-2.0, 2.0, args.sigma_points)


def run(args):
    device = resolve_device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    if args.smoke:
        args.steps = 100
        args.strategy_every = 20
        args.partition_samples = min(args.partition_samples, 1500)
        args.probe_per_component = min(args.probe_per_component, 32)
        args.batch_per_component = min(args.batch_per_component, 8)
    elif args.quick:
        args.steps = min(args.steps, 6000)

    x_train, y_train, x_test, y_test = load_mnist(device)

    print("building common unsupervised representation", flush=True)
    train_cluster, test_cluster = fit_common_partition(
        x_train, x_test, args, device
    )
    train_sizes, class_distribution = cluster_summary(
        train_cluster, y_train, args.components
    )
    test_sizes, _ = cluster_summary(
        test_cluster, y_test, args.components
    )
    print("train component sizes:", train_sizes, flush=True)

    train_pools, probe_ids = split_environment(
        train_cluster,
        args.components,
        args.probe_per_component,
        args.partition_seed + 17,
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
            "cluster_class_distribution": class_distribution,
            "partition_uses_labels": False,
            "shared_minibatch": True,
            "identical_initial_networks": True,
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
            sigma_index,
            args,
            x_train,
            y_train,
            x_test,
            y_test,
            test_cluster,
            train_pools,
            probe_ids,
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--units", type=int, default=10)
    p.add_argument("--components", type=int, default=3)
    p.add_argument("--depth", type=int, default=1)
    p.add_argument("--width", type=int, default=20)
    p.add_argument("--steps", type=int, default=15000)
    p.add_argument("--sigma-points", type=int, default=17)
    p.add_argument("--batch-per-component", type=int, default=32)
    p.add_argument("--probe-per-component", type=int, default=128)
    p.add_argument("--strategy-every", type=int, default=25)
    p.add_argument("--epsilon", type=float, default=2.0)
    p.add_argument("--exploration", type=float, default=0.002)
    p.add_argument("--strategy-noise", type=float, default=0.01)
    p.add_argument("--tail-fraction", type=float, default=0.2)
    p.add_argument("--learning-rate", type=float, default=0.005)
    p.add_argument("--weight-decay", type=float, default=0.001)
    p.add_argument("--representation-dim", type=int, default=8)
    p.add_argument("--partition-samples", type=int, default=6000)
    p.add_argument("--partition-seed", type=int, default=17)
    p.add_argument("--kmeans-iterations", type=int, default=30)
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
