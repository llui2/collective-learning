"""Collective performance of the actual coupled neural learners.

Identical initial networks and common minibatches are used for matched adaptive,
frozen-strategy and uniform-routing controls. The learning rule is unchanged;
only the strategy evolution differs. Performance is the held-out ensemble
probability and the mean competence loss, not a fitted effective-theory proxy.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .adaptive_neural import (
    fit_shared_representation,
    identical_ensemble,
    load_mnist,
    natural_strategy_step,
    resolve_device,
    sample_competence,
    sample_fitness,
    strategy_weights,
    routing_probabilities,
)
from .core import coupled_sgd_step


VERSION = 1


@torch.no_grad()
def collective_metrics(ensemble, x, y, batch_size=1024):
    """Evaluate a uniform probability ensemble on data never used for routing."""
    total = y.numel()
    for model in ensemble:
        model.eval()
    correct = nll = collective_loss = individual_nll = individual_correct = 0.0

    for start in range(0, total, batch_size):
        xb = x[start : start + batch_size]
        yb = y[start : start + batch_size]
        probabilities = torch.stack(
            [model(xb).softmax(dim=-1) for model in ensemble], dim=0
        )
        mean_probability = probabilities.mean(dim=0)
        p_true = mean_probability.gather(1, yb[:, None]).squeeze(1)
        correct += (mean_probability.argmax(dim=1) == yb).sum().item()
        nll += -p_true.clamp_min(1e-12).log().sum().item()
        collective_loss += 0.5 * (1.0 - p_true).square().sum().item()

        individual_true = probabilities.gather(
            2, yb[None, :, None].expand(len(ensemble), -1, 1)
        ).squeeze(-1)
        individual_nll += (
            -individual_true.clamp_min(1e-12).log().mean(dim=0).sum().item()
        )
        individual_correct += (
            probabilities.argmax(dim=-1) == yb[None, :]
        ).float().mean(dim=0).sum().item()

    return {
        "ensemble_accuracy": correct / total,
        "ensemble_nll": nll / total,
        "collective_loss": collective_loss / total,
        "individual_accuracy": individual_correct / total,
        "individual_nll": individual_nll / total,
    }


@torch.no_grad()
def strategy_observables(g, phi, competence, temperature):
    p = routing_probabilities(g, phi, temperature)
    means = p @ phi
    R = (means - means.mean(dim=0, keepdim=True)).square().sum(dim=1).mean()
    sample_variance = (
        competence - competence.mean(dim=0, keepdim=True)
    ).square().mean()
    entropy = -(p * p.clamp_min(1e-12).log()).sum(dim=1).mean()
    entropy /= np.log(p.shape[1])
    return {
        "R": float(R),
        "S_sample": float(sample_variance),
        "routing_entropy": float(entropy),
    }


def split_training_pool(size, n_probe, n_validation, seed, device):
    """Training, strategy probe, and validation labels are disjoint."""
    if n_probe + n_validation >= size:
        raise ValueError("probe and validation subsets exceed training data")
    gen = torch.Generator(device=device).manual_seed(seed)
    perm = torch.randperm(size, generator=gen, device=device)
    return (
        perm[n_probe + n_validation :],
        perm[:n_probe],
        perm[n_probe : n_probe + n_validation],
    )


def run(args):
    if args.units < 2:
        raise ValueError("at least two coupled learners are required")
    if args.steps < 1 or args.strategy_every < 1 or args.eval_every < 1:
        raise ValueError("step intervals must be positive")

    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    x_train, y_train, x_test, y_test = load_mnist(device)
    train_ids, probe_ids, val_ids = split_training_pool(
        len(y_train), args.probe_samples, args.validation_samples,
        args.split_seed, device
    )
    # Fit PCA only on the SGD pool; the probe, validation and test examples
    # do not influence the shared representation fitted for this experiment.
    _, phi_train = fit_shared_representation(
        x_train[train_ids], x_train, args, device
    )

    x_probe = x_train[probe_ids]
    y_probe = y_train[probe_ids]
    phi_probe = phi_train[probe_ids]
    x_val, y_val = x_train[val_ids], y_train[val_ids]

    ensemble = identical_ensemble(args, device, args.seed)
    strategy_gen = torch.Generator(device=device).manual_seed(
        args.seed + 100003
    )
    g = args.strategy_noise * torch.randn(
        (args.units, args.representation_dim),
        device=device,
        generator=strategy_gen,
    )
    if args.control == "uniform":
        g.zero_()

    gen = torch.Generator(device=device).manual_seed(args.seed + 200003)
    adjacency = (
        torch.ones((args.units, args.units), device=device)
        - torch.eye(args.units, device=device)
    )
    history = []
    budget_error = 0.0

    def record(step):
        competence = sample_competence(ensemble, x_probe, y_probe)
        observation = strategy_observables(
            g, phi_probe, competence, args.temperature
        )
        metrics = collective_metrics(ensemble, x_val, y_val)
        observation.update(metrics)
        observation["step"] = step
        history.append(observation)
        return competence

    competence = record(0)
    iterator = tqdm(
        range(args.steps), desc=(
            f"NN {args.control} sigma={args.sigma:g} "
            f"rate={args.strategy_rate:g}"
        ), unit="step", dynamic_ncols=True
    )
    for step in iterator:
        batch_indices = train_ids[
            torch.randint(
                len(train_ids),
                (args.batch_size,),
                generator=gen,
                device=device,
            )
        ]
        xb, yb, pb = (
            x_train[batch_indices],
            y_train[batch_indices],
            phi_train[batch_indices],
        )

        if args.control == "uniform":
            weights = None
        else:
            weights = strategy_weights(g, pb, args.temperature)
            budget_error = max(
                budget_error,
                float((weights.mean(dim=1) - 1.0).abs().max().item())
            )

        coupled_sgd_step(
            ensemble, [(xb, yb)] * args.units,
            learning_rate=args.learning_rate,
            coupling=args.sigma,
            weight_decay=args.weight_decay,
            adjacency=adjacency,
            sample_weights=(
                [weights[i] for i in range(args.units)]
                if weights is not None else None
            ),
        )

        if (step + 1) % args.strategy_every == 0:
            if args.control == "adaptive":
                competence = sample_competence(ensemble, x_probe, y_probe)
                fitness = sample_fitness(competence)
                g, _ = natural_strategy_step(
                    g, phi_probe, fitness,
                    temperature=args.temperature,
                    rate=args.strategy_rate,
                    exploration=args.exploration,
                    ridge=args.fisher_ridge,
                    max_step=args.max_strategy_step,
                )

        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            record(step + 1)
            latest = history[-1]
            iterator.set_postfix(
                CE=f"{latest['ensemble_nll']:.3g}",
                R=f"{latest['R']:.3g}",
                H=f"{latest['routing_entropy']:.3g}",
                refresh=False,
            )

    # The MNIST test set is measured once, at the predeclared final step.
    test_metrics = collective_metrics(ensemble, x_test, y_test)
    output = {
        "version": VERSION,
        "complete": True,
        "config": vars(args),
        "n_training": len(train_ids),
        "n_probe": len(probe_ids),
        "n_validation": len(val_ids),
        "budget_error": budget_error,
        "validation": history,
        "test_final": test_metrics,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2))
    print(
        f"saved {out} | val CE={history[-1]['ensemble_nll']:.4f} "
        f"| val collective loss={history[-1]['collective_loss']:.4f} "
        f"| test CE={test_metrics['ensemble_nll']:.4f} "
        f"| test accuracy={test_metrics['ensemble_accuracy']:.4f}",
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", choices=("adaptive", "frozen", "uniform"),
                        required=True)
    parser.add_argument("--sigma", type=float, required=True)
    parser.add_argument("--strategy-rate", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--units", type=int, default=10)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--strategy-every", type=int, default=25)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--exploration", type=float, default=0.0005)
    parser.add_argument("--strategy-noise", type=float, default=0.02)
    parser.add_argument("--fisher-ridge", type=float, default=1e-3)
    parser.add_argument("--max-strategy-step", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--representation-dim", type=int, default=8)
    parser.add_argument("--representation-seed", type=int, default=13)
    parser.add_argument("--representation-samples", type=int, default=6000)
    parser.add_argument("--probe-samples", type=int, default=512)
    parser.add_argument("--validation-samples", type=int, default=2048)
    parser.add_argument("--split-seed", type=int, default=23)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
