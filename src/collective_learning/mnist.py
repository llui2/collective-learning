"""Reproduce the released MNIST collective-learning experiment.

The code follows the public notebook implementation while separating the
neural dynamics from data handling and measurement.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets

from .core import NeuralUnit, coupled_sgd_step, cross_accuracy, evaluate, mean_parameter


def resolve_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def released_sigma_grid():
    low = np.logspace(-4, 1, 25)
    high = np.linspace(10, 390, 26)
    return np.concatenate((low, high[1:]))


def load_mnist(device):
    train = datasets.MNIST("data", train=True, download=True)
    test = datasets.MNIST("data", train=False, download=True)

    x_train = train.data.float().div_(255.0).unsqueeze(1).to(device)
    y_train = train.targets.to(device)
    x_test = test.data.float().div_(255.0).unsqueeze(1).to(device)
    y_test = test.targets.to(device)
    class_indices = [torch.where(y_train == c)[0] for c in range(10)]
    return x_train, y_train, x_test, y_test, class_indices


def make_ensemble(depth, width, device, init):
    ensemble = [NeuralUnit(depth=depth, width=width).to(device) for _ in range(10)]
    if init == "normal":
        with torch.no_grad():
            for model in ensemble:
                for p in model.parameters():
                    p.normal_(0.0, 1.0)
    return ensemble


def sample_private_batches(x, y, class_indices, batch_size, generator):
    batches = []
    for label in range(10):
        ids = class_indices[label]
        picks = torch.randint(ids.numel(), (batch_size,), generator=generator, device=ids.device)
        idx = ids[picks]
        batches.append((x[idx], y[idx]))
    return batches


def train_steps(
    ensemble,
    steps,
    sigma,
    x_train,
    y_train,
    class_indices,
    args,
    generator,
    measure=False,
    x_eval=None,
    y_eval=None,
):
    magnetization = []
    losses = []
    accuracy = []

    for step in range(steps):
        batches = sample_private_batches(
            x_train, y_train, class_indices, args.batch_size, generator
        )
        coupled_sgd_step(
            ensemble,
            batches,
            learning_rate=args.learning_rate,
            coupling=sigma,
            weight_decay=args.weight_decay,
        )

        if measure and step % args.eval_every == 0:
            magnetization.append(np.mean([mean_parameter(model) for model in ensemble]))
            metrics = [evaluate(model, x_eval, y_eval) for model in ensemble]
            accuracy.append(np.mean([m[0] for m in metrics]))
            losses.append(np.mean([m[1] for m in metrics]))

    if not measure:
        return None

    return {
        "magnetization": float(np.mean(magnetization)),
        "loss": float(np.mean(losses)),
        "accuracy": float(np.mean(accuracy)),
    }


def run(args):
    device = resolve_device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    x_train, y_train, x_test, y_test, class_indices = load_mnist(device)
    gen = torch.Generator(device=device).manual_seed(args.seed)

    sigmas = released_sigma_grid()
    depths = [0, 1, 2]
    if args.smoke:
        sigmas = np.array([1e-4, 1.0, 30.0])
        depths = [0, 1]
        args.transient_steps = 30
        args.measure_steps = 30
        args.eval_every = 10
        args.runs = 1
        args.test_samples = 100

    test_pick = torch.randperm(y_test.numel(), generator=gen, device=device)[: args.test_samples]
    x_eval = x_test[test_pick]
    y_eval = y_test[test_pick]

    results = []
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    for run_idx in range(args.runs):
        for depth in depths:
            ensemble = None

            for sigma in sigmas:
                if ensemble is None or args.protocol == "nonadiabatic":
                    ensemble = make_ensemble(depth, args.width, device, args.init)

                train_steps(
                    ensemble,
                    args.transient_steps,
                    sigma,
                    x_train,
                    y_train,
                    class_indices,
                    args,
                    gen,
                )
                measured = train_steps(
                    ensemble,
                    args.measure_steps,
                    sigma,
                    x_train,
                    y_train,
                    class_indices,
                    args,
                    gen,
                    measure=True,
                    x_eval=x_eval,
                    y_eval=y_eval,
                )

                ax = cross_accuracy(ensemble, x_test, y_test)
                row = {
                    "run": run_idx,
                    "depth": depth,
                    "sigma_equation": float(sigma),
                    "sigma_released_plot": float(sigma / 10.0),
                    "magnetization": measured["magnetization"],
                    "loss": measured["loss"],
                    "accuracy": measured["accuracy"],
                    "cross_accuracy": ax.tolist(),
                }
                results.append(row)
                out.write_text(json.dumps({"config": vars(args), "results": results}, indent=2))
                print(
                    f"run={run_idx} D={depth} sigma={sigma:g} "
                    f"loss={row['loss']:.4f} acc={row['accuracy']:.3f}"
                )

    print(f"wrote {out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--transient-steps", type=int, default=20000)
    p.add_argument("--measure-steps", type=int, default=20000)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--test-samples", type=int, default=50)
    p.add_argument("--learning-rate", type=float, default=0.005)
    p.add_argument("--weight-decay", type=float, default=0.001)
    p.add_argument("--width", type=int, default=20)
    p.add_argument("--protocol", choices=("nonadiabatic", "adiabatic"), default="nonadiabatic")
    p.add_argument("--init", choices=("source", "normal"), default="source")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="results/mnist_baseline.json")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
