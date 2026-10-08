"""Core dynamics for coupled neural learners.

The baseline implements Eq. (2) of Arola-Fernandez and Lacasa,
Phys. Rev. Research 6, L042040 (2024), using simultaneous diffusive
coupling evaluated before the local SGD update.
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class NeuralUnit(nn.Module):
    def __init__(self, input_size=28 * 28, output_size=10, depth=1, width=20):
        super().__init__()
        if depth == 0:
            layers = [nn.Linear(input_size, output_size)]
        else:
            layers = [nn.Linear(input_size, width)]
            for _ in range(depth - 1):
                layers += [nn.ReLU(), nn.Linear(width, width)]
            layers += [nn.ReLU(), nn.Linear(width, output_size)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x.flatten(start_dim=1))


def parameter_vector(model):
    return parameters_to_vector(model.parameters()).detach().clone()


def mean_parameter(model):
    return parameter_vector(model).mean().item()


def mean_square_parameter(model):
    w = parameter_vector(model)
    return w.square().mean().item()


def coupled_sgd_step(
    ensemble,
    batches,
    learning_rate,
    coupling,
    weight_decay,
    adjacency=None,
    sample_weights=None,
    loss_fn=None,
):
    """Apply one local SGD step followed by simultaneous diffusive coupling.

    coupling is sigma in
        theta_i <- theta_i - eta grad L_i
                   + eta sigma / N sum_j q_ij (theta_j - theta_i).

    sample_weights optionally reweights examples within each learner's batch.
    Its default None exactly recovers the baseline dynamics.

    loss_fn returns a vector of per-sample losses. Its default is
    cross-entropy; passing a squared loss permits deep-linear regression.
    """
    n = len(ensemble)
    if len(batches) != n:
        raise ValueError("one batch is required per learner")

    if adjacency is None:
        device = next(ensemble[0].parameters()).device
        adjacency = torch.ones((n, n), device=device) - torch.eye(n, device=device)
    else:
        adjacency = adjacency.to(next(ensemble[0].parameters()).device)

    before = torch.stack([parameter_vector(model) for model in ensemble])
    losses = []

    for i, (model, batch) in enumerate(zip(ensemble, batches)):
        x, y = batch
        model.train()
        model.zero_grad(set_to_none=True)

        prediction = model(x)
        if loss_fn is None:
            per_sample = F.cross_entropy(prediction, y, reduction="none")
        else:
            per_sample = loss_fn(prediction, y)
        if per_sample.ndim != 1 or per_sample.numel() != len(y):
            raise ValueError("loss_fn must return one loss per sample")
        if sample_weights is None or sample_weights[i] is None:
            loss = per_sample.mean()
        else:
            w = sample_weights[i].to(per_sample.device, per_sample.dtype)
            if w.ndim != 1 or w.numel() != per_sample.numel():
                raise ValueError("sample weights must match the batch size")
            loss = (w * per_sample).mean()

        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.add_(p.grad + weight_decay * p, alpha=-learning_rate)
        losses.append(loss.detach())

    if coupling:
        after = torch.stack([parameter_vector(model) for model in ensemble])
        degree = adjacency.sum(dim=1, keepdim=True)
        diffusion = adjacency @ before - degree * before
        after += learning_rate * coupling * diffusion / n
        for model, vector in zip(ensemble, after):
            vector_to_parameters(vector, model.parameters())

    return torch.stack(losses)


@torch.no_grad()
def evaluate(model, x, y, batch_size=1024):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for start in range(0, y.numel(), batch_size):
        xb = x[start : start + batch_size]
        yb = y[start : start + batch_size]
        logits = model(xb)
        total_loss += F.cross_entropy(logits, yb, reduction="sum").item()
        correct += (logits.argmax(dim=1) == yb).sum().item()
        total += yb.numel()
    return correct / total, total_loss / total


@torch.no_grad()
def cross_accuracy(ensemble, x, y, num_classes=10, batch_size=1024):
    matrix = torch.zeros((len(ensemble), num_classes), dtype=torch.float64)
    for label in range(num_classes):
        mask = y == label
        xl = x[mask]
        yl = y[mask]
        for i, model in enumerate(ensemble):
            matrix[i, label] = evaluate(model, xl, yl, batch_size=batch_size)[0]
    return matrix.numpy()
