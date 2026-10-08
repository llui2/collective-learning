"""Deterministic checks of coupled SGD and baseline effective dynamics."""

import copy
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from collective_learning.core import NeuralUnit, coupled_sgd_step
from collective_learning.theory import dynamics


class CoreTests(unittest.TestCase):
    def test_unit_weights_equal_unweighted_learning(self):
        torch.manual_seed(7)
        original = [NeuralUnit(input_size=3, output_size=2, depth=0) for _ in range(2)]
        weighted = copy.deepcopy(original)
        x = torch.randn(5, 3)
        y = torch.tensor([0, 1, 0, 0, 1])
        batches = [(x, y), (x.flip(0), y.flip(0))]
        settings = dict(learning_rate=0.04, coupling=0.5, weight_decay=0.01)
        coupled_sgd_step(original, batches, **settings)
        coupled_sgd_step(
            weighted, batches,
            sample_weights=[torch.ones(5), torch.ones(5)],
            **settings
        )
        for left, right in zip(original, weighted):
            for p, q in zip(left.parameters(), right.parameters()):
                torch.testing.assert_close(p, q, atol=1e-7, rtol=0)

    def test_weighted_step_is_weighted_loss_gradient(self):
        torch.manual_seed(3)
        model = NeuralUnit(input_size=2, output_size=2, depth=0)
        reference = copy.deepcopy(model)
        x = torch.tensor([[1.0, -1.0], [0.5, 0.5], [-1.0, 1.0]])
        y = torch.tensor([0, 1, 0])
        weights = torch.tensor([1.5, 0.5, 1.0])
        rate = 0.03
        loss = (weights * F.cross_entropy(reference(x), y, reduction="none")).mean()
        loss.backward()
        expected = [p.detach() - rate * p.grad for p in reference.parameters()]
        coupled_sgd_step(
            [model], [(x, y)], rate, coupling=0.0,
            weight_decay=0.0, sample_weights=[weights]
        )
        for actual, target in zip(model.parameters(), expected):
            torch.testing.assert_close(actual, target, atol=1e-7, rtol=0)

    def test_mean_field_linear_limit(self):
        m = np.array([-0.4, 0.2, 0.7])
        delta = np.array([0.1, -0.3, 0.4])
        dm = dynamics(0.0, m, 0, 0.2, 1.3, delta)
        self.assertAlmostEqual(dm.mean(), delta.mean() - 1.2 * m.mean())


if __name__ == "__main__":
    unittest.main()
