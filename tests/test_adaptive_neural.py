"""Fast synthetic checks for the microscopic adaptive-neural experiment."""

import unittest

import numpy as np
import torch

from collective_learning.adaptive_neural import (
    natural_strategy_step,
    relative_window_drift,
    routing_probabilities,
    sample_fitness,
    strategy_weights,
)


class AdaptiveNeuralTests(unittest.TestCase):
    def test_equal_budget_for_each_learner(self):
        torch.manual_seed(4)
        g = torch.randn(4, 3)
        phi = torch.randn(13, 3)
        weights = strategy_weights(g, phi, temperature=0.5)
        self.assertEqual(tuple(weights.shape), (4, 13))
        self.assertTrue(
            torch.allclose(weights.mean(dim=1), torch.ones(4), atol=1e-6)
        )

    def test_uniform_strategy_gives_uniform_sample_weights(self):
        g = torch.zeros(3, 2)
        phi = torch.randn(8, 2)
        weights = strategy_weights(g, phi, temperature=0.5)
        self.assertTrue(torch.allclose(weights, torch.ones_like(weights)))

    def test_identical_phenotypes_have_zero_marginal_contribution(self):
        competence = torch.full((4, 7), 0.6)
        self.assertTrue(
            torch.allclose(sample_fitness(competence), torch.zeros_like(competence))
        )

    def test_adaptation_changes_sample_preferences(self):
        phi = torch.tensor([[-1., 0.], [1., 0.], [0., 1.]])
        g = torch.zeros(2, 2)
        fitness = torch.tensor([[0., 1., 0.], [1., 0., 0.]])
        before = routing_probabilities(g, phi, 1.0)
        updated, max_step = natural_strategy_step(
            g, phi, fitness, temperature=1.0,
            rate=0.1, exploration=0.0, ridge=1e-3,
            max_step=1.0
        )
        after = routing_probabilities(updated, phi, 1.0)
        self.assertGreater(max_step, 0.0)
        self.assertGreater(float(after[0, 1]), float(before[0, 1]))
        self.assertGreater(float(after[1, 0]), float(before[1, 0]))

    def test_stationarity_drift(self):
        steps = list(range(0, 12001, 25))
        plateau = np.ones(len(steps)) * 0.2
        growing = np.linspace(0.01, 0.5, len(steps))
        self.assertAlmostEqual(
            relative_window_drift(steps, plateau, 2000), 0.0, places=6
        )
        self.assertGreater(
            relative_window_drift(steps, growing, 2000), 0.02
        )


if __name__ == "__main__":
    unittest.main()
