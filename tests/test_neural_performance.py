"""Invariants for matched neural performance comparisons (no MNIST download)."""

import copy
import unittest

import torch

from collective_learning.core import NeuralUnit
from collective_learning.neural_performance import (
    collective_metrics,
    split_training_pool,
    strategy_observables,
)


class NeuralPerformanceTests(unittest.TestCase):
    def test_disjoint_splits_and_reproducibility(self):
        device = torch.device("cpu")
        train, probe, validation = split_training_pool(
            100, 12, 17, 7, device
        )
        all_ids = torch.cat((train, probe, validation))
        self.assertEqual(len(torch.unique(all_ids)), 100)
        self.assertEqual(len(train), 71)
        other = split_training_pool(100, 12, 17, 7, device)
        for left, right in zip((train, probe, validation), other):
            self.assertTrue(torch.equal(left, right))

    def test_collective_loss_matches_identical_network(self):
        torch.manual_seed(8)
        model = NeuralUnit(input_size=4, output_size=3, depth=0)
        ensemble = [model, copy.deepcopy(model), copy.deepcopy(model)]
        x = torch.randn(19, 1, 2, 2)
        y = torch.randint(0, 3, (19,))
        metrics = collective_metrics(ensemble, x, y, batch_size=7)
        self.assertAlmostEqual(
            metrics["ensemble_nll"], metrics["individual_nll"], places=6
        )
        self.assertAlmostEqual(
            metrics["ensemble_accuracy"], metrics["individual_accuracy"],
            places=6,
        )
        self.assertGreaterEqual(metrics["collective_loss"], 0)
        self.assertLessEqual(metrics["collective_loss"], 0.5)

    def test_initially_identical_competence_has_zero_divergence(self):
        torch.manual_seed(3)
        g = torch.zeros(4, 2)
        phi = torch.randn(13, 2)
        competence = torch.full((4, 13), 0.3)
        obs = strategy_observables(g, phi, competence, 0.5)
        self.assertAlmostEqual(obs["R"], 0.0, places=6)
        self.assertAlmostEqual(obs["S_sample"], 0.0, places=6)
        self.assertAlmostEqual(obs["routing_entropy"], 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
