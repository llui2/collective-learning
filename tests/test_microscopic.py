"""Small checks for microscopic allocation and its common-task dynamics."""

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from collective_learning.microscopic import (
    allocation_weights, cross_loss, ensemble_loss, run, task_data, training_stream,
)


class MicroscopicTests(unittest.TestCase):
    def test_equal_total_budget(self):
        a = np.array([0.02, 0.5, 0.98])
        weights = allocation_weights(a, 7, torch.device("cpu"))
        for w in weights:
            torch.testing.assert_close(w.mean(), torch.tensor(1.0), atol=1e-6, rtol=0)

    def test_two_tasks_are_complementary(self):
        device = torch.device("cpu")
        data = task_data(16, torch.Generator().manual_seed(4), device)
        self.assertTrue(torch.all(data[0][0][:, 1] == 0))
        self.assertTrue(torch.all(data[1][0][:, 0] == 0))
        for x, y in data:
            torch.testing.assert_close(y, x)
        stream = training_stream(2, 8, torch.Generator().manual_seed(4), device)
        self.assertEqual(stream[0][0].shape, (16, 2))

    def test_rank_one_cannot_fit_both_tasks_but_ensemble_can(self):
        data = [
            (torch.tensor([[1.0, 0.0]]), torch.tensor([[1.0, 0.0]])),
            (torch.tensor([[0.0, 1.0]]), torch.tensor([[0.0, 1.0]])),
        ]
        models = [
            nn.Sequential(nn.Linear(2, 1, bias=False), nn.Linear(1, 2, bias=False))
            for _ in range(2)
        ]
        with torch.no_grad():
            for k, model in enumerate(models):
                model[0].weight.zero_()
                model[1].weight.zero_()
                model[0].weight[0, k] = 1.0
                model[1].weight[k, 0] = 1.0
        matrix = cross_loss(models, data)
        np.testing.assert_allclose(matrix, [[0, 0.5], [0.5, 0]], atol=1e-7)
        self.assertAlmostEqual(matrix.mean(), 0.25)
        self.assertAlmostEqual(ensemble_loss(models, data), 0.125)

    def test_smoke_run_and_cross_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            args = Namespace(
                coupling=0.03, rounds=2, window=2, units=3, width=1,
                batch=8, rate=0.08, decay=0.001, mutation=0.15,
                evaluation=16, seed=1, device="cpu",
                output=str(Path(directory) / "pilot.json"), smoke=True,
            )
            result = run(args)
            self.assertEqual(len(result["history"]), 3)
            self.assertEqual(len(result["test"]["evolving"]["cross_loss"]), 3)
            self.assertTrue(np.isfinite(result["test"]["evolving"]["ensemble_loss"]))
            self.assertTrue(Path(args.output).with_suffix(".pdf").exists())


if __name__ == "__main__":
    unittest.main()
