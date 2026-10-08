"""Microscopic controls for transfer, capacity and evolutionary allocation."""

import copy
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from collective_learning.microscopic import (
    advance, allocation_weights, cross_loss, population_loss, run,
    task_data, training_stream,
)


class MicroscopicTests(unittest.TestCase):
    def test_learning_budget_and_private_task_weights(self):
        a = np.array([0.0, 0.5, 1.0])
        w = allocation_weights(a, 8, torch.device("cpu"))
        for item in w:
            torch.testing.assert_close(item.mean(), torch.tensor(1.0))
        self.assertTrue(torch.all(w[0][:8] == 0))
        self.assertTrue(torch.all(w[2][8:] == 0))

    def test_common_teacher_and_private_classes(self):
        data = task_data(8, torch.Generator().manual_seed(4), "cpu")
        self.assertTrue(torch.all(data[0][0][:, 1] == 0))
        self.assertTrue(torch.all(data[1][0][:, 0] == 0))
        for x, y in data:
            torch.testing.assert_close(x, y)
        stream = training_stream(2, 8, torch.Generator().manual_seed(9), "cpu")
        self.assertEqual(stream[0][0].shape, (16, 2))

    def test_coupling_transfers_unstudied_task(self):
        initial = [
            nn.Sequential(nn.Linear(2, 2, bias=False), nn.Linear(2, 2, bias=False))
            for _ in range(2)
        ]
        for model in initial:
            with torch.no_grad():
                for layer in model:
                    layer.weight.copy_(0.5 * torch.eye(2))
        data = [
            (torch.tensor([[1.0, 0.0]]), torch.tensor([[1.0, 0.0]])),
            (torch.tensor([[0.0, 1.0]]), torch.tensor([[0.0, 1.0]])),
        ]
        args = Namespace(batch=1, rate=0.1, decay=0.0, coupling=0.0)
        isolated, coupled = copy.deepcopy(initial), copy.deepcopy(initial)
        for _ in range(40):
            advance(isolated, [1.0, 0.0], [(torch.eye(2), torch.eye(2))], args)
        args.coupling = 1.0
        for _ in range(40):
            advance(coupled, [1.0, 0.0], [(torch.eye(2), torch.eye(2))], args)
        alone = cross_loss(isolated, data)
        linked = cross_loss(coupled, data)
        self.assertLess(linked[0, 1], alone[0, 1])
        self.assertLess(linked[1, 0], alone[1, 0])
        self.assertAlmostEqual(population_loss(isolated, data), alone.mean())

    def test_smoke_sweep_and_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            args = Namespace(
                couplings="0,0.6", seeds="0", rounds=2, window=2,
                units=4, width=2, batch=8, rate=0.04, decay=0.01,
                mutation=0.15, evaluation=24, device="cpu",
                output=str(Path(directory) / "pilot.json"), smoke=True,
            )
            trials = run(args)
            self.assertEqual(len(trials), 2)
            self.assertEqual(len(trials[0]["history"]), 3)
            self.assertEqual(np.shape(trials[0]["test"]["evolving"]["cross_loss"]), (4, 2))
            self.assertTrue(np.isfinite(trials[1]["test"]["uniform"]["individual_loss"]))
            self.assertTrue(Path(args.output).with_suffix(".pdf").exists())


if __name__ == "__main__":
    unittest.main()
