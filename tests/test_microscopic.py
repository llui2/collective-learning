"""Deterministic checks for finite-time specialization and transfer."""

import copy
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from collective_learning.microscopic import (
    advance, allocation_weights, checkpoints_for, cross_loss,
    make_population, run, task_data, training_stream, trial,
)


class MicroscopicTests(unittest.TestCase):
    def test_budget_and_private_tasks(self):
        weights = allocation_weights([0, 0.5, 1], 8, torch.device("cpu"))
        for w in weights:
            torch.testing.assert_close(w.mean(), torch.tensor(1.0))
        self.assertTrue(torch.all(weights[0][:8] == 0))
        self.assertTrue(torch.all(weights[2][8:] == 0))

    def test_depth_and_initialization(self):
        for depth in (0, 1, 2):
            models = make_population(4, depth, 2, 0.1, 7, "cpu")
            self.assertEqual(len(list(models[0].parameters())), depth + 1)
            self.assertEqual(tuple(models[0](torch.zeros(3, 2)).shape), (3, 2))
            same = make_population(4, depth, 2, 0.1, 7, "cpu")
            for first, second in zip(models, same):
                for p, q in zip(first.parameters(), second.parameters()):
                    torch.testing.assert_close(p, q)

    def test_paired_conditions_same_start(self):
        args = Namespace(
            units=4, width=2, batch=4, rate=0.05, decay=0.001,
            steps=5, evaluation=16, device="cpu",
        )
        a = trial(args, depth=1, scale=0.1, coupling=0, seed=1)
        b = trial(args, depth=1, scale=0.1, coupling=1, seed=1)
        torch.testing.assert_close(
            torch.tensor(a["history"][0]["uniform"]["cross_loss"]),
            torch.tensor(b["history"][0]["uniform"]["cross_loss"]),
        )
        self.assertEqual(a["history"][0]["advantage"], 0)
        self.assertEqual(a["history"][-1]["step"], 5)

    def test_coupling_transfers_knowledge_off_task(self):
        models = make_population(2, 1, 2, 0.2, 4, "cpu")
        data = task_data(128, torch.Generator().manual_seed(102), "cpu")
        stream = training_stream(
            120, 16, torch.Generator().manual_seed(20), "cpu"
        )
        uncoupled, coupled = copy.deepcopy(models), copy.deepcopy(models)
        advance(uncoupled, [1, 0], stream, 16, 0.1, 0, 0)
        advance(coupled, [1, 0], stream, 16, 0.1, 2, 0)
        e0, e1 = cross_loss(uncoupled, data), cross_loss(coupled, data)
        # The averaged off-task error improves even without any local examples.
        self.assertLess((e1[0, 1] + e1[1, 0]) / 2,
                        (e0[0, 1] + e0[1, 0]) / 2)

    def test_smoke_sweep(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "smoke.json"
            args = Namespace(
                depths="0,1", scales="0.1", couplings="0,0.6",
                seeds="0", steps=6, units=4, width=2, batch=4,
                rate=0.05, decay=0.001, evaluation=16, device="cpu",
                output=str(path), smoke=True,
            )
            result = run(args)
            self.assertEqual(len(result), 4)
            self.assertEqual(len(result[-1]["history"][-1]["specialists"]["cross_loss"]), 4)
            self.assertEqual(checkpoints_for(6)[-1], 6)
            self.assertTrue(path.with_suffix(".pdf").exists())
            saved = json.loads(path.read_text())
            self.assertTrue(saved["complete"])
            self.assertEqual(len(saved["trials"]), 4)


if __name__ == "__main__":
    unittest.main()
