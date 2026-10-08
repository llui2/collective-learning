"""Allocation evolution: budget, fairness, reproducibility and controls."""

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

from collective_learning.adaptive import (
    STRATEGIES, compact, propose, run, trial,
)


class AdaptiveTests(unittest.TestCase):
    def args(self):
        return Namespace(
            scales="0.03", couplings="0,1", windows="2", seeds="0",
            steps=8, units=4, width=2, batch=4, rate=0.05,
            decay=0.001, mutation=0.2, evaluation=12,
            thresholds="0.2,0.1", device="cpu", jobs=1,
            output="results/microscopic_adaptive.json", smoke=False,
        )

    def test_mutation_preserves_bounds_and_single_agent(self):
        a = np.array([0.5, 0.5, 0.5, 0.5])
        changed = propose(a, 2, 9.0)
        np.testing.assert_allclose(changed, [0.5, 0.5, 0.98, 0.5])
        np.testing.assert_allclose(a, 0.5)
        self.assertEqual(np.count_nonzero(changed != a), 1)

    def test_paired_start_and_fixed_controls(self):
        args = self.args()
        first = trial(args, 0.03, 1, 2, 0)
        second = trial(args, 0.03, 1, 2, 0)
        self.assertEqual(len(first["history"]), 5)
        self.assertEqual(first["history"], second["history"])
        self.assertEqual(first["first_crossing"], second["first_crossing"])
        at_start = first["history"][0]
        self.assertEqual(len(STRATEGIES), 5)
        self.assertEqual(at_start["uniform"]["individual_loss"],
                         at_start["evolving"]["individual_loss"])
        self.assertEqual(at_start["uniform"]["individual_loss"],
                         at_start["specialists"]["individual_loss"])
        for entry in first["history"]:
            self.assertEqual(entry["uniform"]["polarization"], 0)
            self.assertEqual(entry["specialists"]["polarization"], 1)
            for strategy in STRATEGIES:
                self.assertTrue(np.isfinite(entry[strategy]["individual_loss"]))
                self.assertTrue(
                    all(0 <= x <= 1 for x in entry[strategy]["allocations"])
                )
        self.assertEqual(
            first["history"][0]["frozen"]["allocations"],
            first["history"][-1]["frozen"]["allocations"]
        )
        self.assertEqual(first["proposals"], 4)
        self.assertLessEqual(first["accepted"], first["proposals"])

    def test_smoke_and_censoring_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.args()
            args.output = str(Path(tmp) / "adaptive.json")
            args.smoke = True
            trials = run(args)
            self.assertEqual(len(trials), 2)
            summary = Path(tmp) / "adaptive_summary.json"
            self.assertTrue(summary.exists())
            self.assertTrue(Path(tmp, "adaptive.pdf").exists())
            data = json.loads(summary.read_text())
            self.assertTrue(data["complete"])
            self.assertEqual(len(data["trials"]), 2)
            self.assertIn("0.2", data["trials"][0]["first_crossing"])
            self.assertIn("limitation", data)
            self.assertTrue(
                all(value is None or value >= 0
                    for value in data["trials"][0]["first_crossing"]["0.2"].values())
            )

    def test_summary_is_compact_and_excludes_cross_task_matrix(self):
        args = self.args()
        t = trial(args, 0.03, 1, 2, 0)
        s = compact([t], args, True)
        self.assertNotIn("cross_loss", json.dumps(s))
        self.assertEqual(len(s["trials"][0]["history"]), 5)


if __name__ == "__main__":
    unittest.main()
