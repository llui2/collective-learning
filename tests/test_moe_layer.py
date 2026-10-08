"""Training, routing masks, load penalties and functional diagnostics."""

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch

from collective_learning.moe_layer import TokenMoE, metrics, run, tokens


class TokenMoETests(unittest.TestCase):
    def test_balance_target_and_teacher(self):
        x, y = tokens(20, torch.Generator().manual_seed(7), "cpu")
        self.assertEqual((x[:, 0] > 0).sum().item(), 10)
        torch.testing.assert_close(y, x[:, :1] * x[:, 1:2])

    def test_topk_capacity_and_expert_gradients(self):
        torch.manual_seed(3)
        model = TokenMoE(topk=1, capacity_factor=0.5)
        with torch.no_grad():
            model.router.weight.zero_()
            model.router.bias.copy_(torch.tensor([3., -3.]))
        x, y = tokens(20, torch.Generator().manual_seed(4), "cpu")
        prediction, p, mask = model(x)
        self.assertEqual(int(mask[:, 0].sum()), 5)
        self.assertEqual(int(mask[:, 1].sum()), 0)
        self.assertEqual(int((mask.sum(dim=1) == 0).sum()), 15)
        F = ((prediction-y)**2).mean()
        F.backward()
        self.assertTrue(any(param.grad is not None for param in model.experts[0].parameters()))
        self.assertTrue(all(param.grad is None or torch.all(param.grad == 0)
                            for param in model.experts[1].parameters()))

    def test_dense_top2_is_softmax_mixture(self):
        model = TokenMoE(topk=2)
        x, y = tokens(16, torch.Generator().manual_seed(11), "cpu")
        out, p, mask = model(x)
        reference = (
            p[:, :1] * model.experts[0](x)
            + p[:, 1:] * model.experts[1](x)
        )
        torch.testing.assert_close(out, reference)
        self.assertTrue(torch.all(mask == 1))
        result = metrics(model, x, y)
        self.assertEqual(result["dropped_fraction"], 0)
        self.assertEqual(len(result["expert_cross_loss"]), 2)

    def test_smoke_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            args = Namespace(
                topks="1,2", balances="0.2", seeds="0",
                width=4, temperature=1, capacity=1.25,
                steps=6, record_every=2, batch=16, evaluation=32,
                expert_rate=.05, router_rate=.05, device="cpu",
                output=str(Path(directory) / "layer.json"), smoke=True,
            )
            d = run(args)
            self.assertEqual(len(d["trials"]), 2)
            self.assertTrue(Path(directory, "layer.pdf").exists())


if __name__ == "__main__":
    unittest.main()
