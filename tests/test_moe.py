"""Numerical validation of the physical router-expert reduction."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from collective_learning.moe import (
    dynamics, eigenvalues, energy_and_gradient, jacobian,
    observable, parse_args, routing, run, trial,
)


class ReducedMoETests(unittest.TestCase):
    def test_expected_loss_agrees_with_autograd(self):
        state = np.array([0.12, -0.37, 0.5, -0.22])
        params = dict(temperature=1.4, expert_decay=0.06,
                      router_decay=0.12, balance=0.7)
        loss, grad = energy_and_gradient(state, **params)
        v = torch.tensor(state, dtype=torch.float64, requires_grad=True)
        m, d, q, b = v.unbind()
        c = torch.tensor([-1., 1.], dtype=torch.float64)
        p = torch.sigmoid((b + q * c) / params["temperature"])
        pred = m + (2 * p - 1) * d
        objective = 0.25 * ((pred - c) ** 2).sum()
        objective += params["expert_decay"] * (m*m + d*d)
        objective += 0.5 * params["router_decay"] * (q*q + b*b)
        objective += 0.5 * params["balance"] * (p.mean()-0.5)**2
        objective.backward()
        self.assertAlmostEqual(loss, objective.item(), places=12)
        np.testing.assert_allclose(grad, v.grad.numpy(), rtol=1e-11, atol=1e-11)

    def test_jacobian_matches_finite_differences(self):
        pars = (1.3, .21, .27, .08, .09, .4)
        eps = 1e-5
        estimate = np.column_stack([
            (dynamics(eps * np.eye(4)[i], *pars)
             - dynamics(-eps * np.eye(4)[i], *pars)) / (2*eps)
            for i in range(4)
        ])
        np.testing.assert_allclose(jacobian(*pars), estimate, rtol=1e-8, atol=1e-9)

    def test_specialization_and_collapse_are_distinct(self):
        balanced = observable(np.array([0, 1, 2, 0]), 1)
        collapse = observable(np.array([0, 0, 0, 2]), 1)
        self.assertGreater(balanced["specialization"], 0.5)
        self.assertAlmostEqual(balanced["load_imbalance"], 0, places=12)
        self.assertAlmostEqual(collapse["specialization"], 0, places=12)
        self.assertGreater(collapse["load_imbalance"], 0.4)

    def test_stability_transition_and_coupled_rates(self):
        args = (.25, .25, .08, .08, .2)
        self.assertGreater(max(eigenvalues(1, *args)), 0)
        self.assertLess(max(eigenvalues(6, *args)), 0)
        # The collapse mode is damped by balancing, not the specialization mode.
        j = jacobian(1, *args)
        self.assertAlmostEqual(j[3, 3], -.25 * (.08 + .2 / 16))
        self.assertNotEqual(j[1, 2], 0)
        self.assertNotEqual(j[2, 1], 0)

    def test_seeded_symmetric_state_is_stationary(self):
        value = dynamics(np.zeros(4), 1, .25, .25, .08, .08, .2)
        np.testing.assert_allclose(value, 0, atol=1e-14)

    def test_smoke_creates_two_panel_figure(self):
        with tempfile.TemporaryDirectory() as directory:
            from argparse import Namespace
            args = Namespace(
                temperatures="1,6", seeds="0,1", steps=60,
                record_every=10, dt=.2, noise=.025,
                expert_rate=.25, router_rate=.25, expert_decay=.08,
                router_decay=.08, balance=.2, smoke=True,
                output=str(Path(directory)/"moe.json"),
            )
            result = run(args)
            self.assertEqual(len(result["trials"]), 4)
            self.assertTrue(Path(directory, "moe.pdf").exists())
            self.assertGreater(len(result["trials"][0]["history"]), 2)


if __name__ == "__main__":
    unittest.main()
