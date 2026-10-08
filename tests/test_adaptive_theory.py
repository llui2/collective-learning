"""Small analytical checks for the collective-performance observables."""

import unittest

import numpy as np

from collective_learning.adaptive_theory import observables, rhs


class AdaptiveTheoryTests(unittest.TestCase):
    def test_homogeneous_state_has_zero_gain_and_covariance(self):
        runs, units, components = 2, 4, 3
        drive, relaxation = 0.35, 0.25
        m_star = np.sqrt(drive - relaxation)

        m = np.full((runs, units, components), m_star)
        a = np.full_like(m, 1.0 / components)
        initial = np.concatenate((m.ravel(), a.ravel()))
        samples = np.repeat(initial[:, None], 3, axis=1)

        g, s, loss, cov = observables(
            samples, runs, units, components
        )
        target_loss = components * (1.0 - m_star) ** 2 / 2.0

        np.testing.assert_allclose(g, 0, atol=1e-14)
        np.testing.assert_allclose(s, 0, atol=1e-14)
        np.testing.assert_allclose(cov, 0, atol=1e-14)
        np.testing.assert_allclose(loss, target_loss, atol=1e-14)

        derivative = rhs(
            0.0, initial, runs, units, components,
            drive, relaxation, 0.2, 0.002, 0.5
        )
        np.testing.assert_allclose(derivative, 0, atol=1e-13)

    def test_covariance_and_collective_loss(self):
        runs, units, components = 1, 2, 2
        m = np.array([[[0.8, 0.2], [0.2, 0.8]]])
        a = np.array([[[0.9, 0.1], [0.1, 0.9]]])
        initial = np.concatenate((m.ravel(), a.ravel()))
        solution = np.repeat(initial[:, None], 2, axis=1)
        g, s, loss, cov = observables(
            solution, runs, units, components
        )

        self.assertAlmostEqual(float(loss[0]), 0.25)
        self.assertAlmostEqual(float(cov[0]), 0.24)
        self.assertGreater(float(g[0]), 0)
        self.assertGreater(float(s[0]), 0)


if __name__ == "__main__":
    unittest.main()
