"""
Bayesian/test_waiting_time.py
------------------------------
Unit tests for the unified waiting-time distributions module.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import unittest
import numpy as np
from waiting_time_unification import (
    GeometricWaiting,
    ExponentialWaiting,
    ErlangWaiting,
    DiracDeltaWaiting,
    PhaseTypeWaiting
)

class TestWaitingTimeDistributions(unittest.TestCase):

    def test_geometric(self):
        p = 0.2
        geom = GeometricWaiting(p)
        
        # Test theoretical moments
        self.assertAlmostEqual(geom.theoretical_mean(), 5.0)
        self.assertAlmostEqual(geom.theoretical_variance(), 20.0)
        
        # Test PMF / CDF values
        self.assertAlmostEqual(geom.pdf(1), 0.2)
        self.assertAlmostEqual(geom.pdf(2), 0.16)
        self.assertAlmostEqual(geom.cdf(1), 0.2)
        self.assertAlmostEqual(geom.cdf(2), 0.36)
        
        # Test sampling size
        samples = geom.sample(100)
        self.assertEqual(len(samples), 100)
        self.assertTrue(np.all(samples >= 1))

    def test_exponential(self):
        rate = 2.0
        expo = ExponentialWaiting(rate)
        
        # Test moments
        self.assertAlmostEqual(expo.theoretical_mean(), 0.5)
        self.assertAlmostEqual(expo.theoretical_variance(), 0.25)
        
        # Test PDF / CDF values
        self.assertAlmostEqual(expo.pdf(0.0), 2.0)
        self.assertAlmostEqual(expo.cdf(0.0), 0.0)
        self.assertAlmostEqual(expo.cdf(1.0), 1.0 - np.exp(-2.0))
        
        # Test sampling
        samples = expo.sample(100)
        self.assertEqual(len(samples), 100)
        self.assertTrue(np.all(samples >= 0.0))

    def test_erlang(self):
        k = 4
        rate = 2.0
        erl = ErlangWaiting(k, rate)
        
        # Test moments (mean = k/rate = 2.0, var = k/rate^2 = 1.0)
        self.assertAlmostEqual(erl.theoretical_mean(), 2.0)
        self.assertAlmostEqual(erl.theoretical_variance(), 1.0)
        
        # Test sampling
        samples = erl.sample(100)
        self.assertEqual(len(samples), 100)
        self.assertTrue(np.all(samples >= 0.0))

    def test_dirac_delta(self):
        t0 = 3.5
        sigma = 0.01
        dd = DiracDeltaWaiting(t0, sigma_dirac=sigma)
        
        # Test moments
        self.assertAlmostEqual(dd.theoretical_mean(), t0)
        self.assertAlmostEqual(dd.theoretical_variance(), sigma ** 2)
        
        # Test PDF/CDF limits
        # At exactly t = t0, PDF should be high (standard normal height scaled by 1/sigma)
        self.assertAlmostEqual(dd.pdf(t0), 1.0 / (sigma * np.sqrt(2 * np.pi)))
        # CDF at t0 should be exactly 0.5
        self.assertAlmostEqual(dd.cdf(t0), 0.5)

    def test_phase_type(self):
        # Represent Erlang(k=2, rate=3.0) as Phase-Type
        k = 2
        rate = 3.0
        alpha = np.array([1.0, 0.0])
        S = np.array([[-rate, rate],
                      [0.0, -rate]])
                      
        ph = PhaseTypeWaiting(alpha, S)
        erl = ErlangWaiting(k, rate)
        
        # Verify theoretical moments match Erlang
        self.assertAlmostEqual(ph.theoretical_mean(), erl.theoretical_mean())
        self.assertAlmostEqual(ph.theoretical_variance(), erl.theoretical_variance())
        
        # Test PDF evaluation matches Erlang PDF at sample points
        test_points = np.array([0.1, 0.5, 1.0, 2.0])
        np.testing.assert_allclose(ph.pdf(test_points), erl.pdf(test_points), rtol=1e-5)
        np.testing.assert_allclose(ph.cdf(test_points), erl.cdf(test_points), rtol=1e-5)
        
        # Test sampling
        samples = ph.sample(50)
        self.assertEqual(len(samples), 50)
        self.assertTrue(np.all(samples >= 0.0))

if __name__ == '__main__':
    unittest.main()
