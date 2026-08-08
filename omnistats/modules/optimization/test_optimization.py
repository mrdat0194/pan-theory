import unittest
import numpy as np
import sys
import os

# Add pan-theory root to python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from sklearn.preprocessing import PolynomialFeatures
from omnistats.modules.optimization.proximal import (
    prox_l2_ball,
    prox_block_l2_ball,
    adaptive_primal_dual
)
from omnistats.modules.causal.bma import fit_lipschitz_cate, run_bma



class TestOptimizationAndCATE(unittest.TestCase):

    def test_prox_l2_ball(self):
        # 1. Test vector inside the ball (should be unchanged)
        x1 = np.array([0.5, 0.2, -0.1])
        res1 = prox_l2_ball(x1, r=1.0)
        np.testing.assert_allclose(res1, x1)

        # 2. Test vector outside the ball (should be projected to boundary)
        x2 = np.array([2.0, 0.0, 0.0])
        res2 = prox_l2_ball(x2, r=1.0)
        np.testing.assert_allclose(res2, np.array([1.0, 0.0, 0.0]))

    def test_prox_block_l2_ball(self):
        # Two blocks of size 2, r=1.0
        y = np.array([0.5, 0.5, 2.0, 0.0])
        # block 1: [0.5, 0.5] (norm < 1.0) -> unchanged
        # block 2: [2.0, 0.0] (norm > 1.0) -> projected to [1.0, 0.0]
        res = prox_block_l2_ball(y, r=1.0, block_size=2)
        expected = np.array([0.5, 0.5, 1.0, 0.0])
        np.testing.assert_allclose(res, expected)

    def test_adaptive_primal_dual_l2_constrained(self):
        # Solve: min_x 0.5 * ||A x - b||^2 s.t. ||x||_2 <= 1.0
        # This is a classic primal-dual problem where:
        # K = Identity, h = indicator function of L2 ball
        np.random.seed(42)
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        b = np.array([5.0, 6.0])  # Unconstrained minimizer is around [1.8, 1.4], norm > 1.0
        
        # Unconstrained solution:
        x_unconstrained = np.linalg.solve(A, b)
        self.assertGreater(np.linalg.norm(x_unconstrained), 1.0)
        
        # Setup primal-dual
        def grad_f(x):
            return A.T @ (A @ x - b)
            
        def prox_g(x, tau):
            return x  # g(x) = 0
            
        def prox_h_conj(y, sigma):
            return y - prox_l2_ball(y, r=sigma * 1.0)
            
        # K = Identity
        K = np.eye(2)
        
        # Calculate L_f and L_K
        Lf = float(np.linalg.eigvalsh(A.T @ A)[-1])
        LK = 1.0
        
        x0 = np.zeros(2)
        y0 = np.zeros(2)
        
        x_star = adaptive_primal_dual(
            x0, y0, grad_f, prox_g, prox_h_conj, K, K.T, max_iter=5000, tol=1e-8, L_f=Lf, L_K=LK
        )
        
        # Optimal solution must lie on the boundary (norm = 1.0)
        self.assertAlmostEqual(np.linalg.norm(x_star), 1.0, places=4)
        
        # Primal-dual gradient of Lagrangian at optimal should be zero (or aligned)
        # grad f(x*) + K^T y* = 0 -> A^T(A x* - b) + y* = 0 -> y* = A^T(b - A x*)
        # Since y* is projected onto L2 ball conjugate, it should be collinear to x*
        cos_angle = np.dot(x_star, grad_f(x_star)) / (np.linalg.norm(x_star) * np.linalg.norm(grad_f(x_star)))
        # gradient of f points opposite to optimal x* step (inward normal to the ball)
        self.assertAlmostEqual(cos_angle, -1.0, places=3)

    def test_lipschitz_cate_estimator(self):
        # Generate simulated data
        np.random.seed(42)
        N = 100
        d = 2
        X_demo = np.random.uniform(-1, 1, (N, d))
        T_vec = np.random.binomial(1, 0.5, N)
        
        # True CATE function: tau(x) = 1.5 * x1 + 0.5 * x2^2 (smooth, bounded gradient)
        tau_true = 1.5 * X_demo[:, 0] + 0.5 * X_demo[:, 1]**2
        Y_outcome = 2.0 + 1.0 * X_demo[:, 0] + T_vec * tau_true + np.random.normal(0, 0.1, N)
        
        theta, cate_pred, poly = fit_lipschitz_cate(
            X_demo, Y_outcome, T_vec, lambda_reg=0.1, L_bound=1.5
        )
        
        # Verify output formats and shapes
        self.assertEqual(len(cate_pred), N)
        self.assertEqual(len(theta), poly.fit_transform(X_demo).shape[1])
        
        # Estimated marginalized ATT should be close to actual average treatment effect
        self.assertAlmostEqual(cate_pred.mean(), tau_true.mean(), delta=0.5)


if __name__ == '__main__':
    unittest.main()
