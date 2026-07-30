#!/usr/bin/env python3
"""
QILinearEstimator.py
--------------------
Phase 1: Quantum-Inspired Linear Systems Solver (Ax = b).

Based on:
 - Tang, Ewin (2019). "A quantum-inspired classical algorithm for recommendation systems."
 - Chia, Gilyen, Li, Lin, Tang, Wang (2020). "Sampling-based sublinear low-rank matrix
   arithmetic framework for dequantization."
 - XanaduAI/quantum-inspired-algorithms (linear_systems subroutine)
 - TNO-Quantum/ml.regression.linear_regression (QILinearEstimator)

Algorithm overview (FKV sketching):
  Given A ∈ R^{n×d} and b ∈ R^n, find approximate x such that Ax ≈ b.

  1. Store A in BSTMatrix for O(log n) row-norm-proportional sampling.
  2. Sample c rows of A proportional to their squared L2 norm.
  3. Scale each sampled row to form a sketch C ∈ R^{c×d}.
  4. Compute truncated SVD of C: C ≈ U Σ V^T (keep top `rank` singular values).
  5. Return approximate solution: x̂ = V Σ^{-1} U^T b.

Phase 2 (Recommendation Systems) and Phase 3 (Portfolio Optimization) — later.
"""

import math
import numpy as np
from scipy.linalg import svd as scipy_svd

import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from BST_MatrixVector import BSTMatrix, BSTVector


class QILinearEstimator:
    """
    Quantum-Inspired Linear Systems Solver via FKV row sketching.

    Approximates the least-squares solution x to Ax ≈ b by:
      - Building a BSTMatrix from A for O(log n) row-norm sampling.
      - Constructing a small sketch C ∈ R^{c×d} from c sampled rows.
      - Solving on the sketch via truncated SVD.

    Parameters
    ----------
    rank : int
        Number of singular values to retain in the truncated SVD of the sketch.
        Higher rank → more accurate, more expensive.
    c : int
        Number of rows to sample for the FKV sketch.
        Higher c → more accurate approximation.
    rng : int or None
        Random seed for reproducibility. None means random.

    Example
    -------
    >>> import numpy as np
    >>> rng = np.random.RandomState(0)
    >>> A = rng.randn(200, 10)
    >>> x_true = rng.randn(10)
    >>> b = A @ x_true + rng.randn(200) * 0.01
    >>> qi = QILinearEstimator(rank=10, c=50, rng=0)
    >>> qi.fit(A, b)
    >>> x_hat = qi.predict_x()
    >>> print(np.allclose(x_hat, x_true, atol=0.5))
    """

    def __init__(self, rank: int = 5, c: int = 20, rng=None):
        self.rank = rank
        self.c = c
        self._rng = np.random.RandomState(rng) if isinstance(rng, int) else (rng or np.random.RandomState())
        self._x_hat = None
        self._bst_matrix = None
        self._frob_norm = None
        self._n = None
        self._d = None

    # ── Public interface ───────────────────────────────────────────────────────

    def fit(self, A: np.ndarray, b: np.ndarray) -> "QILinearEstimator":
        """
        Fit the estimator by constructing the FKV sketch of A and
        computing the approximate least-squares solution.

        Parameters
        ----------
        A : np.ndarray of shape (n, d)
            Training data (design matrix).
        b : np.ndarray of shape (n,)
            Target vector.

        Returns
        -------
        self
        """
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)
        n, d = A.shape
        self._n, self._d = n, d

        # ── Step 1: Load A into BSTMatrix ─────────────────────────────────────
        bst = BSTMatrix(n, d)
        for i in range(n):
            for j in range(d):
                if A[i, j] != 0.0:
                    bst.set(i, j, A[i, j])
        self._bst_matrix = bst

        frob2 = bst.frob_norm2
        self._frob_norm = math.sqrt(frob2) if frob2 > 0 else 1.0

        # ── Step 2: FKV row sampling — sample c rows ∝ row norm squared ───────
        c = min(self.c, n)
        frob_norm = self._frob_norm
        C_rows = []
        b_sub_vals = []

        for _ in range(c):
            row_idx = bst.sample_row_norms()
            row_norm = bst.get_row_norm(row_idx)  # L2 norm of row i
            # Scaling factor: ||A||_F / (sqrt(c) * ||a_i||)
            scale = frob_norm / (math.sqrt(c) * row_norm) if row_norm > 0 else 0.0
            C_rows.append(scale * A[row_idx])
            b_sub_vals.append(scale * b[row_idx])

        C = np.vstack(C_rows)           # shape: (c, d)
        b_sub = np.array(b_sub_vals)    # shape: (c,)

        # ── Step 3: Truncated SVD of the sketch C ─────────────────────────────
        rank = min(self.rank, *C.shape)
        U, s, Vt = scipy_svd(C, full_matrices=False)

        # Keep only the top `rank` components
        U = U[:, :rank]
        s = s[:rank]
        Vt = Vt[:rank, :]



        # x̂ = V Σ^{-1} U^T b_sub
        s_inv = np.where(s > 1e-10, 1.0 / s, 0.0)
        self._x_hat = Vt.T @ (s_inv * (U.T @ b_sub))

        return self

    def predict_x(self) -> np.ndarray:
        """
        Return the estimated coefficient vector x̂.

        Returns
        -------
        np.ndarray of shape (d,)
        """
        if self._x_hat is None:
            raise RuntimeError("Call fit() before predict_x().")
        return self._x_hat.copy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for new input X using estimated coefficients.

        Parameters
        ----------
        X : np.ndarray of shape (m, d)

        Returns
        -------
        np.ndarray of shape (m,)
        """
        if self._x_hat is None:
            raise RuntimeError("Call fit() before predict().")
        return np.array(X, dtype=float) @ self._x_hat

    def __repr__(self) -> str:
        return (f"QILinearEstimator(rank={self.rank}, c={self.c})")


# ── Phase 2 placeholder ────────────────────────────────────────────────────────

class _QIRecommendationSystem:
    """
    Phase 2 (Later): Quantum-Inspired Recommendation System.

    Will use BSTMatrix for user-item matrix sampling to produce
    low-rank matrix completions in sublinear time.

    Based on:
     - Kerenidis & Prakash (2017) Quantum Recommendation Systems.
     - Tang (2019) A quantum-inspired classical algorithm for recommendation systems.

    Not yet implemented.
    """
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Phase 2 (Recommendation Systems) is not yet implemented.")


# ── Phase 3 placeholder ────────────────────────────────────────────────────────

class _QIPortfolioOptimizer:
    """
    Phase 3 (Later): Quantum-Inspired Portfolio Optimization.

    Will use the linear solver from Phase 1 to solve the Markowitz
    mean-variance portfolio optimization problem in near-linear time.

    Based on:
     - XanaduAI/quantum-inspired-algorithms portfolio subroutine.

    Not yet implemented.
    """
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Phase 3 (Portfolio Optimization) is not yet implemented.")
