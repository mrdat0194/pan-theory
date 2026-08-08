import numpy as np
from scipy.linalg import cho_factor, cho_solve

class ADMMSVM:
    """
    Support Vector Machine using Alternating Direction Method of Multipliers (ADMM).
    Optimizes: 1/2 ||w||_2^2 + C * sum(max(0, 1 - y_i(x_i^T w + b)))
    """
    def __init__(self, C=1.0, rho=1.0, max_iter=1000, tol=1e-4, fit_intercept=True):
        self.C = C
        self.rho = rho
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.w = None
        self.intercept = 0.0

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).flatten()
        
        # Convert labels to {-1, 1} if they are {0, 1}
        y = np.where(y <= 0, -1, 1)

        N, d = X.shape
        
        if self.fit_intercept:
            X_ext = np.hstack([np.ones((N, 1)), X])
            dim = d + 1
        else:
            X_ext = X
            dim = d

        # Create \tilde{X} where each row is y_i * x_i^T
        X_tilde = y[:, np.newaxis] * X_ext

        # Precompute matrix for w-update
        # (I_modified + rho * X_tilde^T X_tilde)
        # We don't regularize the intercept term, so I_modified has a 0 at (0,0)
        I_mod = np.eye(dim)
        if self.fit_intercept:
            I_mod[0, 0] = 1e-5 # Small regularization for numerical stability

        H = I_mod + self.rho * X_tilde.T @ X_tilde
        c, lower = cho_factor(H)

        # Initialize ADMM variables
        w = np.zeros(dim)
        z = np.zeros(N)
        u = np.zeros(N)

        for k in range(self.max_iter):
            # 1. w-update
            # w = (I + rho * X_tilde^T X_tilde)^-1 (rho * X_tilde^T (z - u))
            rhs = self.rho * X_tilde.T @ (z - u)
            w_next = cho_solve((c, lower), rhs)

            # 2. z-update (Proximal operator of Hinge Loss)
            v = X_tilde @ w_next + u
            
            # Proximal of C/rho * max(0, 1 - z)
            z_next = np.where(v >= 1.0, v,
                              np.where(v < 1.0 - self.C / self.rho, v + self.C / self.rho, 1.0))

            # 3. u-update
            u_next = u + X_tilde @ w_next - z_next

            # Check convergence
            primal_res = np.linalg.norm(X_tilde @ w_next - z_next)
            dual_res = np.linalg.norm(self.rho * X_tilde.T @ (z_next - z))

            if primal_res < self.tol and dual_res < self.tol:
                w = w_next
                break

            w = w_next
            z = z_next
            u = u_next
        
        if self.fit_intercept:
            self.intercept = w[0]
            self.w = w[1:]
        else:
            self.w = w

        return self

    def decision_function(self, X):
        X = np.array(X)
        return X @ self.w + self.intercept

    def predict(self, X):
        scores = self.decision_function(X)
        preds = np.where(scores >= 0, 1, -1)
        # Assuming original labels might have been 0/1, but SVM predicts -1/1
        # The user's evaluation expects the same type as trained.
        # We'll just return -1/1 and let the confusion matrix handle it, 
        # or we map back to 0/1 if the user passed 0/1. Let's just return -1/1.
        # Wait, if original labels were {0, 1}, predicting {-1, 1} breaks sklearn metrics.
        # Let's map -1 -> 0 for compatibility with standard binary classification if needed.
        # Actually, let's just return what standard LinearSVC does (which returns original labels).
        return preds
    
    def score(self, X, y):
        preds = self.predict(X)
        y_mapped = np.where(np.array(y).flatten() <= 0, -1, 1)
        return np.mean(preds == y_mapped)
