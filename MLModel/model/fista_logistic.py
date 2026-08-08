import numpy as np

class FISTALogisticRegression:
    """
    Sparse Logistic Regression using the Fast Iterative Shrinkage-Thresholding Algorithm (FISTA).
    Optimizes: 1/N * sum(logistic_loss(w; X, y)) + lambda_reg * ||w||_1
    """
    def __init__(self, lambda_reg=0.1, max_iter=1000, tol=1e-5, fit_intercept=True):
        self.lambda_reg = lambda_reg
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.w = None
        self.intercept = 0.0

    def _sigmoid(self, z):
        # Clip z to avoid overflow in exp
        z = np.clip(z, -250, 250)
        return 1.0 / (1.0 + np.exp(-z))

    def _soft_threshold(self, x, lmbda):
        return np.sign(x) * np.maximum(np.abs(x) - lmbda, 0.0)

    def fit(self, X, y):
        """
        Fit the model using FISTA.
        y is expected to be {0, 1}.
        """
        X = np.array(X)
        y = np.array(y).flatten()
        
        N, d = X.shape
        
        # Add intercept term if needed
        if self.fit_intercept:
            X_ext = np.hstack([np.ones((N, 1)), X])
            dim = d + 1
        else:
            X_ext = X
            dim = d

        # Compute Lipschitz constant L = ||X_ext||_2^2 / (4N)
        # We can approximate ||X_ext||_2^2 using the max eigenvalue of X_ext^T X_ext
        XtX = X_ext.T @ X_ext
        L_X = np.linalg.eigvalsh(XtX)[-1]
        L = L_X / (4.0 * N)
        
        alpha = 1.0 / L if L > 1e-8 else 1.0

        # Initialize FISTA variables
        w_prev = np.zeros(dim)
        y_fista = np.zeros(dim)
        t_prev = 1.0

        for k in range(self.max_iter):
            # Gradient of logistic loss at y_fista
            z = X_ext @ y_fista
            probs = self._sigmoid(z)
            grad = (X_ext.T @ (probs - y)) / N

            # Gradient descent step
            w_unreg = y_fista - alpha * grad

            # Proximal step (soft thresholding on weights, not intercept)
            w_next = np.copy(w_unreg)
            if self.fit_intercept:
                # Do not penalize intercept (index 0)
                w_next[1:] = self._soft_threshold(w_unreg[1:], alpha * self.lambda_reg)
            else:
                w_next = self._soft_threshold(w_unreg, alpha * self.lambda_reg)

            # Check convergence
            if np.linalg.norm(w_next - w_prev) < self.tol:
                self.w_ext = w_next
                break

            # Nesterov momentum update
            t_next = (1.0 + np.sqrt(1.0 + 4.0 * t_prev**2)) / 2.0
            momentum = (t_prev - 1.0) / t_next
            y_fista = w_next + momentum * (w_next - w_prev)

            # Prepare next iteration
            w_prev = w_next
            t_prev = t_next
            
        else:
            self.w_ext = w_prev

        if self.fit_intercept:
            self.intercept = self.w_ext[0]
            self.w = self.w_ext[1:]
        else:
            self.w = self.w_ext

        return self

    def predict_proba(self, X):
        X = np.array(X)
        z = X @ self.w + self.intercept
        probs = self._sigmoid(z)
        return np.vstack([1 - probs, probs]).T

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)
    
    def score(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == np.array(y).flatten())
