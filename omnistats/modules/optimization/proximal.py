"""
OmniStats Optimization Module: Adaptive Proximal Algorithms
Implements AdPG based on arXiv:2301.04431 for locally Lipschitz continuous gradients.
"""
import numpy as np

def prox_l1(x: np.ndarray, gamma: float) -> np.ndarray:
    """Proximal operator for L1 norm: g(x) = ||x||_1"""
    return np.sign(x) * np.maximum(np.abs(x) - gamma, 0)

def prox_simplex(x: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Proximal operator for the probability simplex (projection onto the simplex).
    gamma is ignored because projection is independent of step size for indicator functions.
    Algorithm based on Duchi et al. (2008).
    """
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    
    n_features, n_samples = x.shape
    projected = np.zeros_like(x)
    
    for i in range(n_samples):
        v = x[:, i]
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - 1.0
        ind = np.arange(n_features) + 1
        cond = u - cssv / ind > 0
        rho = ind[cond][-1]
        theta = cssv[cond][-1] / float(rho)
        projected[:, i] = np.maximum(v - theta, 0)
        
    return projected.flatten() if x.shape[1] == 1 else projected

def adaptive_proximal_gradient(x0, grad_f, f, prox_g, max_iter=1000, tol=1e-5, L0=1.0, eta=2.0):
    """
    Adaptive Proximal Gradient Algorithm (AdPG).
    Handles locally Lipschitz continuous gradients using a backtracking line search.
    
    Args:
        x0: Initial guess (np.ndarray)
        grad_f: Function that computes the gradient of f(x)
        f: Function that computes the value of f(x)
        prox_g: Function that computes the proximal operator of g(x) with step size gamma
        max_iter: Maximum number of iterations
        tol: Tolerance for stopping criterion (norm of step)
        L0: Initial guess for the Lipschitz constant
        eta: Backtracking multiplier (> 1)
        
    Returns:
        Optimal x (np.ndarray)
    """
    x = np.copy(x0)
    L = L0
    
    for k in range(max_iter):
        g_x = grad_f(x)
        fx = f(x)
        
        while True:
            gamma = 1.0 / L
            x_next = prox_g(x - gamma * g_x, gamma)
            
            # Check local descent / Lipschitz condition
            diff = x_next - x
            f_next = f(x_next)
            
            # Quadratic upper bound condition
            quad_bound = fx + np.vdot(g_x, diff) + (L / 2.0) * np.sum(diff**2)
            
            if f_next <= quad_bound + 1e-12:
                break
            
            # Increase Lipschitz estimate (decrease step size)
            L *= eta
            
        # Update
        step_norm = np.linalg.norm(x_next - x)
        x = x_next
        
        if step_norm < tol:
            break
            
        # Optionally decrease L slightly for the next iteration to be adaptive
        L = max(L0, L / 1.2)
        
    return x
