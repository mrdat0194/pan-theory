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


def prox_l2_ball(x: np.ndarray, r: float) -> np.ndarray:
    """
    Projection onto the L2-ball: ||x||_2 <= r.
    This corresponds to the proximal operator of the indicator function of the L2-ball.
    """
    norm = np.linalg.norm(x)
    if norm <= r:
        return x
    return x * (r / (norm + 1e-15))


def prox_block_l2_ball(y: np.ndarray, r: float, block_size: int) -> np.ndarray:
    """
    Projection of a stacked vector y onto the product of L2-balls, where each block
    of size block_size is restricted to have L2-norm <= r.
    """
    y_reshaped = y.reshape(-1, block_size)
    norms = np.linalg.norm(y_reshaped, axis=1, keepdims=True)
    scale = np.minimum(1.0, r / (norms + 1e-15))
    return (y_reshaped * scale).flatten()


def adaptive_primal_dual(x0, y0, grad_f, prox_g, prox_h_conj, K, K_transpose,
                         max_iter=1000, tol=1e-5, L_f=1.0, L_K=1.0, adaptive=True):
    """
    Adaptive Primal-Dual Hybrid Gradient (APDHG) method with Goldstein residual balancing.
    Solves:
        min_x f(x) + g(x) + h(K x)
    where f is smooth, g is prox-friendly, and h is prox-friendly.
    """
    x = np.copy(x0)
    y = np.copy(y0)
    
    K_op = K if callable(K) else (lambda v: K @ v)
    K_t_op = K_transpose if callable(K_transpose) else (lambda v: K_transpose @ v)
    
    # 1. Compute initial steps satisfying the Condat-Vu convergence condition
    # tau * (L_f/2 + sigma * L_K) < 1
    tau = 0.99 / L_f if L_f > 1e-5 else 1.0
    sigma = 0.49 / (tau * L_K) if L_K > 1e-5 else 1.0
    
    alpha = 0.5
    c_ratio = 10.0 # balancing factor
    
    for k in range(max_iter):
        x_old = np.copy(x)
        y_old = np.copy(y)
        
        # Primal step
        x = prox_g(x_old - tau * grad_f(x_old) - tau * K_t_op(y_old), tau)
        
        # Extrapolation
        x_bar = 2 * x - x_old
        
        # Dual step
        y = prox_h_conj(y_old + sigma * K_op(x_bar), sigma)
        
        # Convergence check
        dx = x - x_old
        dy = y - y_old
        norm_dx = np.linalg.norm(dx)
        norm_dy = np.linalg.norm(dy)
        
        if norm_dx < tol and norm_dy < tol:
            break
            
        # Adapt step sizes (Goldstein et al. 2013 residual balancing)
        if adaptive and norm_dx > 1e-15 and norm_dy > 1e-15:
            r_p = norm_dx / tau
            r_d = norm_dy / sigma
            
            if r_p > c_ratio * r_d:
                # decrease tau, increase sigma (keeps tau*sigma constant)
                gamma = 1.0 - alpha
                tau *= gamma
                sigma /= gamma
            elif r_d > c_ratio * r_p:
                # increase tau, decrease sigma (keeps tau*sigma constant)
                gamma = 1.0 - alpha
                tau /= gamma
                sigma *= gamma
                
            # decay alpha to guarantee convergence
            alpha *= 0.95
            
    return x


