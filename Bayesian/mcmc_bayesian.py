#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mcmc_bayesian.py

A comparative script to demonstrate the differences in implementation 
between Importance Sampling (IS) and Markov Chain Monte Carlo (MCMC).

This script compares estimating the integral (and sampling from) a 
distribution with an integrable singularity at x=0, specifically f(x) = x^gamma on (0, 1].
"""

import numpy as np
import scipy.special as special
import time

def importance_sampling(gamma, zeta, n_samples, seed=42):
    """
    Importance Sampling using proposal g(x) = (1+zeta) * x^zeta
    """
    rng = np.random.default_rng(seed)
    
    # 1. Sample from proposal g(x) using inverse CDF
    # u = x^(1+zeta)  =>  x = u^(1/(1+zeta))
    u = rng.random(n_samples)
    u = np.maximum(u, 1e-15) # avoid pure zero
    samples = u ** (1.0 / (1.0 + zeta))
    
    # 2. Compute weights: w(x) = f(x) / g(x) = (x^gamma) / ((1+zeta) * x^zeta)
    # log weights for numerical stability
    log_f = gamma * np.log(samples)
    log_g = np.log(1.0 + zeta) + zeta * np.log(samples)
    log_weights = log_f - log_g
    
    # Normalize weights
    max_log_w = np.max(log_weights)
    weights = np.exp(log_weights - max_log_w)
    normalized_weights = weights / np.sum(weights)
    
    # Estimate integral
    estimate = np.mean(np.exp(log_weights))
    
    # Compute Effective Sample Size (ESS)
    ess = 1.0 / np.sum(normalized_weights ** 2)
    
    return estimate, ess, samples

def mcmc_metropolis(gamma, delta, n_samples, seed=42):
    """
    MCMC (Metropolis algorithm) targeting f(x) = x^gamma
    """
    rng = np.random.default_rng(seed)
    
    samples = np.zeros(n_samples)
    x = 1.0 # start far from the singularity
    accepted = 0
    
    for i in range(n_samples):
        # Propose new state
        x_new = x + rng.uniform(-delta, delta)
        
        # Check domain boundaries (0, 1]
        if 0 < x_new <= 1.0:
            # Acceptance probability ratio: f(x_new) / f(x)
            p_accept = (x_new / x) ** gamma
            if rng.random() < p_accept:
                x = x_new
                accepted += 1
                
        samples[i] = x
        
    # Estimate integral
    # In MCMC, the samples are drawn *from* the target distribution (proportional to f(x)).
    # We cannot directly compute the normalizing constant (integral) just by averaging samples
    # like we do in Importance Sampling. This is a key limitation of standard MCMC!
    # Instead, we will just return the acceptance rate and the samples.
    
    acceptance_rate = accepted / n_samples
    return acceptance_rate, samples

def mcmc_metropolis_hastings(gamma, zeta, n_samples, seed=42):
    """
    Evolved MCMC (Metropolis-Hastings Independence Sampler).
    Uses the Importance Sampling proposal g(x) = (1+zeta) * x^zeta to generate MCMC steps.
    This solves the local random-walk trapping of standard MCMC near singularities.
    """
    rng = np.random.default_rng(seed)
    
    samples = np.zeros(n_samples)
    x = 1.0
    accepted = 0
    
    for i in range(n_samples):
        # Propose globally from g(x) using inverse CDF
        u = max(rng.random(), 1e-15)
        x_new = u ** (1.0 / (1.0 + zeta))
        
        # Metropolis-Hastings acceptance ratio: (f(x_new) / g(x_new)) / (f(x) / g(x))
        # f(x) / g(x) is proportional to x^(gamma - zeta)
        w_new = x_new ** (gamma - zeta)
        w_old = x ** (gamma - zeta)
        
        p_accept = w_new / w_old if w_old > 0 else 1.0
        
        if rng.random() < p_accept:
            x = x_new
            accepted += 1
            
        samples[i] = x
        
    return accepted / n_samples, samples

def mcmc_mirror_hmc(gamma, L_steps, step_size, n_samples, seed=42):
    """
    Algorithm 4: Mirror HMC (Reparameterized HMC).
    Uses a logit transform y = ln(x / (1-x)) to map bounded x in (0, 1] 
    to unbounded y in R. This perfectly demonstrates the 'Reparameterization Trick'
    used in Deep Learning/VAEs to allow smooth gradient descent over complex bounded variables.
    """
    rng = np.random.default_rng(seed)
    samples_x = np.zeros(n_samples)
    
    # Initialize in unconstrained space
    x = 0.5
    y = np.log(x / (1.0 - x))
    accepted = 0
    
    def U(y_val):
        x_val = 1.0 / (1.0 + np.exp(-np.clip(y_val, -500, 500)))
        return -((gamma + 1.0) * np.log(max(x_val, 1e-15)) + np.log(max(1.0 - x_val, 1e-15)))
        
    def grad_U(y_val):
        x_val = 1.0 / (1.0 + np.exp(-np.clip(y_val, -500, 500)))
        return x_val - (gamma + 1.0) * (1.0 - x_val)
        
    for i in range(n_samples):
        p = rng.normal(0, 1)
        current_p = p
        current_y = y
        
        y_new = y
        p_new = p
        
        p_new = p_new - 0.5 * step_size * grad_U(y_new)
        for step in range(L_steps):
            y_new = y_new + step_size * p_new
            if step != L_steps - 1:
                p_new = p_new - step_size * grad_U(y_new)
        p_new = p_new - 0.5 * step_size * grad_U(y_new)
        
        current_U = U(current_y)
        current_K = 0.5 * current_p**2
        proposed_U = U(y_new)
        proposed_K = 0.5 * p_new**2
        
        if not np.isnan(proposed_U) and rng.random() < np.exp(current_U + current_K - proposed_U - proposed_K):
            y = y_new
            accepted += 1
            
        x = 1.0 / (1.0 + np.exp(-np.clip(y, -500, 500)))
        samples_x[i] = x
        
    return accepted / n_samples, samples_x

def mcmc_tempered_hmc_gmm(T, L_steps, step_size, n_samples, seed=42):
    """
    Algorithm 5: Tempered HMC on a bimodal Gaussian Mixture Model (GMM).
    If T=1.0, it's standard HMC (which gets trapped in one mode).
    If T > 1.0, it's Tempered HMC, flattening the energy landscape so 
    Hamiltonian trajectories can cross the energy valley.
    """
    rng = np.random.default_rng(seed)
    samples = np.zeros(n_samples)
    x = -8.0 # Start trapped in the left mode
    accepted = 0
    
    def U(x_val):
        N1 = np.exp(-0.5 * (x_val + 8.0)**2)
        N2 = np.exp(-0.5 * (x_val - 8.0)**2)
        return -(1.0 / T) * np.log(0.5 * N1 + 0.5 * N2 + 1e-15)
        
    def grad_U(x_val):
        N1 = np.exp(-0.5 * (x_val + 8.0)**2)
        N2 = np.exp(-0.5 * (x_val - 8.0)**2)
        denom = 0.5 * N1 + 0.5 * N2 + 1e-15
        num = 0.5 * N1 * -(x_val + 8.0) + 0.5 * N2 * -(x_val - 8.0)
        return -(1.0 / T) * (num / denom)
        
    for i in range(n_samples):
        p = rng.normal(0, 1)
        current_p = p
        current_x = x
        
        x_new = x
        p_new = p
        
        p_new = p_new - 0.5 * step_size * grad_U(x_new)
        for step in range(L_steps):
            x_new = x_new + step_size * p_new
            if step != L_steps - 1:
                p_new = p_new - step_size * grad_U(x_new)
        p_new = p_new - 0.5 * step_size * grad_U(x_new)
        
        current_U = U(current_x)
        current_K = 0.5 * current_p**2
        proposed_U = U(x_new)
        proposed_K = 0.5 * p_new**2
        
        if not np.isnan(proposed_U) and rng.random() < np.exp(current_U + current_K - proposed_U - proposed_K):
            x = x_new
            accepted += 1
            
        samples[i] = x
        
    return accepted / n_samples, samples




if __name__ == "__main__":
    print("=" * 70)
    print("COMPARISON: Importance Sampling (IS) vs MCMC (Metropolis)")
    print("Target Distribution: f(x) = x^gamma on (0, 1]")
    print("=" * 70)
    
    gamma = -0.8
    n_samples = 50000
    true_integral = 1.0 / (gamma + 1.0)
    
    print(f"\nTarget: gamma = {gamma}")
    print(f"True Analytical Integral: {true_integral:.5f}")
    
    # --- 1. Importance Sampling ---
    print("\n--- 1. Importance Sampling ---")
    zeta = -0.7  # Proposal parameter
    print(f"Proposal: g(x) = (1+zeta)*x^zeta with zeta = {zeta}")
    
    start = time.time()
    is_est, ess, is_samples = importance_sampling(gamma, zeta, n_samples)
    is_time = time.time() - start
    
    print(f"Estimated Integral: {is_est:.5f}")
    print(f"Effective Sample Size (ESS): {ess:.2f} / {n_samples}")
    print(f"Time Taken: {is_time:.4f} seconds")
    print("Pros: Independent samples (no trapping), calculates normalizing constant directly, highly parallelizable.")
    print("Cons: Requires designing a good proposal distribution. If zeta > 2*gamma+1, variance becomes infinite. Fails in high dimensions.")
    
    # --- 2. Markov Chain Monte Carlo (MCMC) ---
    print("\n--- 2. MCMC (Metropolis) ---")
    delta = 0.05
    print(f"Proposal: Uniform local step with delta = {delta}")
    
    start = time.time()
    acc_rate, mcmc_samples = mcmc_metropolis(gamma, delta, n_samples)
    mcmc_time = time.time() - start
    
    print(f"Acceptance Rate: {acc_rate:.2%}")
    print(f"Time Taken: {mcmc_time:.4f} seconds")
    
    # Autocorrelation (lag-1)
    mean_mcmc = np.mean(mcmc_samples)
    var_mcmc = np.var(mcmc_samples)
    autocorr = np.sum((mcmc_samples[:-1] - mean_mcmc) * (mcmc_samples[1:] - mean_mcmc)) / ( (n_samples - 1) * var_mcmc )
    print(f"Lag-1 Autocorrelation: {autocorr:.4f}")
    
    print("Pros: Easy to implement, explores high-dimensional spaces well by finding the typical set.")
    print("Cons: Samples are correlated (high autocorrelation), requires burn-in, CANNOT easily compute the normalizing constant (integral).")
    print(f"      (Notice how we couldn't print an 'Estimated Integral' for MCMC here!)")
    
    # --- 3. Evolved MCMC (Metropolis-Hastings Independence Sampler) ---
    print("\n--- 3. Evolved MCMC (Metropolis-Hastings) ---")
    print(f"Proposal: Global informed step from g(x) = (1+zeta)*x^zeta with zeta = {zeta}")
    
    start = time.time()
    mh_acc_rate, mh_samples = mcmc_metropolis_hastings(gamma, zeta, n_samples)
    mh_time = time.time() - start
    
    print(f"Acceptance Rate: {mh_acc_rate:.2%}")
    print(f"Time Taken: {mh_time:.4f} seconds")
    
    mean_mh = np.mean(mh_samples)
    var_mh = np.var(mh_samples)
    autocorr_mh = np.sum((mh_samples[:-1] - mean_mh) * (mh_samples[1:] - mean_mh)) / ( (n_samples - 1) * var_mh )
    print(f"Lag-1 Autocorrelation: {autocorr_mh:.4f}")
    
    print("Pros: Solves local trapping by proposing globally. Vastly reduces autocorrelation.")
    print("Cons: Still cannot compute the normalizing constant. Relies on a good global proposal.")

    # --- 4. State of the Art: Mirror HMC (Reparameterization Trick) ---
    print("\n--- 4. SOTA: Mirror HMC (Reparameterized Hamiltonian) ---")
    print("Proposal: Gradient descent via leapfrog integration on a logit-transformed unbounded space.")
    start = time.time()
    hmc_acc, hmc_samples = mcmc_mirror_hmc(gamma, L_steps=10, step_size=0.1, n_samples=n_samples)
    hmc_time = time.time() - start
    
    print(f"Acceptance Rate: {hmc_acc:.2%}")
    print(f"Time Taken: {hmc_time:.4f} seconds")
    mean_hmc = np.mean(hmc_samples)
    var_hmc = np.var(hmc_samples)
    autocorr_hmc = np.sum((hmc_samples[:-1] - mean_hmc) * (hmc_samples[1:] - mean_hmc)) / ( (n_samples - 1) * var_hmc )
    print(f"Lag-1 Autocorrelation: {autocorr_hmc:.4f}")
    print("Pros: Uses gradients to deterministically glide toward high-probability areas. Reparameterization elegantly handles boundaries.")

    # --- 5. State of the Art: Tempered HMC for Multimodal GMM ---
    print("\n" + "=" * 70)
    print("--- 5. SOTA: Tempered HMC for Multimodal GMM ---")
    print("Target: Bimodal GMM at x=-8 and x=+8.")
    
    # Standard HMC gets trapped
    acc_std, sam_std = mcmc_tempered_hmc_gmm(T=1.0, L_steps=15, step_size=0.25, n_samples=n_samples)
    modes_std = np.unique(np.sign(sam_std))
    print(f"[T=1.0 Standard HMC] Acc: {acc_std:.2%}. Modes explored: {'Both' if len(modes_std)>1 else 'Trapped in One Mode!'}")
    
    # Tempered HMC successfully hops
    acc_tmp, sam_tmp = mcmc_tempered_hmc_gmm(T=5.0, L_steps=15, step_size=0.25, n_samples=n_samples)
    modes_tmp = np.unique(np.sign(sam_tmp))
    print(f"[T=5.0 Tempered HMC] Acc: {acc_tmp:.2%}. Modes explored: {'Both!' if len(modes_tmp)>1 else 'Trapped.'}")
    print("Pros: Tempering flattens energy barriers, allowing HMC's physical trajectories to cross impossible valleys.")

    print("\n" + "=" * 70)
    print("EVOLUTION OF MCMC IMPLEMENTATION IN THIS REPOSITORY:")
    print("1. Standard Local MCMC: Gets trapped near boundaries and valleys (critical slowing down).")
    print("2. Informed MH (Independence Sampler): Global proposals manually crafted to escape traps.")
    print("3. Path Integral Monte Carlo (PIMC): Samples continuous trajectories to handle quantum mechanics.")
    print("4. Cluster Algorithms (Wolff): Flips macroscopic structures simultaneously.")
    print("5. SOTA - Mirror HMC: Uses Reparameterization Tricks to transform bounded parameters into smooth spaces for gradient descent.")
    print("6. SOTA - Tempered HMC: Tempers gradients to effortlessly bridge multimodal distributions.")
    print("7. Sequential Importance Sampling: Variable-dimension step-by-step state building.")
    print("=" * 70)

