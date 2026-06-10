#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
importance_sampling_bayesian.py

A modular educational framework demonstrating:
1. Base Importance Sampler interface.
2. Toy Importance Sampler (1D integral matching Algorithm 1.30).
3. Stick-Breaking Importance Sampler (Explicit Dirichlet Process).
4. Chinese Restaurant/Remainder Process (CRP) Sequential Importance Sampler.
5. Indian Buffet Process (IBP) Sequential Matrix Sampler.
"""

import os
from abc import ABC, abstractmethod
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# =====================================================================
# 1. Base Importance Sampler Class
# =====================================================================
class BaseImportanceSampler(ABC):
    """
    Abstract base class for Importance Sampling.
    Provides standard infrastructure for drawing samples, calculating weights,
    normalizing weights, and computing diagnostics (like Effective Sample Size).
    """
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def sample_proposal(self):
        """
        Draw a single sample from the proposal distribution g(x).
        Returns:
            sample: The proposed state/object.
        """
        pass

    @abstractmethod
    def log_proposal_prob(self, sample):
        """
        Evaluate log probability density of the proposal distribution ln g(sample).
        """
        pass

    @abstractmethod
    def log_target_prob(self, sample):
        """
        Evaluate log density of the unnormalized target distribution ln f(sample).
        """
        pass

    def run(self, n_samples):
        """
        Execute importance sampling.
        Returns:
            samples: list of drawn samples.
            normalized_weights: numpy array of normalized weights.
            ess: Effective Sample Size.
            log_weights: raw log weights (unnormalized).
        """
        samples = []
        log_weights = np.zeros(n_samples)

        for i in range(n_samples):
            sample = self.sample_proposal()
            samples.append(sample)
            
            ln_g = self.log_proposal_prob(sample)
            ln_f = self.log_target_prob(sample)
            log_weights[i] = ln_f - ln_g

        # Normalize weights safely using the log-sum-exp trick to prevent underflow/overflow
        max_log_w = np.max(log_weights)
        weights = np.exp(log_weights - max_log_w)
        sum_weights = np.sum(weights)
        normalized_weights = weights / sum_weights

        # Compute Effective Sample Size (ESS)
        # ESS = 1 / sum(w_i^2)
        ess = 1.0 / np.sum(normalized_weights ** 2)

        return samples, normalized_weights, ess, log_weights


# =====================================================================
# 2. Toy Importance Sampler (Algorithm 1.30 analog)
# =====================================================================
class ToyImportanceSampler(BaseImportanceSampler):
    """
    Estimates the integral of x^gamma over (0, 1] using a proposal distribution x^zeta.
    Re-creates the logic of Algorithm 1.30 within our modular framework.
    """
    def __init__(self, gamma, zeta, seed=None):
        super().__init__(seed)
        self.gamma = gamma
        self.zeta = zeta

    def sample_proposal(self):
        # Sample from g(x) = (1 + zeta) * x^zeta via inverse CDF
        u = self.rng.random()
        # Avoid u = 0 to prevent division by zero or log(0) issues
        u = max(u, 1e-15)
        return u ** (1.0 / (1.0 + self.zeta))

    def log_proposal_prob(self, x):
        # ln g(x) = ln(1 + zeta) + zeta * ln(x)
        return np.log(1.0 + self.zeta) + self.zeta * np.log(x)

    def log_target_prob(self, x):
        # ln f(x) = gamma * ln(x)
        return self.gamma * np.log(x)


# =====================================================================
# 3. Stick-Breaking Importance Sampler (Explicit DP mixture)
# =====================================================================
class StickBreakingImportanceSampler(BaseImportanceSampler):
    """
    Importance sampler for Dirichlet Process (DP) mixture model weights.
    We propose weights pi from the Stick-Breaking prior and data partitions z,
    and compute weights based on the data likelihood.
    """
    def __init__(self, data, alpha, K_trunc=10, seed=None):
        super().__init__(seed)
        self.data = data
        self.alpha = alpha
        self.K = K_trunc
        # Let clusters have standard Normal priors for their means: G_0 = N(0, 3^2)
        self.mu_prior_std = 3.0
        self.likelihood_std = 0.5

    def sample_proposal(self):
        # 1. Propose beta parameters from Beta(1, alpha) prior
        betas = self.rng.beta(1.0, self.alpha, size=self.K - 1)
        
        # 2. Build stick-breaking weights pi
        pi = np.zeros(self.K)
        rem = 1.0
        for k in range(self.K - 1):
            pi[k] = betas[k] * rem
            rem *= (1.0 - betas[k])
        pi[-1] = rem  # the rest of the stick
        
        # 3. Propose cluster centers theta_k from G_0 = N(0, mu_prior_std^2)
        thetas = self.rng.normal(0, self.mu_prior_std, size=self.K)
        
        # 4. Propose cluster assignments z for each data point
        z = self.rng.choice(self.K, size=len(self.data), p=pi)
        
        return {"pi": pi, "thetas": thetas, "z": z}

    def log_proposal_prob(self, sample):
        # Since we sample directly from the prior, the proposal probability is the prior.
        # This makes the importance weights reduce to the likelihood of the data.
        # However, to be mathematically explicit, we evaluate the log prior.
        pi = sample["pi"]
        thetas = sample["thetas"]
        z = sample["z"]
        
        # Log prior of thetas
        log_prior_thetas = np.sum(stats.norm.logpdf(thetas, 0, self.mu_prior_std))
        
        # Log prior of z given pi
        log_prior_z = np.sum(np.log(pi[z] + 1e-15))
        
        # Log prior of pi (Beta parts)
        # Note: beta_k = pi_k / (1 - sum_{j<k} pi_j)
        log_prior_pi = 0
        rem = 1.0
        for k in range(self.K - 1):
            beta_k = pi[k] / rem
            beta_k = clip_val(beta_k)
            log_prior_pi += stats.beta.logpdf(beta_k, 1.0, self.alpha)
            rem *= (1.0 - beta_k)
            if rem < 1e-15:
                break
                
        return log_prior_thetas + log_prior_z + log_prior_pi

    def log_target_prob(self, sample):
        # Target = Prior * Likelihood
        log_prior = self.log_proposal_prob(sample)
        
        # Likelihood of data given assignments z and locations thetas
        thetas = sample["thetas"]
        z = sample["z"]
        log_like = np.sum(stats.norm.logpdf(self.data, thetas[z], self.likelihood_std))
        
        return log_prior + log_like


# =====================================================================
# 4. Chinese Restaurant Process (CRP) Sequential Importance Sampler
# =====================================================================
class CRPImportanceSampler:
    """
    Implements Sequential Importance Sampling (SIS) for the Chinese Restaurant Process.
    Since CRP is sequential, we assign customer clusters one-by-one.
    We compare a "Blind" proposal (prior) to a "Likelihood-Informed" proposal.
    """
    def __init__(self, data, alpha, likelihood_std=0.5, seed=None):
        self.data = data
        self.alpha = alpha
        self.likelihood_std = likelihood_std
        self.rng = np.random.default_rng(seed)
        self.mu_prior_std = 3.0

    def run_sis(self, n_samples, informed=True):
        """
        Runs Sequential Importance Sampling.
        Returns:
            partitions: List of partition arrays.
            weights: Normalized weights.
        """
        N = len(self.data)
        partitions = []
        log_weights = np.zeros(n_samples)

        for s in range(n_samples):
            # z[i] stores cluster assignment of customer i
            z = []
            # n_c maps cluster label to count of customers
            counts = {}
            # track parameters/means for active clusters
            cluster_means = {}
            
            ln_g_total = 0.0
            ln_f_total = 0.0

            for i in range(N):
                x_i = self.data[i]
                
                # 1. Compute prior CRP probabilities
                # Table c chosen with prob n_c / (i + alpha)
                # New table chosen with prob alpha / (i + alpha)
                active_clusters = list(counts.keys())
                probs = []
                for c in active_clusters:
                    probs.append(counts[c] / (i + self.alpha))
                probs.append(self.alpha / (i + self.alpha))
                probs = np.array(probs)

                # 2. Compute likelihood of x_i under each cluster option
                likelihoods = []
                for c in active_clusters:
                    # Likelihood under cluster mean
                    mu_c = cluster_means[c]
                    likelihoods.append(stats.norm.pdf(x_i, mu_c, self.likelihood_std))
                # Likelihood under new cluster option (marginalize over G_0 base distribution)
                # Int N(x_i | mu, sig^2) * N(mu | 0, mu_prior_std^2) dmu = N(x_i | 0, sig^2 + mu_prior_std^2)
                marginal_std = np.sqrt(self.likelihood_std**2 + self.mu_prior_std**2)
                likelihoods.append(stats.norm.pdf(x_i, 0, marginal_std))
                likelihoods = np.array(likelihoods)

                if informed:
                    # Proposal incorporates data likelihood
                    proposal_probs = probs * likelihoods
                    proposal_probs /= np.sum(proposal_probs)
                else:
                    # Proposal is blind (CRP prior only)
                    proposal_probs = probs

                # Draw assignment
                idx = self.rng.choice(len(proposal_probs), p=proposal_probs)
                
                if idx < len(active_clusters):
                    chosen_cluster = active_clusters[idx]
                    z.append(chosen_cluster)
                    counts[chosen_cluster] += 1
                else:
                    # Create new cluster
                    new_cluster = len(counts)
                    z.append(new_cluster)
                    counts[new_cluster] = 1
                    # Draw a mean for this new cluster from G_0 posterior given x_i
                    # Posterior of mu | x_i:
                    # precision = 1/mu_prior_std^2 + 1/likelihood_std^2
                    # mean = (x_i / likelihood_std^2) / precision
                    prec_prior = 1.0 / (self.mu_prior_std**2)
                    prec_like = 1.0 / (self.likelihood_std**2)
                    post_var = 1.0 / (prec_prior + prec_like)
                    post_mean = (x_i * prec_like) * post_var
                    cluster_means[new_cluster] = self.rng.normal(post_mean, np.sqrt(post_var))

                # Accumulate logs
                # Prior probability of chosen step
                chosen_prior_prob = probs[idx]
                # Proposal probability of chosen step
                chosen_proposal_prob = proposal_probs[idx]
                # Likelihood probability of chosen step
                chosen_likelihood = likelihoods[idx]

                ln_f_total += np.log(chosen_prior_prob + 1e-15) + np.log(chosen_likelihood + 1e-15)
                ln_g_total += np.log(chosen_proposal_prob + 1e-15)

            partitions.append(np.array(z))
            log_weights[s] = ln_f_total - ln_g_total

        # Normalize weights
        max_log_w = np.max(log_weights)
        weights = np.exp(log_weights - max_log_w)
        normalized_weights = weights / np.sum(weights)

        return partitions, normalized_weights


# =====================================================================
# 5. Indian Buffet Process (IBP) Sequential Matrix Sampler
# =====================================================================
class IBPImportanceSampler:
    """
    Implements Sequential Importance Sampling for the Indian Buffet Process.
    Generates binary matrices where row i represents feature allocations for customer i.
    """
    def __init__(self, alpha, num_customers, seed=None):
        self.alpha = alpha
        self.N = num_customers
        self.rng = np.random.default_rng(seed)

    def run_sis(self, n_samples):
        """
        Draw binary matrices from the IBP prior.
        We return the generated samples and their weights (which are 1/n_samples since
        proposal is the prior itself).
        """
        matrices = []
        for s in range(n_samples):
            # We track the binary matrix as a list of lists.
            # Since K is infinite, columns expand dynamically.
            Z = []
            
            # Customer 1 tries Poisson(alpha) dishes
            k1 = self.rng.poisson(self.alpha)
            Z.append([1] * k1)
            
            # Customers 2 to N
            for i in range(1, self.N):
                customer_row = []
                current_K = len(Z[0]) if len(Z) > 0 else 0
                
                # 1. Existing dishes
                for k in range(current_K):
                    # Count previous selections of dish k
                    m_k = sum(Z[j][k] for j in range(i))
                    p_k = m_k / (i + 1)
                    customer_row.append(1 if self.rng.random() < p_k else 0)
                
                # 2. Propose Poisson(alpha / (i + 1)) new dishes
                k_new = self.rng.poisson(self.alpha / (i + 1))
                customer_row.extend([1] * k_new)
                
                # Pad earlier rows with 0s for the new dishes
                for j in range(i):
                    Z[j].extend([0] * k_new)
                    
                Z.append(customer_row)
            
            matrices.append(np.array(Z))
            
        # Weights are uniform since proposal = prior
        weights = np.ones(n_samples) / n_samples
        return matrices, weights


# Helper functions
def clip_val(val, min_val=1e-15, max_val=1.0 - 1e-15):
    return min(max(val, min_val), max_val)


# =====================================================================
# Execution & Demonstration
# =====================================================================
if __name__ == "__main__":
    print("--------------------------------------------------")
    print("Starting Bayesian Importance Sampling Framework")
    print("--------------------------------------------------")

    # 1. Run Toy Importance Sampler (Matches previous findings)
    print("\n--- 1. Running Toy Importance Sampler (1D Integral & Expectation) ---")
    gamma = -0.8
    zeta = -0.25
    toy_sampler = ToyImportanceSampler(gamma=gamma, zeta=zeta, seed=42)
    samples, weights, ess, log_weights = toy_sampler.run(n_samples=5000)
    
    # Estimating expectation of X under the normalized target distribution p(x) ~ x^gamma
    # E[X] = (gamma + 1) / (gamma + 2)
    est_expectation = np.sum(np.array(samples) * weights)
    true_expectation = (gamma + 1.0) / (gamma + 2.0)
    
    # Estimating the unnormalized integral (normalizing constant Z)
    # Z = int_0^1 x^gamma dx = 1 / (gamma + 1)
    # The standard importance sampling estimator of Z is the sample mean of exp(log_weights)
    est_integral = np.mean(np.exp(log_weights))
    true_integral = 1.0 / (gamma + 1.0)
    
    print(f"1. Integral (Normalizing Constant Z) Estimation:")
    print(f"   Estimated Integral value:    {est_integral:.5f}")
    print(f"   Analytical Integral value:   {true_integral:.5f}")
    print(f"2. Expectation of X Estimation:")
    print(f"   Estimated expectation E[X]:  {est_expectation:.5f}")
    print(f"   Analytical expectation E[X]: {true_expectation:.5f}")
    print(f"Effective Sample Size (ESS):     {ess:.2f} / 5000")

    # 2. Run Stick-Breaking Importance Sampler
    print("\n--- 2. Running Stick-Breaking DP Mixture Sampler ---")
    # Generate some synthetic grouped data: 3 true clusters around -2, 0, and 2
    true_means = [-2.0, 0.0, 2.0]
    rng = np.random.default_rng(42)
    synthetic_data = np.concatenate([
        rng.normal(true_means[0], 0.2, 5),
        rng.normal(true_means[1], 0.2, 5),
        rng.normal(true_means[2], 0.2, 5)
    ])
    
    sb_sampler = StickBreakingImportanceSampler(data=synthetic_data, alpha=1.5, K_trunc=8, seed=42)
    sb_samples, sb_weights, sb_ess, sb_log_weights = sb_sampler.run(n_samples=2000)
    
    # Find the best sample (maximum weight)
    best_idx = np.argmax(sb_weights)
    best_sample = sb_samples[best_idx]
    
    print(f"Stick-Breaking ESS: {sb_ess:.2f} / 2000")
    print(f"Max-weight sample cluster locations (thetas): {np.round(best_sample['thetas'], 2)}")
    print(f"Max-weight sample mixture weights (pi):       {np.round(best_sample['pi'], 2)}")

    # 3. Run CRP Sequential Importance Sampler
    print("\n--- 3. Running Chinese Restaurant Process SIS ---")
    crp_sampler = CRPImportanceSampler(data=synthetic_data, alpha=1.5, likelihood_std=0.5, seed=42)
    
    # Run both blind (prior-only) and informed (likelihood-assisted) samplers
    partitions_blind, weights_blind = crp_sampler.run_sis(n_samples=1000, informed=False)
    partitions_info, weights_info = crp_sampler.run_sis(n_samples=1000, informed=True)
    
    # Compute ESS for both
    ess_blind = 1.0 / np.sum(weights_blind ** 2)
    ess_info = 1.0 / np.sum(weights_info ** 2)
    
    print(f"CRP Blind Proposal ESS:      {ess_blind:.2f} / 1000")
    print(f"CRP Informed Proposal ESS:   {ess_info:.2f} / 1000")
    
    # Show best partition found
    best_part_idx = np.argmax(weights_info)
    print(f"Informed Proposal - Best partition of data points:")
    print(f"Data:       {np.round(synthetic_data, 1)}")
    print(f"Partitions: {partitions_info[best_part_idx]}")

    # 4. Run Indian Buffet Process Sampler
    print("\n--- 4. Running Indian Buffet Process Sampler ---")
    ibp_sampler = IBPImportanceSampler(alpha=2.0, num_customers=10, seed=42)
    ibp_matrices, ibp_weights = ibp_sampler.run_sis(n_samples=5)
    
    print(f"Generated {len(ibp_matrices)} matrices from the IBP prior.")
    print("Example binary matrix (Customer x Feature allocation):")
    print(ibp_matrices[0])

    # 5. Save Visualization Plot
    print("\n--- 5. Generating and Saving Diagnostic Plot ---")
    os.makedirs("./figs", exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Top Left: Toy Sampler expectation convergence of E[X]
    cumsum_expect = np.cumsum(np.array(samples) * weights) / np.cumsum(weights)
    axes[0, 0].plot(cumsum_expect, color="blue", label="Running Estimate")
    axes[0, 0].axhline(true_expectation, color="red", linestyle="--", label="True Expectation")
    axes[0, 0].set_title("Toy Sampler E[X] Convergence")
    axes[0, 0].set_xlabel("Sample index")
    axes[0, 0].set_ylabel("Expectation")
    axes[0, 0].legend()
    
    # Top Right: Stick-Breaking mixture weight profile of best sample
    axes[0, 1].bar(range(len(best_sample['pi'])), best_sample['pi'], color="teal", alpha=0.8)
    axes[0, 1].set_title("Stick-Breaking Truncated Mixture Weights (Best Sample)")
    axes[0, 1].set_xlabel("Cluster index")
    axes[0, 1].set_ylabel("Probability mass (pi)")
    
    # Bottom Left: CRP data assignment clustering
    best_z = partitions_info[best_part_idx]
    scatter = axes[1, 0].scatter(range(len(synthetic_data)), synthetic_data, c=best_z, cmap="rainbow", s=80, edgecolors='k')
    axes[1, 0].set_title("CRP Best Partition Data Assignments")
    axes[1, 0].set_xlabel("Data Point Index")
    axes[1, 0].set_ylabel("Data Value")
    
    # Bottom Right: IBP Feature Matrix heatmap
    axes[1, 1].imshow(ibp_matrices[0], cmap="Greys", aspect="auto", interpolation="nearest")
    axes[1, 1].set_title("IBP Latent Feature Binary Matrix (Sample 1)")
    axes[1, 1].set_xlabel("Feature Index")
    axes[1, 1].set_ylabel("Customer Index")
    
    fig.tight_layout()
    plot_path = "./figs/importance_sampling_bayesian.png"
    fig.savefig(plot_path)
    print(f"Visualizations saved successfully to: [figs/importance_sampling_bayesian.png](file:///{os.path.abspath(plot_path).replace(os.sep, '/')})")
    print("--------------------------------------------------")
