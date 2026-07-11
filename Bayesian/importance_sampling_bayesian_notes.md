# Importance Sampling, Laplace Transforms, and Dirac Measures in Bayesian Nonparametrics

This document consolidates the mathematical foundations and verification logs for the integrations in [importance_sampling_bayesian.py](importance_sampling_bayesian.py).

---

## 1. Mathematical Formulations

### A. Laplace Transform via Importance Sampling
The Laplace transform of a continuous probability density function $f(x)$ on $[0, \infty)$ for frequency $s$ is:
$$\mathcal{L}\{f\}(s) = \int_{0}^{\infty} e^{-sx} f(x) \, dx = \mathbb{E}[e^{-sX}]$$

We estimate the Laplace transform of a power-law function $x^\gamma$ over the domain $(0, 1]$:
$$I(s) = \int_{0}^{1} e^{-sx} x^\gamma \, dx$$
Using a proposal density $g(x) = (1+\zeta)x^\zeta$, we reformulate the integral as:
$$I(s) = \int_{0}^{1} \frac{e^{-sx} x^\gamma}{g(x)} g(x) \, dx = \mathbb{E}_{X \sim g}\left[ \frac{e^{-sX} X^\gamma}{(1+\zeta)X^\zeta} \right]$$

#### Analytical baseline
Using the substitution $u = sx$ ($dx = du/s$), the integral evaluates to:
$$I(s) = s^{-(\gamma+1)} \int_{0}^{s} e^{-u} u^\gamma \, du = s^{-(\gamma+1)} \gamma(\gamma+1, s)$$
where $\gamma(a, z) = \int_0^z e^{-u} u^{a-1} \, du$ is the **lower incomplete gamma function** (`scipy.special.gammainc(a, z) * scipy.special.gamma(a)`).

---

### B. Laplace Smoothing in CRP & Stick-Breaking
Laplace smoothing (or Lidstone smoothing) estimates categorical probabilities by adding pseudo-counts (prior parameters) to categories, avoiding zero-probability assignments for unseen classes.

1.  **Chinese Restaurant Process (CRP)**:
    In the CRP, the probability that customer $i+1$ sits at an existing table $c$ or starts a new table is:
    $$P(z_{i+1} = c \mid z_{1:i}) = \frac{n_c}{i + \alpha}, \quad P(z_{i+1} = \text{new} \mid z_{1:i}) = \frac{\alpha}{i + \alpha}$$
    The concentration parameter $\alpha$ acts precisely as the **additive pseudo-count** for the "unseen category" (the new cluster option). Without $\alpha > 0$, the probability of creating a new cluster would be $0$, collapsing the process into a single category.
2.  **Truncated Stick-Breaking Process**:
    In a truncated stick-breaking model of dimension $K$, the cluster weights $\pi_1, \dots, \pi_K$ sum to $1$. However, because weights decay exponentially, truncated weights for higher-index clusters can be extremely small or zero due to numerical underflow.
    To prevent proposal collapse where data points have $0$ probability of selecting a cluster, we apply Laplace smoothing with a parameter $\epsilon$:
    $$P(z_i = k) = \frac{\pi_k + \epsilon}{1 + K\epsilon}$$

---

### C. Dirac Delta Measures
The Dirac delta function $\delta_\theta(x)$ is defined such that:
$$\delta_\theta(x) = \begin{cases} \infty & \text{if } x = \theta \\ 0 & \text{if } x \neq \theta \end{cases} \quad \text{and} \quad \int \delta_\theta(x) \, dx = 1$$
It represents a discrete probability point mass (atom) located at $\theta$.

1.  **Dirichlet Process (DP)**:
    A draw from a Dirichlet Process $G \sim \text{DP}(\alpha, G_0)$ is discrete almost surely, represented by:
    $$G = \sum_{k=1}^{\infty} \pi_k \delta_{\theta_k}$$
    where $\theta_k \sim G_0$ are location parameters and $\pi_k$ are stick-breaking weights.
    In continuous density estimation, we evaluate the predictive distribution of a data point $x$ as:
    $$p(x) = \int \mathcal{N}(x \mid \theta, \sigma^2) \, dG(\theta) = \sum_{k=1}^K \pi_k \mathcal{N}(x \mid \theta_k, \sigma^2)$$
    Here, the Gaussian kernel $\mathcal{N}(x \mid \theta_k, \sigma^2)$ acts as a **smoothed Dirac delta** $\delta_{\theta_k}(x)$. As $\sigma \to 0$, this mixture density collapses back to the discrete measure $G$.
2.  **CRP (Yes)**:
    The Chinese Restaurant Process is the collapsed version of the Dirichlet Process. When we marginalize out the infinite weights $\boldsymbol{\pi}$, the predictive distribution of the next cluster location $\theta_i$ given previous locations is:
    $$\theta_i \mid \theta_{1:i-1} \sim \frac{\alpha}{\alpha + i - 1} G_0 + \sum_{c=1}^C \frac{n_c}{\alpha + i - 1} \delta_{\mu_c}$$
    Here, the **Dirac delta** $\delta_{\mu_c}$ explicitly directs the new point to choose the exact cluster center coordinate $\mu_c$, producing the clustering effect.
3.  **IBP (No, but uses indicators)**:
    The Indian Buffet Process generates binary feature matrices where feature allocations are represented by **indicator variables** $Z_{i,k} = \mathbb{I}(\text{customer } i \text{ has feature } k)$. While binary feature draws are discrete and can be modeled as Bernoulli point masses, the IBP does not partition coordinate space using continuous spatial Dirac atoms like the DP or CRP. Instead, it models feature subsets.

---

## 1.5 Comparison: Importance Sampling vs. MCMC

This repository implements both Importance Sampling (IS) and Markov Chain Monte Carlo (MCMC). Here is a comparison of their strengths, weaknesses, and how MCMC implementations evolved to solve specific problems.

### Pros and Cons
| Feature | Importance Sampling (IS) | Markov Chain Monte Carlo (MCMC) |
| :--- | :--- | :--- |
| **Sample Independence** | **Independent** (No trapping) | **Dependent** (Highly correlated, requires burn-in) |
| **Singularities / Peaks** | **Excellent** (if proposal is good) | **Poor** (mixes slowly or gets physically trapped) |
| **High Dimensions** | **Extremely Poor** (Weight collapse, ESS $\to 1$) | **Good** (Finds the high-probability typical set) |
| **Normalizing Constant** | Calculates integral directly | Cannot easily compute the integral |

### Evolution of MCMC in this Repository
1. **Standard Local MCMC (Ch 1-2):** E.g., the Metropolis algorithm in `markov-zeta.py`. Works well to scale into higher dimensions but gets physically trapped near non-integrable singularities (e.g. $\gamma \le -1.0$) or gets stuck in local modes (critical slowing down).
2. **Informed Metropolis-Hastings (Independence Sampler):** Implemented in `mcmc_bayesian.py`, this evolution borrows the global proposal from Importance Sampling to generate MCMC steps, vastly reducing autocorrelation and allowing escape from local traps.
3. **Path Integral Monte Carlo (PIMC) (Ch 7):** As seen in the `homework_7` BEC simulations, MCMC evolves beyond sampling discrete variables or single coordinates by sampling entire continuous paths (Lévy flights) and permutation cycles to handle quantum indistinguishability.
4. **Cluster Algorithms (Ch 5 & 8):** Algorithms like the Wolff cluster update in `cluster_ising.py` solve critical slowing down by flipping macroscopic clusters of spins at once, allowing global, rejection-free moves.
5. **SOTA - Mirror HMC (Reparameterization Trick):** Implemented in `mcmc_bayesian.py`. The absolute state-of-the-art for continuous spaces relies on gradients. Mirror HMC applies a logit transform to map bounded parameters into an unbounded smooth space, allowing Hamiltonian Leapfrog integration to deterministically glide toward high probabilities—exactly mirroring how VAEs in Deep Learning use the reparameterization trick.
6. **SOTA - Tempered HMC:** Also implemented in `mcmc_bayesian.py`. For multimodal distributions (like a bimodal Gaussian Mixture), gradients can trap samplers in local valleys. Tempered HMC temporarily flattens the energy landscape to allow physical trajectories to cross impossible barriers before returning to the target density.
7. **Dynamic Monte Carlo / Kinetic MC (Ch 9):** Scripts like `dynamic_ising.py` solve the low-acceptance-rate problem at low temperatures by tracking all possible moves and executing them rejection-free based on their physical transition rates.
8. **Sequential Importance Sampling (CRP/IBP):** For complex, growing-dimension Bayesian nonparametric models, the repo builds configurations step-by-step with likelihood-informed proposals, beautifully combining the independence of Importance Sampling with the sequential state-building of MCMC.

---

## 2. Implementation Overview

The components are implemented inside [importance_sampling_bayesian.py](importance_sampling_bayesian.py):
*   **`ToyImportanceSampler`**: Evaluates Laplace transform estimates of $x^\gamma$ at arbitrary frequency $s$.
*   **`StickBreakingImportanceSampler`**: Implements Laplace smoothing `epsilon` to prevent cluster proposal collapse, and provides `evaluate_dirac_mixture_density` to compute DP mixture density curves.
*   **`CRPImportanceSampler`**: Combines tables using sequential importance sampling (SIS) with likelihood-informed proposals.
*   **`IBPImportanceSampler`**: Generates latent feature binary matrices sequentially.

---

## 3. Verification & Execution Logs

Executing the file:
`python book/importance_sampling_bayesian.py`

### Run Output:
```text
--------------------------------------------------
Starting Bayesian Importance Sampling Framework
--------------------------------------------------

--- 1. Running Toy Importance Sampler (Laplace Transform) ---
1. Standard Integral (Normalizing Constant Z, s=0) Estimation:
   Estimated Integral value:    4.50692
   Analytical Integral value:   5.00000
2. Laplace Transform (s=2.0) Estimation:
   Estimated L{x^gamma}(2.0):   3.45323
   Analytical L{x^gamma}(2.0):  3.94466
Effective Sample Size (ESS):     326.23 / 5000

--- 2. Running Stick-Breaking DP Mixture Sampler (Laplace Smoothed) ---
Stick-Breaking ESS: 1.00 / 2000
Max-weight sample cluster locations (thetas): [ 1.04 -0.1  -3.06  2.74  8.11  3.57 -0.33 -0.37]
Max-weight sample mixture weights (pi):       [0.24 0.5  0.15 0.04 0.01 0.01 0.02 0.02]

--- 3. Running Chinese Restaurant Process SIS ---
CRP Blind Proposal ESS:      1.00 / 1000
CRP Informed Proposal ESS:   235.22 / 1000
Informed Proposal - Best partition of data points:
Data:       [-1.9 -2.2 -1.8 -1.8 -2.4 -0.3  0.  -0.1 -0.  -0.2  2.2  2.2  2.   2.2
  2.1]
Partitions: [0 0 0 0 0 1 1 1 1 2 3 3 3 3 3]

--- 4. Running Indian Buffet Process Sampler ---
Generated 5 matrices from the IBP prior.
Example binary matrix (Customer x Feature allocation):
[[1 1 1 1 0 0 0 0]
 [0 0 0 1 1 0 0 0]
 [0 0 0 1 1 1 0 0]
 [0 0 0 1 0 0 1 0]
 [0 1 1 1 0 0 0 0]
 [0 1 1 1 1 0 0 1]
 [0 0 0 1 1 0 1 0]
 [0 0 0 1 1 0 1 1]
 [0 0 0 1 1 0 1 1]
 [0 0 0 1 1 0 1 0]]

--- 5. Generating and Saving Diagnostic Plot ---
Visualizations saved successfully to: [figs/importance_sampling_bayesian.png](figs/importance_sampling_bayesian.png)
--------------------------------------------------
```
