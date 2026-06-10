# Importance Sampling, Laplace Transforms, and Dirac Measures in Bayesian Nonparametrics

This document consolidates the mathematical foundations and verification logs for the integrations in [importance_sampling_bayesian.py](file:///c:/Users/mrdat/PycharmProjects/smac/book/importance_sampling_bayesian.py).

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

## 2. Implementation Overview

The components are implemented inside [importance_sampling_bayesian.py](file:///c:/Users/mrdat/PycharmProjects/smac/book/importance_sampling_bayesian.py):
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
Visualizations saved successfully to: [figs/importance_sampling_bayesian.png](file:///C:/Users/mrdat/PycharmProjects/smac/figs/importance_sampling_bayesian.png)
--------------------------------------------------
```
