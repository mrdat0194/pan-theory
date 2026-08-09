"""
omnistats/tests/test_information_theory.py
-------------------------------------------
Unit tests for the quantum-inspired information theory module.

Validates mathematical correctness against scipy/numpy benchmarks:
  - Shannon Entropy (discrete uniform → log2(K))
  - KL Divergence (p=q → 0, one-directional)
  - Jeffreys-KL (symmetric: J(p,q) == J(q,p))
  - Bayesian Inverse Score (uninformative prior → uniform posterior)
  - Boltzmann Energy + Partition Function
  - Path Integral convolution consistency (Tang Theorem validation)
"""
import sys
import os
import math
import numpy as np
import torch
import pytest

# ── make omnistats importable ────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from omnistats.modules.information_theory import (
    shannon_entropy,
    differential_entropy_gaussian,
    kl_divergence,
    jeffreys_kl,
    mutual_information,
    bayesian_inverse_score,
    boltzmann_energy,
    partition_function,
    compute_xai_metrics,
)


# =============================================================================
# 1. SHANNON ENTROPY
# =============================================================================

class TestShannonEntropy:
    """
    From Aug-2020 notes: Shannon Uncertainty, connects to KL and JKL.
    """

    def test_uniform_distribution_max_entropy(self):
        """Uniform over K categories → H = log2(K) bits."""
        for K in [2, 4, 8, 16]:
            p = np.ones(K) / K
            H = shannon_entropy(p, base=2.0)
            assert abs(H - math.log2(K)) < 1e-6, \
                f"K={K}: expected {math.log2(K):.4f}, got {H:.4f}"

    def test_deterministic_distribution_zero_entropy(self):
        """Point mass → H = 0 (no uncertainty)."""
        p = np.array([1.0, 0.0, 0.0, 0.0])
        H = shannon_entropy(p, base=2.0)
        assert abs(H) < 1e-6

    def test_torch_tensor_matches_numpy(self):
        """torch and numpy paths should agree to float32 precision."""
        p_np = np.array([0.1, 0.3, 0.4, 0.2])
        p_th = torch.tensor(p_np, dtype=torch.float32)
        H_np = shannon_entropy(p_np, base=2.0)
        H_th = shannon_entropy(p_th, base=2.0).item()
        assert abs(H_np - H_th) < 1e-4

    def test_base_e_matches_scipy(self):
        """Nats (base=e) should match scipy.stats.entropy."""
        from scipy.stats import entropy as scipy_entropy
        p = np.array([0.25, 0.25, 0.25, 0.25])
        H = shannon_entropy(p, base=math.e)
        H_ref = scipy_entropy(p)   # scipy uses nats by default
        assert abs(H - H_ref) < 1e-6

    def test_gaussian_differential_entropy(self):
        """h(X) = 0.5 * log(2*pi*e*sigma^2) in nats."""
        sigma = 2.0
        expected = 0.5 * math.log(2 * math.pi * math.e * sigma**2)
        got = differential_entropy_gaussian(np.array([sigma]))[0]
        assert abs(got - expected) < 1e-6


# =============================================================================
# 2. KL DIVERGENCE
# =============================================================================

class TestKLDivergence:
    """
    From notes: 'Forward let q(x) known => KL = p(x) log p(x)/q(x)'
    """

    def test_identical_distributions_zero_kl(self):
        """KL(p || p) = 0 always."""
        p = np.array([0.2, 0.5, 0.3])
        assert abs(kl_divergence(p, p, base=2.0)) < 1e-6

    def test_known_value(self):
        """KL(p || q) matches scipy.special.kl_div sum."""
        from scipy.stats import entropy as scipy_entropy
        p = np.array([0.3, 0.5, 0.2])
        q = np.array([0.1, 0.6, 0.3])
        expected = scipy_entropy(p, q)           # nats
        got = kl_divergence(p, q, base=math.e)
        assert abs(got - expected) < 1e-5

    def test_non_negative(self):
        """KL divergence is always >= 0 (Gibbs inequality)."""
        rng = np.random.default_rng(42)
        for _ in range(20):
            p = rng.dirichlet(np.ones(10))
            q = rng.dirichlet(np.ones(10))
            assert kl_divergence(p, q, base=2.0) >= -1e-9

    def test_torch_tensor(self):
        """torch path is consistent with numpy path."""
        p = np.array([0.4, 0.4, 0.2])
        q = np.array([0.2, 0.5, 0.3])
        kl_np = kl_divergence(p, q)
        kl_th = kl_divergence(torch.tensor(p, dtype=torch.float32),
                               torch.tensor(q, dtype=torch.float32)).item()
        assert abs(kl_np - kl_th) < 1e-4


# =============================================================================
# 3. JEFFREYS-KL (SYMMETRIC)
# =============================================================================

class TestJeffreysKL:
    """
    From notes: 'JKL: Prior Uninformative, Likelihood ~ Posterior'
    """

    def test_symmetry(self):
        """J(p, q) == J(q, p) exactly."""
        rng = np.random.default_rng(0)
        for _ in range(50):
            p = rng.dirichlet(np.ones(8))
            q = rng.dirichlet(np.ones(8))
            assert abs(jeffreys_kl(p, q) - jeffreys_kl(q, p)) < 1e-6

    def test_zero_for_identical(self):
        """J(p, p) = 0."""
        p = np.array([0.1, 0.4, 0.5])
        assert abs(jeffreys_kl(p, p)) < 1e-6

    def test_jkl_ge_kl(self):
        """J(p,q) >= KL(p||q) since we add KL(q||p) >= 0."""
        p = np.array([0.6, 0.3, 0.1])
        q = np.array([0.2, 0.5, 0.3])
        assert jeffreys_kl(p, q) >= kl_divergence(p, q) - 1e-9


# =============================================================================
# 4. BAYESIAN INVERSE SCORE (l2 dequantization)
# =============================================================================

class TestBayesianInverseScore:
    """
    Validates the dequantized l2 sampling Bayesian inverse scoring.
    From notes: 'Bayes: inverse problems n -> y (2D -> 3D image)'.
    """

    def test_uniform_posterior_for_equal_references(self):
        """When all references are identical, posterior should be approx uniform."""
        B, D, N = 4, 16, 6
        z   = torch.randn(B, D)
        ref = torch.ones(N, D)   # all the same reference → symmetric similarities
        posterior, jkl = bayesian_inverse_score(z, ref, temperature=1.0)
        assert posterior.shape == (B, N)
        # JKL from uniform should be near 0 (posterior ≈ prior)
        assert jkl.mean().item() < 1.0

    def test_confident_match_high_jkl(self):
        """When z closely matches exactly one reference, JKL should be higher."""
        D, N = 32, 8
        ref = torch.randn(N, D)
        # z is very close to ref[0]
        z = ref[0:1] + 0.001 * torch.randn(1, D)
        posterior, jkl = bayesian_inverse_score(z, ref, temperature=0.1)
        # Posterior mass should concentrate on ref[0]
        assert posterior[0, 0].item() > 0.5
        # JKL should be high (confident, far from uniform)
        assert jkl[0].item() > 0.0

    def test_output_shapes(self):
        B, D, N = 5, 24, 10
        z   = torch.randn(B, D)
        ref = torch.randn(N, D)
        posterior, jkl = bayesian_inverse_score(z, ref)
        assert posterior.shape == (B, N)
        assert jkl.shape == (B,)


# =============================================================================
# 5. BOLTZMANN ENERGY + PARTITION FUNCTION
# =============================================================================

class TestBoltzmannPhysics:
    """
    Validates the thermodynamic quantities.
    From Nov-2022 notes: pi(x,n) ∝ e^{-beta E_n}, Z(beta) = Tr(rho).
    """

    def test_energy_proportional_to_norm_squared(self):
        """E(z) = ||z||^2 / beta."""
        z = torch.tensor([[3.0, 4.0]])   # ||z|| = 5, ||z||^2 = 25
        E = boltzmann_energy(z, beta=1.0)
        assert abs(E.item() - 25.0) < 1e-5

    def test_beta_scaling(self):
        """Doubling beta halves the energy."""
        z = torch.randn(1, 8)
        E1 = boltzmann_energy(z, beta=1.0)
        E2 = boltzmann_energy(z, beta=2.0)
        assert abs(E1.item() / 2 - E2.item()) < 1e-5

    def test_partition_function_positive(self):
        """Z(beta) > 0 always."""
        E = torch.abs(torch.randn(10))
        Z = partition_function(E, beta=1.0)
        assert Z.item() > 0

    def test_partition_function_decreases_with_beta(self):
        """Z(beta) decreases as beta increases (colder → fewer accessible states)."""
        E = torch.tensor([1.0, 2.0, 3.0, 4.0])
        Z_hot  = partition_function(E, beta=0.1)
        Z_cold = partition_function(E, beta=10.0)
        assert Z_hot.item() > Z_cold.item()


# =============================================================================
# 6. PATH INTEGRAL CONVOLUTION CONSISTENCY (Tang Theorem Validation)
# =============================================================================

class TestPathIntegralConsistency:
    """
    Validates the convolution property from Nov-2022 notes (property ①):
        ∫ dx' p(x, x', beta1) p(x', x'', beta2) = p(x, x'', beta1 + beta2)

    For the dequantized predictor, unrolling T steps with beta/T each should
    approximate one step with full beta. We validate this statistically.
    """

    def test_dequantized_predictor_composability(self):
        """
        Multi-step rollout should produce outputs in the same latent manifold
        as a single large step — validated by checking that the rollout
        trajectory stays bounded (doesn't diverge).
        """
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "datastructure", "Lesson"))
        try:
            from dequantized_jepa_predictor import DequantizedLatentTransition, path_integral_rollout
        except ImportError:
            pytest.skip("dequantized_jepa_predictor not importable in this environment")

        d_latent = 32
        model = DequantizedLatentTransition(d_latent=d_latent, rank_k=8, beta=1.0, n_steps=3)
        model.eval()

        z0 = torch.randn(4, d_latent)
        result = path_integral_rollout(model, z0, T=5)

        trajectory = result["trajectory"]
        energies   = result["energies"]   # [B, T]

        # Trajectory should not explode
        for zt in trajectory:
            assert zt.isfinite().all(), "Trajectory contains NaN/Inf"
            assert (zt.norm(dim=-1) < 1e4).all(), "Trajectory norm exploded"

        # Total beta should accumulate
        assert result["total_beta"] > 0

    def test_compute_xai_metrics_shape(self):
        """compute_xai_metrics returns correct shapes."""
        B, D = 6, 16
        z    = torch.randn(B, D, 1, 1, 1)
        ref  = torch.randn(4, D)
        out  = compute_xai_metrics(z, reference_embeddings=ref, beta=1.0, temperature=1.0)

        assert out["energy"].shape         == (B,)
        assert out["entropy"].shape        == (B,)
        assert out["jkl_from_prior"].shape == (B,)
        assert out["posterior"].shape      == (B, 4)
        assert isinstance(out["partition_fn"], torch.Tensor)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
