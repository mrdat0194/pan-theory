"""
spectral_ode_solver.py
======================
Educational Step-by-Step Guide: Solving ODEs & PDEs with the FFT
(Spectral Methods)

WHY DOES FFT CONNECT TO ODES?
==============================
When we differentiate a function in the TIME / SPACE domain, it corresponds to a
simple MULTIPLICATION in the FREQUENCY domain. This is the key insight:

    If F(omega) = FFT[f(t)], then:

    FFT[ df/dt ](omega) = (i * omega) * F(omega)

This means instead of approximating derivatives with finite differences (which
accumulate O(h) error), we can:
    1. Take the FFT of our signal → F(omega)
    2. Multiply by (i * omega) → exact spectral derivative
    3. Take the IFFT → recover df/dt in the time domain

This is the foundation of SPECTRAL METHODS, which underpin:
  - Neural ODE solvers in physics-informed networks
  - PDE solvers in computational fluid dynamics
  - The Fourier Neural Operator (FNO) — a SOTA SOTA ML model for PDEs

This file demonstrates:
    STEP 1: The Spectral Derivative (d/dt via FFT)
    STEP 2: Solving the 1D Heat Equation using Spectral Methods
    STEP 3: The connection between Spectral Methods and Neural ODEs

Run with: python datastructure/Lesson/spectral_ode_solver.py
"""

import numpy as np
import math


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: The Spectral Derivative
# d/dt of a function computed using FFT — O(N log N), no finite differences
# ─────────────────────────────────────────────────────────────────────────────

def spectral_derivative(f: np.ndarray, L: float = 2 * math.pi) -> np.ndarray:
    """
    Compute the exact derivative df/dx of a periodic function f(x)
    using the Fourier Transform.

    Mathematical Foundation:
    -----------------------
    For a periodic function f with period L, its Fourier Transform gives
    coefficients F(omega). The derivative formula becomes:

        d/dx [ f(x) ] ↔ (i * omega_k) * F(omega_k)

    where omega_k = 2*pi*k/L are the wavenumbers.

    This is EXACT (up to floating point) for smooth, periodic functions —
    unlike finite differences which only approximate to O(h^2).

    Args:
        f: array of function values at N equally-spaced points over [0, L)
        L: the period of the domain (default: 2*pi)

    Returns:
        df_dx: the exact spectral derivative at the same grid points
    """
    N = len(f)

    # Step 1a: Transform f to frequency domain
    # F[k] = sum_j f[j] * exp(-2*pi*i*j*k/N)
    F = np.fft.fft(f)

    # Step 1b: Build the wavenumber array omega_k = 2*pi*k/L
    # np.fft.fftfreq gives k/N, so we multiply by 2*pi*N/L to get physical wavenumbers
    k = np.fft.fftfreq(N, d=L / (2 * math.pi * N))

    # Step 1c: Multiply by i*omega in the frequency domain
    # This is the spectral derivative operator
    dF = 1j * k * F

    # Step 1d: Transform back to the time/space domain (IFFT)
    df_dx = np.real(np.fft.ifft(dF))

    return df_dx


def demonstrate_spectral_derivative():
    """
    Compare the spectral derivative to the known analytical derivative.

    Test function: f(x) = sin(3x)
    Exact answer:  df/dx = 3*cos(3x)
    """
    print("=" * 60)
    print(" STEP 1: Spectral Derivative via FFT")
    print("=" * 60)
    print()
    print("  Test function : f(x) = sin(3x)  on  x ∈ [0, 2π)")
    print("  Exact answer  : df/dx = 3·cos(3x)")
    print()

    N = 64
    x = np.linspace(0, 2 * math.pi, N, endpoint=False)

    f = np.sin(3 * x)
    df_exact = 3 * np.cos(3 * x)
    df_spectral = spectral_derivative(f, L=2 * math.pi)

    max_error = np.max(np.abs(df_spectral - df_exact))
    print(f"  N grid points      : {N}")
    print(f"  Max absolute error : {max_error:.2e}")
    print()
    print("  Sample comparison (first 5 points):")
    print(f"  {'x':>8}  {'exact df':>12}  {'spectral df':>12}  {'error':>12}")
    print("  " + "-" * 50)
    for i in range(5):
        print(f"  {x[i]:>8.4f}  {df_exact[i]:>12.6f}  {df_spectral[i]:>12.6f}  {abs(df_spectral[i] - df_exact[i]):>12.2e}")
    print()
    print("  → Spectral derivative is machine-precision accurate.")
    print("     Finite differences would give ~O(h²) = O(1/N²) error only.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Solving the 1D Heat Equation with Spectral Methods
#
# The Heat Equation (PDE):
#   ∂u/∂t = α * ∂²u/∂x²
#
# where u(x, t) is temperature and α is thermal diffusivity.
#
# SPECTRAL APPROACH:
#   Take FFT of both sides in x:
#     d/dt [U_k(t)] = α * (i*omega_k)² * U_k(t)
#                   = -α * omega_k² * U_k(t)
#
#   This is now a SIMPLE ODE for each Fourier mode U_k(t)!
#   It has the exact closed-form solution:
#     U_k(t) = U_k(0) * exp(-α * omega_k² * t)
#
#   No time-stepping error — just evolve the modes analytically!
# ─────────────────────────────────────────────────────────────────────────────

def solve_heat_equation_spectral(u0: np.ndarray, alpha: float,
                                  t_values: list, L: float = 2 * math.pi) -> list:
    """
    Solve the 1D Heat Equation ∂u/∂t = α * ∂²u/∂x² spectrally.

    Instead of using finite differences which accumulate time-stepping errors,
    this method:
        1. Transforms the initial condition u(x, 0) to frequency domain → U_k(0)
        2. Each Fourier mode decays independently as: U_k(t) = U_k(0) * exp(-α*ω²*t)
        3. At any time t, take IFFT(U_k(t)) to recover u(x, t)

    This is a SPECTRAL METHOD — exact for periodic, smooth initial conditions.

    Args:
        u0:       initial temperature profile u(x, 0), shape [N]
        alpha:    thermal diffusivity coefficient
        t_values: list of time points at which to compute u(x, t)
        L:        domain length (period)

    Returns:
        solutions: list of u(x, t) arrays, one per t in t_values
    """
    N = len(u0)

    # Fourier transform the initial condition
    U0 = np.fft.fft(u0)

    # Build the wavenumbers omega_k = 2*pi*k/L
    k = np.fft.fftfreq(N, d=L / (2 * math.pi * N))

    # The decay exponent for each mode: -alpha * omega_k^2
    decay_rates = -alpha * k**2

    solutions = []
    for t in t_values:
        # Each mode evolves independently: U_k(t) = U_k(0) * exp(decay_rate * t)
        # This is the exact analytical solution to the ODE for each Fourier mode
        decay_factors = np.exp(decay_rates * t)
        Ut = U0 * decay_factors

        # Transform back to physical space
        ut = np.real(np.fft.ifft(Ut))
        solutions.append(ut)

    return solutions


def demonstrate_heat_equation():
    """
    Solve the 1D Heat Equation for a simple Gaussian blob initial condition.
    Show how the Gaussian spreads (diffuses) over time.

    Exact solution for a Gaussian: width grows as sqrt(width_0^2 + 2*alpha*t)
    """
    print("=" * 60)
    print(" STEP 2: 1D Heat Equation via Spectral Method")
    print("=" * 60)
    print()
    print("  PDE  : ∂u/∂t = α · ∂²u/∂x²   (Heat / Diffusion Equation)")
    print("  u0   : Gaussian blob centred at x=π")
    print("  α    : 0.1  (thermal diffusivity)")
    print()

    N = 128
    L = 2 * math.pi
    x = np.linspace(0, L, N, endpoint=False)

    # Initial condition: narrow Gaussian
    sigma0 = 0.3
    u0 = np.exp(-((x - math.pi) ** 2) / (2 * sigma0 ** 2))

    alpha = 0.1
    t_values = [0.0, 0.5, 1.0, 2.0, 5.0]

    solutions = solve_heat_equation_spectral(u0, alpha, t_values, L)

    print(f"  {'Time':>6}  {'Max u':>10}  {'Std (width)':>12}  {'Expected width':>14}")
    print("  " + "-" * 48)
    for t, u in zip(t_values, solutions):
        expected_width = math.sqrt(sigma0**2 + 2 * alpha * t)
        # Compute numerical std as a proxy for peak width
        total = np.sum(u) + 1e-15
        mean_x = np.sum(x * u) / total
        std_x = math.sqrt(np.sum(((x - mean_x) ** 2) * u) / total)
        print(f"  {t:>6.1f}  {np.max(u):>10.5f}  {std_x:>12.5f}  {expected_width:>14.5f}")

    print()
    print("  → Spectral 'std (width)' matches the analytical 'expected width'.")
    print("  → The Gaussian diffuses exactly as predicted — no numerical drift.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: The Bridge to Neural ODEs
#
# The spectral method above solved the ODE for each Fourier mode:
#   dU_k/dt = -α * ω_k² * U_k
#
# In a NEURAL ODE (like our LatentODE_JEPA), we replace the fixed physics:
#   dU_k/dt = -α * ω_k² * U_k    (fixed: known PDE)
#
# with a LEARNED dynamics function:
#   dz/dt = f_theta(z, t)          (learned: Neural ODE)
#
# The same RK4 or spectral integration approach is used to step the
# learned dynamics forward. The model learns whatever ODE best explains
# the observed data — giving us flexible, physics-inspired time series models.
# ─────────────────────────────────────────────────────────────────────────────

def demonstrate_bridge_to_neural_ode():
    """
    Show that a learned Neural ODE and a spectral solver share the same structure.
    Both solve: dz/dt = f(z, t) with a numerical integrator.
    The difference is only whether f is a known physics equation or a neural network.
    """
    print("=" * 60)
    print(" STEP 3: Bridge — Spectral Methods → Neural ODEs")
    print("=" * 60)
    print()
    print("  The Heat Equation spectral solver used:")
    print()
    print("    Fixed physics:   dU_k/dt = −α·ω_k² · U_k")
    print()
    print("  A Neural ODE (like our LatentODE_JEPA) replaces this with:")
    print()
    print("    Learned dynamics: dz/dt = f_θ(z, t)")
    print()
    print("  Both are integrated using the same numerical integrators (Euler, RK4).")
    print("  The ONLY difference is whether the ODE's right-hand side f is:")
    print()
    print("    (a) Known physics   → Spectral / Finite-Difference solver")
    print("    (b) Neural network  → Neural ODE (Latent ODE JEPA)")
    print()
    print("  See: MLModel/AIModel/model/latent_ode_jepa.py")
    print("       → ODEDynamicsNet.forward(z, t)    ← this IS f_theta")
    print("       → odeint_rk4(f, z0, t0, t1)       ← this IS the integrator")
    print()
    print("  ─────────────────────────────────────────────────────────")
    print("  FFT teaches WHAT a derivative looks like in frequency space.")
    print("  Neural ODEs learn WHAT the derivative IS from data.")
    print("  ─────────────────────────────────────────────────────────")


if __name__ == "__main__":
    print()
    print("=" * 62)
    print("  SPECTRAL METHODS & NEURAL ODES - Educational Walkthrough")
    print("=" * 62)
    print()

    demonstrate_spectral_derivative()
    print()
    demonstrate_heat_equation()
    print()
    demonstrate_bridge_to_neural_ode()
