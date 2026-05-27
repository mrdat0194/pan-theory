"""
Random Matrix Theory exercise solution.
Inspired by Sethna, "Entropy, Order Parameters, and Complexity", ex. 1.6
"""

import numpy as np
import matplotlib.pyplot as plt

def GOE(N):
    """
    Creates an NxN element of the Gaussian Orthogonal Ensemble (GOE).
    Generates a matrix of Gaussian random variables (mean=0, std=1)
    and adds it to its transpose to symmetrize it.
    """
    m = np.random.standard_normal((N, N))
    m = m + np.transpose(m)
    return m

def GOE_Ensemble(num, N):
    """
    Generates a list (ensemble) of 'num' GOE NxN matrices.
    """
    return [GOE(N) for _ in range(num)]

def CenterEigenvalueDifferences(ensemble):
    """
    For each symmetric matrix in the ensemble, calculates the difference
    between the two center eigenvalues, and returns the differences as an array.
    """
    N = len(ensemble[0])
    diffs = []
    for mat in ensemble:
        # Use eigvalsh as it's optimized for symmetric/Hermitian matrices
        eig = sorted(np.linalg.eigvalsh(mat))
        diffs.append(eig[N // 2] - eig[N // 2 - 1])
    return np.array(diffs)

def Wigner(s):
    """
    Returns the Wigner surmise for the probability distribution rho(s)
    for the eigenvalue differences in the GOE ensemble:
    rho(s) = (pi * s / 2) * exp(-pi * s^2 / 4)
    """
    return (np.pi * s / 2.0) * np.exp(-np.pi * s**2 / 4.0)

def CompareEnsembleWigner(ensemble, title_text="Eigenvalue Spacing"):
    """
    Plots the center eigenvalue difference histogram of an ensemble,
    normalized, and overlays the theoretical Wigner Surmise curve.
    """
    diffs = CenterEigenvalueDifferences(ensemble)
    mean_diff = np.mean(diffs)
    # Normalize differences by dividing by their mean
    normalized_diffs = diffs / mean_diff

    # Plot normalized histogram
    plt.hist(normalized_diffs, bins=50, density=True, alpha=0.6, color='g', label='Ensemble')

    # Plot Wigner surmise theory curve
    s = np.arange(0.0, 3.0, 0.01)
    theory = Wigner(s)
    plt.plot(s, theory, 'r-', linewidth=2, label='Wigner Surmise')
    
    plt.title(title_text)
    plt.xlabel('Normalized Spacing (s)')
    plt.ylabel('Probability Density')
    plt.legend()

def PM1_Ensemble(num, N):
    """
    Generates a symmetric +-1 ensemble by taking the sign of a GOE ensemble.
    """
    return np.sign(GOE_Ensemble(num, N))

def demo():
    M = 10000
    print("Random Matrix Theory Simulation Demo\n")

    # 1. 2x2 GOE vs Wigner
    print("1. Plotting 2x2 GOE matrix eigenvalue differences...")
    plt.figure()
    CompareEnsembleWigner(GOE_Ensemble(M, 2), "2x2 GOE matrix eigenvalue differences")
    plt.show()

    # 2. 4x4 GOE vs Wigner
    print("2. Plotting 4x4 GOE matrix eigenvalue differences...")
    plt.figure()
    CompareEnsembleWigner(GOE_Ensemble(M, 4), "4x4 GOE matrix eigenvalue differences")
    plt.show()

    # 3. 10x10 GOE vs Wigner
    print("3. Plotting 10x10 GOE matrix eigenvalue differences...")
    plt.figure()
    CompareEnsembleWigner(GOE_Ensemble(M, 10), "10x10 GOE matrix eigenvalue differences")
    plt.show()

    # 4. Universality checks with +-1 matrices
    print("4. Plotting +-1 matrix ensembles to demonstrate universality...")
    
    # N = 2
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    CompareEnsembleWigner(PM1_Ensemble(M, 2), "+-1 Matrix Ensemble (N=2)")
    
    # N = 4
    plt.subplot(1, 3, 2)
    CompareEnsembleWigner(PM1_Ensemble(M, 4), "+-1 Matrix Ensemble (N=4)")
    
    # N = 10
    plt.subplot(1, 3, 3)
    CompareEnsembleWigner(PM1_Ensemble(M, 10), "+-1 Matrix Ensemble (N=10)")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    demo()
