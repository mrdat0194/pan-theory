"""
Random Walk exercise solution.
Inspired by Sethna, "Entropy, Order Parameters, and Complexity", ex. 2.5
"""

import numpy as np
import matplotlib.pyplot as plt

def RandomWalk(N, d):
    """
    Generate an N-step random walk in d dimensions, with each step
    uniformly distributed in [-1/2, 1/2] in each dimension.
    """
    steps = np.random.uniform(-0.5, 0.5, (N, d))
    walks = np.cumsum(steps, axis=0)
    return walks

def Endpoints(W, N, d):
    """
    Returns the endpoints of W random walks of N steps each in d dimensions.
    """
    steps = np.random.uniform(-0.5, 0.5, (W, N, d))
    return np.sum(steps, axis=1)

def demo():
    print("Random Walk Simulation Demo")
    
    # 1. Plot 1D random walks for different N
    print("Plotting 1D random walks...")
    plt.figure(figsize=(10, 8))
    
    plt.subplot(3, 1, 1)
    for _ in range(10):
        plt.plot(RandomWalk(10, 1))
    plt.title("1D Random Walks (N=10)")
    
    plt.subplot(3, 1, 2)
    for _ in range(10):
        plt.plot(RandomWalk(100, 1))
    plt.title("1D Random Walks (N=100)")
    
    plt.subplot(3, 1, 3)
    for _ in range(10):
        plt.plot(RandomWalk(10000, 1))
    plt.title("1D Random Walks (N=10000)")
    plt.tight_layout()
    plt.show()

    # 2. Plot 2D random walks
    print("Plotting 2D random walks...")
    plt.figure(figsize=(6, 6))
    plt.gca().set_aspect('equal')
    for _ in range(10):
        x, y = RandomWalk(10000, 2).transpose()
        plt.plot(x, y)
    plt.title("2D Random Walks (N=10000)")
    plt.show()

    # 3. Plot Endpoints (Illustrating emergent symmetry)
    print("Plotting endpoints to illustrate emergent symmetry...")
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.gca().set_aspect('equal')
    x, y = Endpoints(10000, 10, 2).transpose()
    plt.plot(x, y, '.', alpha=0.5)
    plt.title("Endpoints of 10000 walks (N=10)")
    
    plt.subplot(1, 2, 2)
    plt.gca().set_aspect('equal')
    x, y = Endpoints(10000, 1, 2).transpose()
    plt.plot(x, y, '.', alpha=0.5)
    plt.title("Endpoints of 10000 walks (N=1)")
    plt.tight_layout()
    plt.show()

    # 4. Central Limit Theorem verification for N = 1, 2, 5
    print("Verifying Central Limit Theorem...")
    plt.figure(figsize=(12, 4))
    
    for i, N in enumerate([1, 2, 5], 1):
        plt.subplot(1, 3, i)
        endpoints = Endpoints(10000, N, 1).flatten()
        plt.hist(endpoints, bins=50, density=True, alpha=0.6, color='g', label='Simulated')
        
        # Theoretical Gaussian
        sigma = np.sqrt(N / 12.0)
        x = np.linspace(-3.0 * sigma, 3.0 * sigma, 100)
        gauss = (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-x**2 / (2 * sigma**2))
        plt.plot(x, gauss, 'r-', linewidth=2, label='Theory')
        plt.title(f"N = {N}")
        plt.legend()
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    demo()
