import numpy as np
import scipy.linalg as la
import math

def simulate_bernoulli_steps(n_steps, p=0.5):
    """
    1. BERNOULLI PROCESS:
    Simulates a sequence of independent binary trials (left vs. right steps).
    """
    # 1 represents step right (+1), 0 represents step left (-1)
    trials = np.random.binomial(n=1, p=p, size=n_steps)
    steps = np.where(trials == 1, 1, -1)
    return steps

def run_random_walk(steps, ring_size=10):
    """
    2. RANDOM WALK (on a periodic ring of size L):
    Integrates the Bernoulli steps to track the position of a particle on a ring.
    """
    positions = []
    current_pos = 0
    for step in steps:
        current_pos = (current_pos + step) % ring_size
        positions.append(current_pos)
    return np.array(positions)

def build_markov_transition_matrix(ring_size=10, p=0.5):
    """
    3. MARKOV PROCESS:
    Constructs the doubly stochastic transition probability matrix P for the walk.
    P[i, j] represents the transition probability from state i to state j.
    """
    P = np.zeros((ring_size, ring_size))
    for i in range(ring_size):
        P[i, (i + 1) % ring_size] = p      # Probability of stepping right
        P[i, (i - 1) % ring_size] = 1 - p  # Probability of stepping left
    return P

def analyze_spectral_properties(P):
    """
    4. MATRIX SPECTRAL ANALYSIS:
    Analyzes the eigenvalues of the transition matrix P.
    In MCMC, the second largest eigenvalue controls the convergence (mixing) rate.
    """
    eigenvalues = la.eigvals(P)
    # Sort by magnitude (absolute value) in descending order
    sorted_eigenvalues = sorted(eigenvalues, key=abs, reverse=True)
    
    print("\n--- Eigenvalue Analysis (Spectral Properties) ---")
    print(f"Largest Eigenvalue (lambda_1): {sorted_eigenvalues[0]:.4f} (Must be exactly 1.0 for stationary distribution)")
    print(f"Second Largest Eigenvalue (lambda_2): {sorted_eigenvalues[1]:.4f}")
    
    # Calculate spectral gap and relaxation/mixing time
    spectral_gap = 1.0 - abs(sorted_eigenvalues[1])
    if spectral_gap > 1e-10:
        mixing_time = 1.0 / spectral_gap
        print(f"Spectral Gap (1 - |lambda_2|): {spectral_gap:.4f}")
        print(f"Theoretical Mixing Time (steps to converge): {mixing_time:.2f} steps")
    else:
        print("Spectral Gap is 0 (Chain does not converge or is periodic)")

def compare_stirling_combinations(L, N):
    """
    5. STIRLING'S APPROXIMATION:
    Estimates the number of possible states when putting N indistinguishable particles
    on a ring of size L. Using the combination formula: (L + N - 1)! / (N! * (L - 1)!)
    and comparing it with Stirling's approximation: n! ≈ sqrt(2*pi*n) * (n/e)^n
    """
    def stirling_factorial(n):
        if n == 0 or n == 1:
            return 1.0
        return math.sqrt(2 * math.pi * n) * ((n / math.e) ** n)
        
    n_states = math.comb(L + N - 1, N)
    
    # Estimate n! using Stirling
    numerator = stirling_factorial(L + N - 1)
    denominator = stirling_factorial(N) * stirling_factorial(L - 1)
    stirling_states = numerator / denominator
    
    percent_error = abs(stirling_states - n_states) / n_states * 100
    
    print("\n--- Stirling's Approximation & Particle Combinatorics ---")
    print(f"Placing N={N} indistinguishable particles on a ring of size L={L}:")
    print(f"  - Exact state space size: {n_states:,}")
    print(f"  - Stirling's approximation: {stirling_states:,.2f}")
    print(f"  - Approximation error: {percent_error:.4f}%")

def map_grid_neighbors(shape, index, periodic=False):
    """
    6. GRID COORDINATE MAPPING & BOUNDARY CONDITIONS:
    Translates a 1D flat index into a d-dimensional coordinate tuple,
    identifies adjacent neighbors by stepping along each axis, applies
    boundary conditions (periodic wrapping vs. open boundaries), and
    converts coordinates back to 1D flat indices.
    """
    coords = np.unravel_index(index, shape)
    neighbors = []
    
    # Iterate through all dimensions (axes) of the coordinate space
    for dim in range(len(coords)):
        val = coords[dim]
        # Step left (-1) and step right (+1)
        for step in [-1, 1]:
            target = val + step
            if periodic:
                # Wrap coordinates using modulo arithmetic
                target_wrapped = target % shape[dim]
                neighbor_coords = tuple(
                    target_wrapped if k == dim else coords[k]
                    for k in range(len(coords))
                )
                neighbor_flat = np.ravel_multi_index(neighbor_coords, shape)
                neighbors.append(neighbor_flat)
            else:
                # Reject steps that fall out of bounds
                if 0 <= target < shape[dim]:
                    neighbor_coords = tuple(
                        target if k == dim else coords[k]
                        for k in range(len(coords))
                    )
                    neighbor_flat = np.ravel_multi_index(neighbor_coords, shape)
                    neighbors.append(neighbor_flat)
    return coords, neighbors

if __name__ == '__main__':
    print("=" * 65)
    print("INTEGRATING BERNOULLI, RANDOM WALK, MARKOV, LINALG, STIRLING & NEIGHBORS")
    print("=" * 65)
    
    # Parameters
    L = 10       # Size of the ring
    N_steps = 20 # Number of steps for simulation
    
    # 1. Bernoulli steps
    steps = simulate_bernoulli_steps(N_steps, p=0.5)
    print("1. Bernoulli Steps (+1=Right, -1=Left):")
    print("  ", steps)
    
    # 2. Random Walk
    positions = run_random_walk(steps, ring_size=L)
    print("\n2. Random Walk Positions on Ring (size L=10):")
    print("  ", positions)
    
    # 3. Transition Matrix
    P = build_markov_transition_matrix(ring_size=L, p=0.5)
    
    # 4. Spectral properties of the Transition Matrix
    analyze_spectral_properties(P)
    
    # 5. Stirling's approximation
    # If we put 15 Bosons on a ring of size 10, what is the state space size?
    compare_stirling_combinations(L=10, N=15)
    
    # 6. Grid Coordinate Mapping & Boundary Conditions (Ising Model Style)
    shape_2d = (3, 3)
    target_site = 0
    print("\n--- Grid Coordinate Mapping & Boundary Conditions ---")
    print(f"Lattice Shape: {shape_2d}")
    print(f"Target Site Flat Index: {target_site}")
    
    # Open boundaries
    coords_open, nbrs_open = map_grid_neighbors(shape_2d, target_site, periodic=False)
    print(f"  - Open Boundaries: Coordinates {coords_open} -> Neighbors: {nbrs_open}")
    
    # Periodic boundaries
    coords_per, nbrs_per = map_grid_neighbors(shape_2d, target_site, periodic=True)
    print(f"  - Periodic Boundaries: Coordinates {coords_per} -> Neighbors: {nbrs_per}")
