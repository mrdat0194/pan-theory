# Connection between Datastructure/Probability Lessons and SMAC

This note outlines how the mathematical concepts learned in these lessons form the foundation of the algorithms implemented in the **Statistical Mechanics: Algorithms & Computations (SMAC)** repository.

---

## 1. Bernoulli Process $\to$ Direct Sampling
*   **Concept:** A Bernoulli process models independent trials with binary outcomes (success or failure).
*   **SMAC Connection:** It is the foundation of **Direct Monte Carlo**. For example:
    *   **Direct $\pi$ estimation** (`direct_pi.py`): Randomly placing dots in a square and checking if they fall inside a circle is a sequence of Bernoulli trials.
    *   **Buffon's Needle** (`direct_needle.py`): Dropping needles to check if they cross cracks.

## 2. Markov Process & Random Walk $\to$ MCMC Simulations
*   **Concept:** A Random Walk is a path consisting of a succession of random steps. A Markov Process generalizes this, where the next state depends only on the current state.
*   **SMAC Connection:** This is the core of **Markov Chain Monte Carlo (MCMC)**:
    *   **Pebble Game** (`pebble_basic.py`): A local random walk on a 3x3 grid.
    *   **Metropolis Algorithms** (`markov_ising.py`): Random walks through spin configurations of the Ising model, accepting or rejecting moves to sample according to the Boltzmann distribution.

## 3. Random Matrix Theory $\to$ Transition Matrix Eigenvalues
*   **Concept:** Studying the eigenvalues and eigenvectors of matrices with random or structured entries.
*   **SMAC Connection:**
    *   **Transfer Matrices** (`pebble_transfer.py`): The pebble game random walk can be represented as a transition probability matrix $P$.
    *   **Mixing Time:** The speed at which the Markov chain converges to its uniform stationary distribution is governed by the eigenvalues of $P$. Specifically, the second largest eigenvalue ($\lambda_2$) determines the convergence rate (mixing time $\propto 1 / (1 - \lambda_2)$).

## 4. Stirling's Approximation $\to$ Statistical Entropy & Combinatorics
*   **Concept:** Approximating factorials for large numbers ($n! \approx \sqrt{2\pi n}(n/e)^n$).
*   **SMAC Connection:**
    *   **Ising model state spaces** (`enumerate_ising.py`): Estimating the number of possible spin configurations ($2^N$) and the entropy of states.
    *   **Permutation Cycles** (`permutation.py`): Computing cycle probabilities of large permutations, which is essential for path-integral simulations of quantum Bosons.
