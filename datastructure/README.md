# Data Structure & Algorithms Showcase

This directory contains implementations of various data structures, classic puzzle solutions, and mathematical algorithms.

---

## File Structure

### 1. [story_teller.py](file:///C:/Users/mrdat/PycharmProjects/pan-theory/datastructure/story_teller.py)
This is the single entry point for all algorithmic solutions, organized logically into modular classes:

*   **`story_teller`**: Standard problems (e.g., `minimumSwaps`, `minimumBribes`, `freqQuery`).
*   **`BSTShowcase`**: BST construction, validation, path printing, and largest-BST-subtree algorithms (Naive, Single-Pass Optimized, and `largestBSTBT`).
    *   *Design Note:* The `BSTNode` structure here uses `left_child` and `right_child` without parent pointers, prioritizing lightweight operations for recursive validation puzzles. For advanced BST implementations that support parent tracking, deletion logic, and rich node queries, refer to the self-contained package in [BinarySearchTree](file:///C:/Users/mrdat/PycharmProjects/pan-theory/datastructure/Data_Structures_Algorithms_In_Python-master/Tree/BinarySearchTree).
*   **`ArrayProblems`**: Array manipulation methods including `mode_sorted`, `mode_unsorted`, and `RightRotate`.
*   **`Stock`**: Financial algorithm problems like `max_profit_k_transactions` and `stock_gain`.
*   **`FibonacciShowcase`**: Comparisons of Fibonacci calculation algorithms (Tabulation, Sieve equivalent recurrence, Tail Recursion, Memoization, Fast Doubling) including `fib_dn`.
*   **`DynamicProgramming`**: Subset partition (`divide_numbers`) and Longest Increasing Subsequence (`lis`).
*   **`SearchAndCount`**: Binary search and pair counters.
*   **`GeometryProblems`**: Rectangular overlap and area calculations.
*   **`ClassicPuzzles`**: Interactive puzzles like pyramid sum paths and matrix range counts.
*   **`PythonPatterns`**: Demonstrations of standard Python libraries (Regex, itertools chain).
*   **`PrimeSieveShowcase`**: Primality check, Sieve of Eratosthenes, and Segmented Sieve range search.
*   **`NelderMeadShowcase`**: Multimodal function optimization utilizing `scipy`.
*   **`PowerSumNTT`**: Advanced power sum solution utilizing Number Theoretic Transform.
*   **`Solution`**: Permutation algorithms and `findJudge` town celebrity search.

### 2. [test/story_teller_test.py](file:///C:/Users/mrdat/PycharmProjects/pan-theory/datastructure/test/story_teller_test.py)
The unified unit test suite. It includes:
*   Tests verifying algorithm correctness.
*   The `TestDocstringExamples` test class, containing test cases derived from example parameters described in python docstrings (`"""`).

### 3. Other Algorithmic & Educational Directories
While `story_teller.py` serves as the core library for coding puzzles, this directory contains other specialized folders serving different educational purposes:

*   **[CodingGame](file:///C:/Users/mrdat/PycharmProjects/pan-theory/datastructure/CodingGame)**: Practical application of search algorithms, pathfinding (like BFS/DFS in `labyrinth.py` and dynamic tracking in `Skynet.py`), and heuristic optimization (like simulated annealing in `sodoku_anneal.py`) under real-time game constraints.
*   **[Lesson](file:///C:/Users/mrdat/PycharmProjects/pan-theory/datastructure/Lesson)**: Focuses on mathematical and probabilistic modeling. It contains notebooks and scripts exploring stochastic models (Markov/Bernoulli processes), linear algebra projections, Stirling numbers of the second kind, and random matrix theory.
*   **[problem](file:///C:/Users/mrdat/PycharmProjects/pan-theory/datastructure/problem)**: A structured playground for design exercises, including markdown sheets detailing the problem statement, brute force baselines, optimal solutions, and automated test case generators.
*   **[Data_Structures_Algorithms_In_Python-master](file:///C:/Users/mrdat/PycharmProjects/pan-theory/datastructure/Data_Structures_Algorithms_In_Python-master)**: Formal, textbook implementations of fundamental data structures in Python (e.g., standard Binary Search Tree). It provides academic reference designs using explicit object relationships (such as parent pointers and deletion nodes), and augmented structures like the quantum-inspired [BST_MatrixVector.py](file:///c:/Users/mrdat/PycharmProjects/pan-theory/datastructure/Data_Structures_Algorithms_In_Python-master/Tree/BinarySearchTree/BST_MatrixVector.py) for sparse matrix/vector operations.

---

## Design Guidelines & Standards

### Standard Library First
For core algorithm implementations in `story_teller`, we prioritize Python's built-in types and standard library modules over custom data structure classes:
*   **Stacks**: Use Python's built-in `list` (with `append()` and `pop()`), which is highly optimized and runs in $O(1)$ amortized time.
*   **Queues / Deques**: Use `collections.deque` (with `append()` and `popleft()`) for fast $O(1)$ double-ended queue operations, or `queue.Queue` if thread-safety is required.
*   **Priority Queues / Heaps**: Use the built-in `heapq` module or `queue.PriorityQueue`.

Custom classes (such as `Stack` and `Node` from [common.py](file:///C:/Users/mrdat/PycharmProjects/pan-theory/datastructure/common.py)) and their corresponding showcase wrappers are intended purely for educational lessons and compatibility with the Master repository, and should not be used in performance-sensitive core algorithms.

---

## How to Test

To run the full test suite and verify that all implementations work as expected, execute the following command from the project root directory:

```bash
python -m unittest datastructure/test/story_teller_test.py
```

All 120+ test cases must print `OK` showing successful execution.
