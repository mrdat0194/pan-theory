"""
mst_msa_interaction.py
=======================
Demonstrates the interaction between Kruskal's Minimum Spanning Tree (MST)
and Multi-Perturbation Shapley Analysis (MSA).

1. Kruskal's MST serves as the constructive graph engine.
2. MSA evaluates the game-theoretic Shapley value / importance of each node 
   to the global spanning efficiency and structural stability.
"""

import itertools
import math
from typing import List, Tuple, Set, Dict


# =====================================================================
# 1. Classical Kruskal's MST Algorithm (The Constructive Engine)
# =====================================================================
class DisjointSet:
    def __init__(self, vertices):
        self.parent = {v: v for v in vertices}
        self.rank = {v: 0 for v in vertices}

    def find(self, item):
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return False
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        return True


def kruskal_mst(vertices: Set[int], edges: List[Tuple[int, int, float]]) -> Tuple[List[Tuple[int, int, float]], float, bool]:
    """
    Runs Kruskal's Algorithm on a subset of active vertices and edges.
    Returns: (mst_edges, total_cost, is_fully_connected)
    """
    if len(vertices) <= 1:
        return [], 0.0, True

    # Filter edges where both endpoints belong to active vertices
    active_edges = [e for e in edges if e[0] in vertices and e[1] in vertices]
    sorted_edges = sorted(active_edges, key=lambda x: x[2])

    ds = DisjointSet(vertices)
    mst_edges = []
    total_cost = 0.0

    for u, v, w in sorted_edges:
        if ds.union(u, v):
            mst_edges.append((u, v, w))
            total_cost += w
            if len(mst_edges) == len(vertices) - 1:
                break

    is_connected = (len(mst_edges) == len(vertices) - 1)
    return mst_edges, total_cost, is_connected


# =====================================================================
# 2. Game Payoff Function (The Game Objective)
# =====================================================================
def mst_efficiency_game(coalition: Tuple[int, ...], all_edges: List[Tuple[int, int, float]]) -> float:
    """
    Payoff v(S) for active coalition S:
    Measures the cost-efficiency of connecting the surviving nodes via MST.
    """
    active_vertices = set(coalition)
    if len(active_vertices) <= 1:
        return 0.0

    mst_edges, total_cost, is_connected = kruskal_mst(active_vertices, all_edges)
    
    if not is_connected or total_cost == 0:
        return 0.0  # Penalize network fragmentation / disconnected graphs

    # Payoff: Spanning connections formed per unit cost
    num_spanned_connections = len(active_vertices) - 1
    return num_spanned_connections / total_cost


# =====================================================================
# 3. Multi-Perturbation Shapley Attribution (The Analytical Framework)
# =====================================================================
def compute_msa_shapley(all_nodes: List[int], all_edges: List[Tuple[int, int, float]]) -> Dict[int, float]:
    """
    Computes exact Shapley values across all 2^N perturbation coalitions.
    """
    n = len(all_nodes)
    shapley_values = {node: 0.0 for node in all_nodes}

    for node in all_nodes:
        other_nodes = [v for v in all_nodes if v != node]
        
        # Test marginal contribution of 'node' across all possible subsets of other nodes
        for k in range(len(other_nodes) + 1):
            weight = (math.factorial(k) * math.factorial(n - k - 1)) / math.factorial(n)
            
            for subset in itertools.combinations(other_nodes, k):
                # Payoff without the node (lesioned state)
                v_without = mst_efficiency_game(subset, all_edges)
                
                # Payoff with the node included (restored state)
                subset_with_node = subset + (node,)
                v_with = mst_efficiency_game(subset_with_node, all_edges)
                
                marginal_contribution = v_with - v_without
                shapley_values[node] += weight * marginal_contribution

    return shapley_values


# =====================================================================
# 4. Execution & Ranking Output
# =====================================================================
def main():
    # Define a graph: (u, v, weight)
    nodes = [0, 1, 2, 3, 4]
    edges = [
        (0, 1, 2.0),
        (0, 2, 8.0),
        (1, 2, 3.0),
        (1, 3, 1.0),  # Node 1 acts as a critical bridge
        (3, 4, 2.0),
        (2, 4, 9.0),
    ]

    print("=========================================================")
    print("1. KRUSKAL'S MINIMUM SPANNING TREE (MST) CONSTRUCTIVE RESULT")
    print("=========================================================")
    mst_edges, total_cost, is_connected = kruskal_mst(set(nodes), edges)
    print(f"Total MST Weight/Cost: {total_cost}")
    print(f"Graph Fully Connected: {is_connected}")
    print("Selected MST Edges:")
    for u, v, w in mst_edges:
        print(f"  Edge ({u}) <--- weight {w:.1f} ---> ({v})")

    print("\n=========================================================")
    print("2. MULTI-PERTURBATION SHAPLEY ANALYSIS (MSA) RANKING")
    print("=========================================================")
    shapley_results = compute_msa_shapley(nodes, edges)
    
    # Sort nodes by game-theoretic importance
    sorted_nodes = sorted(shapley_results.items(), key=lambda x: x[1], reverse=True)
    
    print("Node Importance Rankings (by Shapley Contribution):")
    for rank, (node_id, phi) in enumerate(sorted_nodes, start=1):
        print(f"  Rank #{rank}: Node {node_id} | Shapley Value = {phi:+.4f}")

    print("\nInterpretation:")
    top_node, top_val = sorted_nodes[0]
    print(f"- Node {top_node} is Ranked #1 because it acts as the primary topological bridge.")
    print("- Lesioning/perturbing Node 1 causes the largest network efficiency drop.")


if __name__ == "__main__":
    main()
