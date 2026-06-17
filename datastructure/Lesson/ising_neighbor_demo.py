import numpy as np

def explain_neighbor_mapping(index, shape=(4, 5), periodic=False):
    """
    Explains step-by-step how a 1D flat index is converted to coordinates,
    stepped in all dimensions, boundary-checked, and converted back.
    """
    print("-" * 70)
    print(f"Analyzing neighbors of flat index {index} on a grid of shape {shape}")
    print(f"Boundary Conditions: {'PERIODIC (Torus)' if periodic else 'OPEN (Bounded Walls)'}")
    print("-" * 70)

    # 1. UNRAVEL: Convert 1D flat index into a d-dimensional coordinate tuple
    coordinates = np.unravel_index(index, shape)
    print(f"Step 1: Unravel flat index {index} -> Coordinates: {coordinates}")
    
    neighbors = []

    # 2. STEPPING: Iterate through each dimension (axis)
    for dim_axis in range(len(coordinates)):
        current_coord_val = coordinates[dim_axis]
        print(f"\n  Checking Dimension/Axis {dim_axis} (Current coordinate value: {current_coord_val})")
        
        # Check both directions along this axis: step left (-1) and step right (+1)
        for target_val in [current_coord_val - 1, current_coord_val + 1]:
            print(f"    Attempting step to value {target_val}: ", end="")
            
            is_in_bounds = (target_val >= 0 and target_val < shape[dim_axis])
            
            if periodic or is_in_bounds:
                # 3. BOUNDARY HANDLING: 
                # If periodic, wrap around using modulo arithmetic
                if periodic:
                    wrapped_val = target_val % shape[dim_axis]
                    status = f"Wrapped to {wrapped_val}"
                else:
                    wrapped_val = target_val
                    status = "Valid step (in-bounds)"

                # Build the new neighbor coordinate tuple by replacing only the active dimension's coordinate
                neighbor_coords = tuple(
                    wrapped_val if k == dim_axis else coordinates[k] 
                    for k in range(len(coordinates))
                )
                
                # 4. RAVEL: Convert the coordinate tuple back to a 1D flat index
                neighbor_flat_index = np.ravel_multi_index(neighbor_coords, shape)
                print(f"{status} -> Coords: {neighbor_coords} -> Ravel index: {neighbor_flat_index}")
                neighbors.append(neighbor_flat_index)
            else:
                print("REJECTED (Out of bounds)")
                
    return neighbors

def build_neighbours_table(shape=(4, 5), periodic=False):
    """
    Mimics the Neighbours class in ising.py.
    Constructs a fixed table where each row contains the neighbors of a node,
    using -1 to represent omitted boundary edges (for open boundary conditions).
    """
    N = np.prod(shape)
    # Maximum neighbors in d-dimensional space is 2 * d (2 directions per axis)
    max_nbrs = 2 * len(shape)
    
    # Initialize table with -1
    table = -1 * np.ones((N, max_nbrs), dtype=int)
    
    for i in range(N):
        # We manually collect neighbors for state i
        coordinates = np.unravel_index(i, shape)
        col_idx = 0
        for dim in range(len(coordinates)):
            for offset in [-1, 1]:
                target = coordinates[dim] + offset
                if periodic:
                    target = target % shape[dim]
                    nbr_coords = tuple(target if k == dim else coordinates[k] for k in range(len(coordinates)))
                    table[i, col_idx] = np.ravel_multi_index(nbr_coords, shape)
                    col_idx += 1
                else:
                    if 0 <= target < shape[dim]:
                        nbr_coords = tuple(target if k == dim else coordinates[k] for k in range(len(coordinates)))
                        table[i, col_idx] = np.ravel_multi_index(nbr_coords, shape)
                        col_idx += 1
                        
    return table

if __name__ == '__main__':
    # =========================================================================
    # Demonstration 1: 2D Grid of shape (3, 3)
    # =========================================================================
    print("DEMO 1: 2D GRID NEIGHBORS (Shape: 3x3)")
    
    # Node 0 (Top-left corner) - Open Boundaries vs. Periodic
    explain_neighbor_mapping(index=0, shape=(3, 3), periodic=False)
    explain_neighbor_mapping(index=0, shape=(3, 3), periodic=True)
    
    # Node 4 (Center site)
    explain_neighbor_mapping(index=4, shape=(3, 3), periodic=False)

    # =========================================================================
    # Demonstration 2: 3D Grid of shape (2, 2, 2)
    # =========================================================================
    print("\n" + "="*80)
    print("DEMO 2: 3D GRID NEIGHBORS (Shape: 2x2x2)")
    print("="*80)
    explain_neighbor_mapping(index=3, shape=(2, 2, 2), periodic=False)

    # =========================================================================
    # Demonstration 3: Neighbor Table Generation (Ising Model Style)
    # =========================================================================
    print("\n" + "="*80)
    print("DEMO 3: COMPLETE ISING NEIGHBOR TABLE (Shape: 3x3, Open Boundaries)")
    print("="*80)
    table = build_neighbours_table(shape=(3, 3), periodic=False)
    print("Rows represent flat indices (0 to 8). Columns represent neighbors (padded with -1):")
    for idx, row in enumerate(table):
        print(f"  Site {idx} Neighbors: {row}")
