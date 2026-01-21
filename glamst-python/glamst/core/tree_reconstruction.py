"""
Core tree reconstruction algorithm.

This module implements the main GLaMST algorithm for reconstructing phylogenetic
lineage trees from observed sequence data.
"""

import numpy as np
from typing import List, Tuple
from tqdm import tqdm
import re

from .edit_distance import edit_distance_only, edit_distance_all
from .mst import mst_from_dist_matrix
from ..graph.utils import find_all_back_reachable_nodes


def reconstruct_tree(observed_sequences: List[str],
                     rewire: bool = True) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    """
    Main tree reconstruction function.

    This is the Python equivalent of reconstruct_tree_minimun_tree_size.m

    Args:
        observed_sequences: List of observed sequences (first MUST be root/germline)
        rewire: Whether to perform rewiring optimization

    Returns:
        all_sequences: All sequences (observed + inferred)
        mst_adj: MST adjacency matrix
        is_observed: Boolean array indicating observed sequences
        directed_adj: Directed tree adjacency matrix (parent to child)
    """
    # Remove duplicates (keep root separate)
    root = observed_sequences[0]
    unique_non_root = list(set(observed_sequences[1:]))
    observed_sequences = [root] + unique_non_root
    n_obs = len(observed_sequences)

    print("Reconstruction:")

    # Step 1: Compute pairwise edit distances
    print(f"  Computing pairwise distances for {n_obs} sequences...")
    pairwise_dist = _compute_pairwise_distances(observed_sequences)

    # Step 2: Build initial MST
    seed_adj = np.zeros_like(pairwise_dist)
    mst_adj, adj_weighted, cost = mst_from_dist_matrix(pairwise_dist, seed_adj)
    seed_adj = (adj_weighted > 0).astype(float)

    # Step 3: Iteratively build tree
    print("  Iteratively building tree...")
    all_sequences, is_observed, directed_adj, mst_adj, pairwise_dist = \
        _iterative_tree_building(observed_sequences, mst_adj, pairwise_dist, seed_adj)

    # Step 4: Trim unnecessary nodes
    print("  Trimming unnecessary nodes...")
    all_sequences, is_observed, directed_adj, mst_adj, pairwise_dist = \
        _trim_tree(all_sequences, is_observed, directed_adj, mst_adj, pairwise_dist)

    # Step 5: Optional rewiring
    if rewire:
        print(f"  Rewiring tree (starting size: {len(all_sequences)})...")
        all_sequences, is_observed, directed_adj, mst_adj, pairwise_dist = \
            _rewire_tree(all_sequences, is_observed, directed_adj, mst_adj, pairwise_dist)

    print(f"  Final tree: {len(all_sequences)} nodes ({sum(is_observed)} observed, "
          f"{len(all_sequences) - sum(is_observed)} inferred)")

    return all_sequences, mst_adj, is_observed, directed_adj


def _compute_pairwise_distances(sequences: List[str]) -> np.ndarray:
    """
    Compute pairwise edit distances between all sequences.

    Args:
        sequences: List of sequences

    Returns:
        Distance matrix (n x n)
    """
    n = len(sequences)
    dist_matrix = np.zeros((n, n))

    # Use tqdm for progress bar
    total = n * (n - 1) // 2
    with tqdm(total=total, desc="    Pairwise distances") as pbar:
        for i in range(n):
            for j in range(i + 1, n):
                dist, _ = edit_distance_only(sequences[i], sequences[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
                pbar.update(1)

    return dist_matrix


def _iterative_tree_building(observed_sequences: List[str],
                             mst_adj: np.ndarray,
                             pairwise_dist: np.ndarray,
                             seed_adj: np.ndarray) -> Tuple:
    """
    Iteratively build the tree by adding intermediate nodes.

    This corresponds to lines 26-145 in reconstruct_tree_minimun_tree_size.m

    Args:
        observed_sequences: List of observed sequences
        mst_adj: Initial MST adjacency
        pairwise_dist: Pairwise distance matrix
        seed_adj: Seed adjacency for MST

    Returns:
        Tuple of (all_sequences, is_observed, directed_adj, mst_adj, pairwise_dist)
    """
    all_sequences = observed_sequences[:]
    reconstructed_indicator = np.zeros(len(all_sequences), dtype=bool)
    reconstructed_indicator[0] = True  # Root is reconstructed
    is_observed = np.ones(len(all_sequences), dtype=bool)
    directed_adj = np.zeros((len(all_sequences), len(all_sequences)))

    iteration = 0
    max_iterations = len(observed_sequences) * 10  # Safety limit

    while not np.all(reconstructed_indicator) and iteration < max_iterations:
        iteration += 1

        # Find unreconstructed components
        tmp_adj = mst_adj.copy()
        tmp_adj[reconstructed_indicator, :] = 0
        tmp_adj[:, reconstructed_indicator] = 0

        # Get connected components of unreconstructed portion
        components = _get_components_to_process(tmp_adj, reconstructed_indicator, mst_adj)

        if len(components) == 0:
            break

        # For each component, find best operations
        tmp_operations, tmp_counts, tmp_parents = _find_best_operations(
            all_sequences, components, reconstructed_indicator, mst_adj
        )

        if len(tmp_operations) == 0:
            break

        # Pick the operation with highest count
        sorted_indices = np.argsort(tmp_counts)[::-1]

        # Try operations in order until we find one that creates a new node
        for idx in sorted_indices:
            best_parent = tmp_parents[idx]
            best_operation = tmp_operations[idx]

            # Apply operation to create new sequence
            new_sequence = _apply_operation(all_sequences[best_parent], best_operation)

            # Check if this sequence already exists
            if new_sequence in all_sequences[len(observed_sequences):]:
                continue
            if new_sequence in [all_sequences[i] for i in range(len(all_sequences)) if reconstructed_indicator[i]]:
                continue

            # Valid new sequence found
            break
        else:
            # All operations created duplicates, shouldn't happen but handle it
            break

        # Check if new sequence is in observed sequences
        if new_sequence in all_sequences:
            node_idx = all_sequences.index(new_sequence)
            reconstructed_indicator[node_idx] = True
            is_observed[node_idx] = True
        else:
            # Add new inferred sequence
            all_sequences.append(new_sequence)
            reconstructed_indicator = np.append(reconstructed_indicator, True)
            is_observed = np.append(is_observed, False)
            node_idx = len(all_sequences) - 1

            # Expand matrices
            n = len(all_sequences)
            new_pairwise = np.zeros((n, n))
            new_pairwise[:n-1, :n-1] = pairwise_dist
            pairwise_dist = new_pairwise

            # Compute distances to new sequence
            for i in range(n - 1):
                dist, _ = edit_distance_only(all_sequences[i], all_sequences[-1])
                pairwise_dist[i, n-1] = dist
                pairwise_dist[n-1, i] = dist

            # Expand other matrices
            new_directed = np.zeros((n, n))
            new_directed[:n-1, :n-1] = directed_adj
            directed_adj = new_directed

        # Add edge from parent to new node
        directed_adj[best_parent, node_idx] = 1
        directed_adj[node_idx, best_parent] = 0

        # Update seed adjacency and recompute MST
        n = len(all_sequences)
        new_seed = np.zeros((n, n))
        new_seed[:seed_adj.shape[0], :seed_adj.shape[1]] = seed_adj
        new_seed[best_parent, node_idx] = 1
        new_seed[node_idx, best_parent] = 1
        seed_adj = new_seed

        mst_adj, _, _ = mst_from_dist_matrix(pairwise_dist, seed_adj)

        if iteration % 10 == 0:
            print(f"    Iteration {iteration}: {np.sum(reconstructed_indicator)} / {len(observed_sequences)} "
                  f"({len(all_sequences)} total nodes)")

    return all_sequences, is_observed, directed_adj, mst_adj, pairwise_dist


def _get_components_to_process(tmp_adj: np.ndarray,
                               reconstructed: np.ndarray,
                               mst_adj: np.ndarray) -> List[np.ndarray]:
    """
    Get unreconstructed components that need processing.

    Args:
        tmp_adj: Adjacency with reconstructed nodes removed
        reconstructed: Boolean array of reconstructed nodes
        mst_adj: Full MST adjacency

    Returns:
        List of component indicator arrays
    """
    n = len(reconstructed)

    # Find components in tmp_adj
    components = []
    visited = np.zeros(n, dtype=bool)

    for start_node in range(n):
        if visited[start_node] or reconstructed[start_node]:
            continue

        if np.sum(tmp_adj[start_node, :]) == 0 and np.sum(tmp_adj[:, start_node]) == 0:
            continue

        # BFS to find component
        component = np.zeros(n, dtype=bool)
        queue = [start_node]
        component[start_node] = True
        visited[start_node] = True

        while queue:
            node = queue.pop(0)
            neighbors = np.where(tmp_adj[node, :] > 0)[0]
            for neighbor in neighbors:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    component[neighbor] = True
                    queue.append(neighbor)

        components.append(component)

    # Merge components with the same reconstructed parent
    merged_components = []
    parent_map = {}

    for comp in components:
        # Find parent(s) of this component
        comp_nodes = np.where(comp)[0]
        parents = set()
        for node in comp_nodes:
            parent_nodes = np.where((mst_adj[:, node] > 0) & reconstructed)[0]
            parents.update(parent_nodes.tolist())

        parent_key = tuple(sorted(parents))
        if parent_key in parent_map:
            parent_map[parent_key] = parent_map[parent_key] | comp
        else:
            parent_map[parent_key] = comp

    return list(parent_map.values())


def _find_best_operations(all_sequences: List[str],
                         components: List[np.ndarray],
                         reconstructed: np.ndarray,
                         mst_adj: np.ndarray) -> Tuple[List[str], List[int], List[int]]:
    """
    Find best operations for each component.

    Args:
        all_sequences: Current list of all sequences
        components: List of component indicators
        reconstructed: Boolean array of reconstructed nodes
        mst_adj: MST adjacency matrix

    Returns:
        Tuple of (operations, counts, parent_indices)
    """
    all_operations = []
    all_counts = []
    all_parents = []

    for component in components:
        # Find parent of this component
        comp_nodes = np.where(component)[0]
        parent_nodes = []
        for node in comp_nodes:
            parents = np.where((mst_adj[:, node] > 0) & reconstructed)[0]
            parent_nodes.extend(parents.tolist())
        parent_nodes = list(set(parent_nodes))

        if len(parent_nodes) == 0:
            continue

        parent = parent_nodes[0]  # Use first parent

        # Get all unique operations from parent to children in component
        operation_counts = {}
        for child in comp_nodes:
            _, _, unique_ops, weights = edit_distance_all(
                all_sequences[parent], all_sequences[child]
            )

            for op, weight in zip(unique_ops, weights):
                if op in operation_counts:
                    operation_counts[op] += weight
                else:
                    operation_counts[op] = weight

        # Add to lists
        for op, count in operation_counts.items():
            all_operations.append(op)
            all_counts.append(count)
            all_parents.append(parent)

    return all_operations, all_counts, all_parents


def _apply_operation(sequence: str, operation: str) -> str:
    """
    Apply an operation string to a sequence.

    Operations are formatted as:
    - "mutate positioin X to Y"
    - "delete positioin X"
    - "insert X after positioin Y"

    Args:
        sequence: Original sequence
        operation: Operation string

    Returns:
        Modified sequence
    """
    parts = operation.split()

    if parts[0] == "mutate":
        # "mutate positioin X to Y"
        position = int(parts[2])
        new_base = parts[4]
        seq_list = list(sequence)
        seq_list[position - 1] = new_base  # 1-indexed to 0-indexed
        return ''.join(seq_list)

    elif parts[0] == "delete":
        # "delete positioin X"
        position = int(parts[2])
        return sequence[:position-1] + sequence[position:]

    elif parts[0] == "insert":
        # "insert X after positioin Y"
        base = parts[1]
        position = int(parts[4])
        return sequence[:position] + base + sequence[position:]

    else:
        raise ValueError(f"Unknown operation: {operation}")


def _trim_tree(all_sequences: List[str],
              is_observed: np.ndarray,
              directed_adj: np.ndarray,
              mst_adj: np.ndarray,
              pairwise_dist: np.ndarray) -> Tuple:
    """
    Remove unnecessary nodes not on paths to observed nodes.

    Args:
        all_sequences: All sequences
        is_observed: Boolean array
        directed_adj: Directed adjacency
        mst_adj: MST adjacency
        pairwise_dist: Distance matrix

    Returns:
        Trimmed versions of all inputs
    """
    observed_indices = np.where(is_observed)[0].tolist()
    reachable = find_all_back_reachable_nodes(directed_adj, observed_indices)

    # Keep only reachable nodes
    all_sequences = [all_sequences[i] for i in reachable]
    is_observed = is_observed[reachable]
    directed_adj = directed_adj[np.ix_(reachable, reachable)]
    mst_adj = mst_adj[np.ix_(reachable, reachable)]
    pairwise_dist = pairwise_dist[np.ix_(reachable, reachable)]

    return all_sequences, is_observed, directed_adj, mst_adj, pairwise_dist


def _rewire_tree(all_sequences: List[str],
                is_observed: np.ndarray,
                directed_adj: np.ndarray,
                mst_adj: np.ndarray,
                pairwise_dist: np.ndarray) -> Tuple:
    """
    Rewire tree to further reduce size.

    This corresponds to lines 163-320 in reconstruct_tree_minimun_tree_size.m

    Args:
        all_sequences: All sequences
        is_observed: Boolean array
        directed_adj: Directed adjacency
        mst_adj: MST adjacency
        pairwise_dist: Distance matrix

    Returns:
        Optimized versions of all inputs
    """
    iteration = 0
    max_iterations = 1000

    while iteration < max_iterations:
        iteration += 1

        # Find nodes eligible for rewiring (observed + branching points)
        observed_nodes = np.where(is_observed)[0]
        branching_nodes = np.where(directed_adj.sum(axis=1) > 1)[0]
        nodes_to_rewire = np.union1d(observed_nodes, branching_nodes)

        if len(nodes_to_rewire) <= 1:
            break

        # Skip root
        nodes_to_rewire = nodes_to_rewire[nodes_to_rewire != 0]

        # Compute rewiring benefits
        benefits = _compute_rewiring_benefits(
            all_sequences, nodes_to_rewire, is_observed,
            directed_adj, pairwise_dist
        )

        # Check if any rewiring is beneficial
        if np.max(benefits) <= 0:
            break

        # Find best node to rewire
        best_idx = np.argmax(benefits)
        node_to_rewire = nodes_to_rewire[best_idx]

        # Find best rewiring destination
        destination = _find_rewire_destination(
            node_to_rewire, is_observed, directed_adj, pairwise_dist
        )

        if destination is None:
            break

        # Perform rewiring
        all_sequences, is_observed, directed_adj, mst_adj, pairwise_dist = \
            _perform_rewire(
                all_sequences, is_observed, directed_adj, mst_adj,
                pairwise_dist, node_to_rewire, destination
            )

        if iteration % 10 == 0:
            print(f"    Rewiring iteration {iteration}: {len(all_sequences)} nodes")

    return all_sequences, is_observed, directed_adj, mst_adj, pairwise_dist


def _compute_rewiring_benefits(all_sequences: List[str],
                               nodes_to_rewire: np.ndarray,
                               is_observed: np.ndarray,
                               directed_adj: np.ndarray,
                               pairwise_dist: np.ndarray) -> np.ndarray:
    """Compute benefit of rewiring each node."""
    benefits = np.zeros(len(nodes_to_rewire))

    # Build temporary adjacency without nodes_to_rewire edges
    tmp_adj = directed_adj.copy()
    tmp_adj[nodes_to_rewire, :] = 0

    for i, node in enumerate(nodes_to_rewire):
        # Compute cost of current wiring
        back_reachable = find_all_back_reachable_nodes(tmp_adj, [node])
        current_cost = len(back_reachable)

        if len(back_reachable) == 0:
            # Not connected to other rewire nodes
            parent = np.where(directed_adj[:, node] > 0)[0]
            if len(parent) > 0 and not is_observed[parent[0]]:
                current_cost = 0.1

        # Compute cost of best alternative wiring
        forward_reachable = find_all_back_reachable_nodes(directed_adj.T, [node])
        unqualified = set(back_reachable) | {node} | set(np.where(directed_adj[:, node] > 0)[0]) | set(forward_reachable)

        distances = pairwise_dist[node, :].copy()
        distances[list(unqualified)] = np.inf
        min_dist = np.min(distances)

        if min_dist < np.inf:
            new_cost = max(0, min_dist - 1)
            if new_cost == 0:
                # Check if destination is observed
                dest_candidates = np.where(distances == min_dist)[0]
                if not np.any(is_observed[dest_candidates]):
                    new_cost = 0.1
        else:
            new_cost = current_cost

        benefits[i] = current_cost - new_cost

    return benefits


def _find_rewire_destination(node: int,
                             is_observed: np.ndarray,
                             directed_adj: np.ndarray,
                             pairwise_dist: np.ndarray) -> int:
    """Find best destination for rewiring a node."""
    # Find nodes that are unqualified (would create cycles or redundancy)
    tmp_adj = directed_adj.copy()
    tmp_adj[node, :] = 0

    back_reachable = find_all_back_reachable_nodes(tmp_adj, [node])
    forward_reachable = find_all_back_reachable_nodes(directed_adj.T, [node])
    children = np.where(directed_adj[node, :] > 0)[0]

    unqualified = set(back_reachable) | {node} | set(children) | set(forward_reachable)

    # Find closest qualified node
    distances = pairwise_dist[node, :].copy()
    distances[list(unqualified)] = np.inf

    min_dist = np.min(distances)
    if np.isinf(min_dist):
        return None

    candidates = np.where(distances == min_dist)[0]

    # Prefer observed nodes
    observed_candidates = candidates[is_observed[candidates]]
    if len(observed_candidates) > 0:
        return observed_candidates[0]

    return candidates[0]


def _perform_rewire(all_sequences: List[str],
                   is_observed: np.ndarray,
                   directed_adj: np.ndarray,
                   mst_adj: np.ndarray,
                   pairwise_dist: np.ndarray,
                   node: int,
                   destination: int) -> Tuple:
    """Perform the rewiring operation."""
    # Remove old parent edge
    old_parent = np.where(directed_adj[:, node] > 0)[0]
    if len(old_parent) > 0:
        old_parent = old_parent[0]
        directed_adj[old_parent, node] = 0
        mst_adj[old_parent, node] = 0
        mst_adj[node, old_parent] = 0

    # Add edge to new destination
    dist = pairwise_dist[destination, node]

    if dist <= 1:
        # Direct connection
        directed_adj[destination, node] = 1
        mst_adj[destination, node] = 1
        mst_adj[node, destination] = 1
    else:
        # Need to add intermediate nodes
        # This is a simplified version - full implementation would add intermediate nodes
        directed_adj[destination, node] = 1
        mst_adj[destination, node] = 1
        mst_adj[node, destination] = 1

    # Trim unreachable nodes
    observed_indices = np.where(is_observed)[0].tolist()
    reachable = find_all_back_reachable_nodes(directed_adj, observed_indices)

    all_sequences = [all_sequences[i] for i in reachable]
    is_observed = is_observed[reachable]
    directed_adj = directed_adj[np.ix_(reachable, reachable)]
    mst_adj = mst_adj[np.ix_(reachable, reachable)]
    pairwise_dist = pairwise_dist[np.ix_(reachable, reachable)]

    return all_sequences, is_observed, directed_adj, mst_adj, pairwise_dist
