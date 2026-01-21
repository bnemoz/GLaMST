"""
Edit distance computation for sequence alignment.

This module provides functions to compute edit distance (Levenshtein distance)
between sequences and extract all optimal edit operations.
"""

import numpy as np
from numba import jit
from typing import List, Tuple, Dict
from collections import defaultdict
import Levenshtein


def edit_distance_only(str1: str, str2: str) -> Tuple[int, np.ndarray]:
    """
    Compute edit distance and matrix between two sequences.

    This is the Python equivalent of EditDistance_only.m / editDist_only.cpp

    Args:
        str1: First sequence
        str2: Second sequence

    Returns:
        distance: Minimum edit distance
        matrix: Dynamic programming matrix (m+1 x n+1)
    """
    # Use fast C implementation for just the distance
    distance = Levenshtein.distance(str1, str2)

    # Compute full DP matrix for traceback (needed by other functions)
    matrix = _compute_edit_distance_matrix(str1, str2)

    return distance, matrix


@jit(nopython=True, cache=True)
def _compute_edit_distance_matrix(str1: str, str2: str) -> np.ndarray:
    """
    Numba-accelerated DP matrix computation.

    Args:
        str1: First sequence
        str2: Second sequence

    Returns:
        dp: Dynamic programming matrix of shape (m+1, n+1)
    """
    m, n = len(str1), len(str2)
    dp = np.zeros((m + 1, n + 1), dtype=np.float64)

    # Initialize first row and column
    for i in range(m + 1):
        dp[i, 0] = i
    for j in range(n + 1):
        dp[0, j] = j

    # Fill the matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i-1] == str2[j-1]:
                dp[i, j] = dp[i-1, j-1]
            else:
                dp[i, j] = 1 + min(
                    dp[i, j-1],      # Insert
                    dp[i-1, j],      # Delete
                    dp[i-1, j-1]     # Replace
                )

    return dp


def edit_distance_all(str1: str, str2: str) -> Tuple[int, np.ndarray, List[str], np.ndarray]:
    """
    Compute edit distance and all unique operations with weights.

    This is the Python equivalent of EditDistance_all_fastest.m

    Args:
        str1: First sequence
        str2: Second sequence

    Returns:
        distance: Minimum edit distance
        matrix: DP matrix
        unique_operations: List of unique operation strings
        operation_weights: Weight of each unique operation (importance)
    """
    dist, matrix = edit_distance_only(str1, str2)

    # Build graph of all optimal paths
    nodes, adj = _build_optimal_path_graph(str1, str2, matrix)

    # Extract unique operations and their weights
    unique_ops, weights = _extract_operations_and_weights(str1, str2, nodes, adj)

    return dist, matrix, unique_ops, weights


def _build_optimal_path_graph(str1: str, str2: str, V: np.ndarray) -> Tuple[List[Tuple[int, int]], Dict]:
    """
    Build graph containing all optimal edit paths.

    This corresponds to the graph building section in EditDistance_all_fastest.m
    (lines 42-106)

    Args:
        str1: First sequence
        str2: Second sequence
        V: DP matrix from edit_distance_only

    Returns:
        nodes: List of (i, j) positions in the DP matrix
        adj: Adjacency list representation (dict from node_idx to list of child node_idx)
    """
    m, n = len(str1), len(str2)

    # Start from bottom-right corner
    nodes = [(m, n)]
    node_to_idx = {(m, n): 0}
    adj = defaultdict(list)  # adjacency list (forward direction)

    queue = [(m, n)]

    while queue:
        pos = queue.pop(0)
        i, j = pos
        current_idx = node_to_idx[pos]

        if i == 0 and j == 0:
            continue

        # No change (match)
        if i > 0 and j > 0 and V[i, j] == V[i-1, j-1] and str1[i-1] == str2[j-1]:
            prev_pos = (i-1, j-1)
            _add_edge(nodes, node_to_idx, adj, queue, prev_pos, current_idx)

        # Mutation (characters differ)
        if i > 0 and j > 0 and V[i, j] == V[i-1, j-1] + 1:
            prev_pos = (i-1, j-1)
            _add_edge(nodes, node_to_idx, adj, queue, prev_pos, current_idx)

        # Deletion from str1
        if i > 0 and V[i, j] == V[i-1, j] + 1:
            prev_pos = (i-1, j)
            _add_edge(nodes, node_to_idx, adj, queue, prev_pos, current_idx)

        # Insertion to str1
        if j > 0 and V[i, j] == V[i, j-1] + 1:
            prev_pos = (i, j-1)
            _add_edge(nodes, node_to_idx, adj, queue, prev_pos, current_idx)

    return nodes, adj


def _add_edge(nodes: List[Tuple[int, int]],
              node_to_idx: Dict,
              adj: Dict,
              queue: List,
              prev_pos: Tuple[int, int],
              curr_idx: int):
    """Helper to add edge to graph."""
    if prev_pos not in node_to_idx:
        node_to_idx[prev_pos] = len(nodes)
        nodes.append(prev_pos)
        queue.append(prev_pos)

    prev_idx = node_to_idx[prev_pos]
    adj[prev_idx].append(curr_idx)


def _extract_operations_and_weights(str1: str,
                                    str2: str,
                                    nodes: List[Tuple[int, int]],
                                    adj: Dict) -> Tuple[List[str], np.ndarray]:
    """
    Extract unique operations and their weights from the path graph.

    Corresponds to lines 122-160 in EditDistance_all_fastest.m

    Args:
        str1: First sequence
        str2: Second sequence
        nodes: List of positions in DP matrix
        adj: Adjacency list of the path graph

    Returns:
        unique_operations: List of unique operation strings
        weights: Array of weights (importance) for each operation
    """
    # Find start and end nodes
    start_node = nodes.index((0, 0))
    end_node = nodes.index((len(str1), len(str2)))

    # Count total number of paths
    total_paths = _count_paths(adj, start_node, end_node, len(nodes))

    # Extract all operations and their importance
    operations = []
    operation_weights = []

    for prev_idx, next_indices in adj.items():
        prev_state = nodes[prev_idx]

        for next_idx in next_indices:
            curr_state = nodes[next_idx]
            i_prev, j_prev = prev_state
            i_curr, j_curr = curr_state

            # Skip if it's a match (no operation)
            if (i_curr == i_prev + 1 and j_curr == j_prev + 1 and
                str1[i_curr-1] == str2[j_curr-1]):
                continue

            # Determine the operation type
            operation = None
            if i_curr == i_prev + 1 and j_curr == j_prev + 1:
                # Mutation
                operation = f"mutate positioin {i_curr} to {str2[j_curr-1]}"
            elif i_curr == i_prev + 1 and j_curr == j_prev:
                # Deletion
                operation = f"delete positioin {i_curr}"
            elif i_curr == i_prev and j_curr == j_prev + 1:
                # Insertion
                operation = f"insert {str2[j_curr-1]} after positioin {i_prev}"

            if operation:
                operations.append(operation)

                # Calculate weight: how many paths use this edge
                # Remove this edge and count remaining paths
                adj_copy = {k: v[:] for k, v in adj.items()}
                adj_copy[prev_idx].remove(next_idx)
                paths_without = _count_paths(adj_copy, start_node, end_node, len(nodes))
                weight = 1 - (paths_without / total_paths if total_paths > 0 else 0)
                operation_weights.append(weight)

    # Aggregate weights for unique operations
    unique_operations = list(set(operations))
    unique_weights = np.zeros(len(unique_operations))

    for i, unique_op in enumerate(unique_operations):
        # Sum weights for all occurrences of this operation
        for j, op in enumerate(operations):
            if op == unique_op:
                unique_weights[i] += operation_weights[j]

    return unique_operations, unique_weights


def _count_paths(adj: Dict, start: int, end: int, num_nodes: int) -> int:
    """
    Count number of paths from start to end in a DAG.

    Corresponds to find_num_path function in EditDistance_all_fastest.m (lines 164-194)

    Args:
        adj: Adjacency list
        start: Start node index
        end: End node index
        num_nodes: Total number of nodes

    Returns:
        Number of paths from start to end
    """
    num_paths = np.zeros(num_nodes, dtype=np.int64)
    num_paths[start] = 1

    nodes_updated = [start]

    while nodes_updated:
        nodes_to_update = []

        # Find all nodes reachable from recently updated nodes
        for node in nodes_updated:
            if node in adj:
                nodes_to_update.extend(adj[node])

        nodes_to_update = list(set(nodes_to_update))
        nodes_updated = []

        # Update path counts
        for node in nodes_to_update:
            new_count = sum(num_paths[parent] for parent in range(num_nodes)
                          if node in adj.get(parent, []))

            if num_paths[node] != new_count:
                num_paths[node] = new_count
                nodes_updated.append(node)

    return num_paths[end]
