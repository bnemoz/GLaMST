"""
Graph utility functions for tree manipulation.

This module provides functions for working with adjacency matrices and graphs,
including connected component extraction and reachability analysis.
"""

import numpy as np
from typing import List, Tuple
import networkx as nx
from scipy.sparse import csr_matrix


def extract_connected_components(adj_matrix: np.ndarray) -> Tuple[np.ndarray, List[int], List[int], np.ndarray]:
    """
    Extract connected components from adjacency matrix.

    This is the Python equivalent of extract_connected_component.m

    Args:
        adj_matrix: Adjacency matrix (can be directed or undirected)

    Returns:
        adj_matrix_subset: Adjacency matrix of largest component
        kept_indices: Indices of nodes in the largest component
        component_sizes: Sizes of all components
        components: Binary matrix where each column is a component indicator
    """
    adj_matrix = np.array(adj_matrix, dtype=float)

    # Make symmetric (undirected) for component detection
    adj_undirected = (np.abs(adj_matrix) + np.abs(adj_matrix.T)) > 0
    adj_undirected = adj_undirected.astype(float)

    # Add self-loops
    adj_undirected = adj_undirected - np.diag(np.diag(adj_undirected)) + np.eye(len(adj_undirected))

    # Find connected components using iterative propagation
    n = adj_matrix.shape[0]
    components = []
    is_assigned = np.zeros(n, dtype=bool)

    while not np.all(is_assigned):
        # Start with an unassigned node
        start_node = np.where(~is_assigned)[0][0]
        component = np.zeros(n, dtype=bool)
        component[start_node] = True

        # Propagate to find all connected nodes
        prev_component = np.zeros(n, dtype=bool)
        while not np.array_equal(component, prev_component):
            prev_component = component.copy()
            component = (adj_undirected @ component.astype(float)) > 0

        components.append(component)
        is_assigned[component] = True

    # Convert to matrix format
    components_matrix = np.column_stack(components)
    component_sizes = components_matrix.sum(axis=0)

    # Get largest component
    largest_idx = np.argmax(component_sizes)
    kept_indices = np.where(components_matrix[:, largest_idx])[0].tolist()

    # Extract submatrix
    adj_matrix_subset = adj_matrix[np.ix_(kept_indices, kept_indices)]

    return adj_matrix_subset, kept_indices, component_sizes.tolist(), components_matrix


def find_all_back_reachable_nodes(adj: np.ndarray, end_nodes: List[int]) -> List[int]:
    """
    Find all nodes reachable by traversing backwards from end_nodes.

    This function finds all ancestors of the given end nodes in a directed graph.
    Corresponds to find_all_back_reachable_nodes in reconstruct_tree_minimun_tree_size.m

    Args:
        adj: Directed adjacency matrix (adj[i,j] = 1 means edge from i to j)
        end_nodes: List of end node indices

    Returns:
        List of all reachable node indices (sorted)
    """
    reachable = set()
    queue = list(end_nodes)

    while queue:
        node = queue.pop(0)

        if node in reachable:
            continue

        reachable.add(node)

        # Find parents (nodes with edges TO current node)
        parents = np.where(adj[:, node] > 0)[0].tolist()
        queue.extend(parents)

    return sorted(list(reachable))


def compute_connected_components_fast(adj: np.ndarray) -> Tuple[List[List[int]], np.ndarray]:
    """
    Fast connected component computation using NetworkX.

    Args:
        adj: Adjacency matrix

    Returns:
        components: List of lists, each containing node indices in a component
        component_matrix: Binary matrix where each column represents a component
    """
    # Make undirected
    adj_sym = (adj + adj.T) > 0

    # Convert to NetworkX graph
    G = nx.from_numpy_array(adj_sym)

    # Find connected components
    components = [list(c) for c in nx.connected_components(G)]

    # Create component matrix
    n = adj.shape[0]
    component_matrix = np.zeros((n, len(components)), dtype=bool)
    for i, comp in enumerate(components):
        component_matrix[comp, i] = True

    return components, component_matrix


def get_parent_vector(directed_adj: np.ndarray) -> np.ndarray:
    """
    Convert directed adjacency matrix to parent vector.

    For a tree represented as directed_adj[i,j] = 1 meaning edge from i to j,
    returns a parent vector where parent[j] = i.

    Args:
        directed_adj: Directed adjacency matrix

    Returns:
        parent: Array where parent[i] is the parent of node i (0 for root)
    """
    n = directed_adj.shape[0]
    parent = np.zeros(n, dtype=int)

    for child in range(n):
        parents = np.where(directed_adj[:, child] > 0)[0]
        if len(parents) > 0:
            parent[child] = parents[0]  # Assuming tree structure (single parent)
        else:
            parent[child] = 0  # Root or disconnected

    return parent


def compute_leaf_distances(directed_adj: np.ndarray) -> np.ndarray:
    """
    Compute distance from each node to its furthest leaf descendant.

    Corresponds to leavedist.m

    Args:
        directed_adj: Directed adjacency matrix

    Returns:
        distances: Array of distances to furthest leaf for each node
    """
    n = directed_adj.shape[0]
    distances = np.zeros(n)

    # Find leaves (nodes with no children)
    has_children = directed_adj.sum(axis=1) > 0

    # Process nodes in reverse topological order (leaves first)
    processed = np.zeros(n, dtype=bool)

    while not np.all(processed):
        # Find nodes whose children are all processed
        for node in range(n):
            if processed[node]:
                continue

            children = np.where(directed_adj[node, :] > 0)[0]

            if len(children) == 0:
                # Leaf node
                distances[node] = 0
                processed[node] = True
            elif np.all(processed[children]):
                # All children processed
                distances[node] = 1 + np.max(distances[children])
                processed[node] = True

    return distances
