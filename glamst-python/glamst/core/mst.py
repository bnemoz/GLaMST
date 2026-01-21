"""
Minimum Spanning Tree construction from distance matrices.

This module provides functions for building MST from pairwise distance matrices,
optionally with seed edges that must be included.
"""

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
from typing import Tuple, Optional
import sys

sys.path.append('..')
from ..graph.utils import extract_connected_components


def mst_from_dist_matrix(dist_matrix: np.ndarray,
                         seed_adj: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Construct minimum spanning tree from distance matrix.

    This is the Python equivalent of mst_from_dist_matrix.m

    The algorithm builds an MST while respecting pre-existing edges in seed_adj.
    It uses a modified version of Prim's/Kruskal's algorithm that treats
    existing components as super-nodes.

    Args:
        dist_matrix: Pairwise distance matrix (n x n)
        seed_adj: Optional seed adjacency matrix (edges to preserve)

    Returns:
        adj: MST adjacency matrix (binary, symmetric)
        adj_weighted: MST with edge weights (symmetric)
        cost: Total MST cost (sum of edge weights)
    """
    n = dist_matrix.shape[0]

    # Make diagonal large to avoid self-loops
    dist_adj = dist_matrix.copy()
    np.fill_diagonal(dist_adj, dist_matrix.max() * 2)

    # Initialize with seed edges if provided
    if seed_adj is not None and np.any(seed_adj):
        adj, adj_weighted, cost, components, active_components = \
            _initialize_from_seed(seed_adj, dist_adj)
    else:
        adj = np.zeros((n, n))
        adj_weighted = np.zeros((n, n))
        cost = 0.0
        components = np.zeros((n, 0))
        active_components = np.zeros(0, dtype=bool)

    # Find nodes not yet in any component
    if components.size > 0:
        in_component = np.any(components[:, active_components], axis=1)
    else:
        in_component = np.zeros(n, dtype=bool)

    unassigned = np.where(~in_component)[0]

    # Add each unassigned node to the tree
    for node in unassigned:
        # Find closest node
        distances = dist_adj[node, :]
        closest = np.argmin(distances)
        min_dist = distances[closest]

        # Add edge
        if adj[node, closest] == 0 and adj[closest, node] == 0:
            adj[node, closest] = 1
            adj[closest, node] = 1

            if min_dist == 0:
                min_dist = 1e-10

            adj_weighted[node, closest] = min_dist
            adj_weighted[closest, node] = min_dist
            cost += min_dist

        # Update components
        components, active_components = _update_components(
            components, active_components, node, closest, n
        )

    # Merge remaining components
    while np.sum(active_components) > 1:
        # Find smallest component
        component_sizes = components[:, active_components].sum(axis=0)
        smallest_idx = np.argmin(component_sizes)
        active_idx = np.where(active_components)[0][smallest_idx]

        # Get nodes in smallest component
        comp1_nodes = np.where(components[:, active_idx])[0]

        # Get nodes not in this component
        comp2_nodes = np.where(~components[:, active_idx])[0]

        # Find minimum distance edge between components
        submatrix = dist_adj[np.ix_(comp1_nodes, comp2_nodes)]
        min_idx = np.argmin(submatrix.ravel())
        i = comp1_nodes[min_idx // len(comp2_nodes)]
        j = comp2_nodes[min_idx % len(comp2_nodes)]
        min_dist = dist_adj[i, j]

        # Add edge
        adj[i, j] = 1
        adj[j, i] = 1

        if min_dist == 0:
            min_dist = 1e-10

        adj_weighted[i, j] = min_dist
        adj_weighted[j, i] = min_dist
        cost += min_dist

        # Merge components
        comp2_idx = None
        for idx in np.where(active_components)[0]:
            if idx != active_idx and components[j, idx]:
                comp2_idx = idx
                break

        if comp2_idx is not None:
            # Merge the two components
            new_component = components[:, active_idx] | components[:, comp2_idx]
            components = np.column_stack([components, new_component])
            active_components = np.append(active_components, True)
            active_components[active_idx] = False
            active_components[comp2_idx] = False
        else:
            # j was not in any component, add to active component
            components[j, active_idx] = True

    return adj, adj_weighted, cost


def _initialize_from_seed(seed_adj: np.ndarray,
                         dist_adj: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Initialize MST construction with seed edges.

    Args:
        seed_adj: Seed adjacency matrix
        dist_adj: Distance matrix

    Returns:
        adj: Initial adjacency matrix
        adj_weighted: Initial weighted adjacency matrix
        cost: Initial cost
        components: Component indicator matrix
        active_components: Boolean array of active components
    """
    n = seed_adj.shape[0]

    # Start with seed adjacency
    adj = seed_adj.copy()
    adj_weighted = seed_adj * dist_adj
    cost = adj_weighted.sum() / 2  # Divide by 2 because symmetric

    # Extract connected components from seed
    _, _, component_sizes, components = extract_connected_components(seed_adj)

    # Remove singleton components
    component_sizes = np.array(component_sizes)
    non_singleton = component_sizes > 1
    components = components[:, non_singleton]
    active_components = np.ones(components.shape[1], dtype=bool)

    return adj, adj_weighted, cost, components, active_components


def _update_components(components: np.ndarray,
                      active_components: np.ndarray,
                      node1: int,
                      node2: int,
                      n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Update component membership after adding an edge.

    Args:
        components: Component indicator matrix
        active_components: Boolean array of active components
        node1: First node of new edge
        node2: Second node of new edge
        n: Total number of nodes

    Returns:
        Updated components and active_components
    """
    if components.size == 0:
        # No components yet, create first one
        components = np.zeros((n, 1), dtype=bool)
        components[[node1, node2], 0] = True
        active_components = np.array([True])
        return components, active_components

    # Find which components contain node1 and node2
    comp1_indices = np.where(components[node1, :] & active_components)[0]
    comp2_indices = np.where(components[node2, :] & active_components)[0]

    if len(comp1_indices) == 0 and len(comp2_indices) == 0:
        # Neither node in a component, create new one
        new_comp = np.zeros(n, dtype=bool)
        new_comp[[node1, node2]] = True
        components = np.column_stack([components, new_comp])
        active_components = np.append(active_components, True)

    elif len(comp1_indices) > 0 and len(comp2_indices) == 0:
        # Only node1 in a component, add node2
        components[node2, comp1_indices[0]] = True

    elif len(comp1_indices) == 0 and len(comp2_indices) > 0:
        # Only node2 in a component, add node1
        components[node1, comp2_indices[0]] = True

    elif comp1_indices[0] != comp2_indices[0]:
        # Both in different components, merge them
        comp1_idx = comp1_indices[0]
        comp2_idx = comp2_indices[0]
        merged = components[:, comp1_idx] | components[:, comp2_idx]
        components = np.column_stack([components, merged])
        active_components = np.append(active_components, True)
        active_components[comp1_idx] = False
        active_components[comp2_idx] = False

    # else: both already in same component, nothing to do

    return components, active_components
