"""
Newick format tree output.

This module provides functions for converting tree structures to Newick format,
a standard format for representing phylogenetic trees.
"""

import numpy as np
from typing import List, Optional


def adjacency_to_newick(adj: np.ndarray,
                        node_ids: Optional[List[str]] = None,
                        root_id: int = 0,
                        dist_to_parent: int = 0) -> str:
    """
    Convert adjacency matrix to Newick format string.

    This is the Python equivalent of newick.m

    Args:
        adj: Directed adjacency matrix (parent to child)
        node_ids: Optional list of node identifiers
        root_id: Index of root node
        dist_to_parent: Distance to parent (for branch length)

    Returns:
        Newick format string
    """
    n = adj.shape[0]

    # Generate default node IDs if not provided
    if node_ids is None:
        node_ids = [str(i) for i in range(n)]

    # Find children of current node
    children = np.where(adj[root_id, :] > 0)[0]

    if len(children) == 0:
        # Leaf node
        return f"{node_ids[root_id]}:{dist_to_parent + 1}"

    elif len(children) == 1:
        # Single child, collapse this node
        return adjacency_to_newick(adj, node_ids, children[0], dist_to_parent + 1)

    else:
        # Internal node with multiple children
        child_strings = []
        for child in children:
            child_str = adjacency_to_newick(adj, node_ids, child, 0)
            child_strings.append(child_str)

        return f"({','.join(child_strings)})"


def write_newick(filename: str,
                adj: np.ndarray,
                node_ids: Optional[List[str]] = None,
                root_id: int = 0):
    """
    Write tree to Newick format file.

    Args:
        filename: Output filename
        adj: Directed adjacency matrix
        node_ids: Optional list of node identifiers
        root_id: Index of root node
    """
    newick_str = adjacency_to_newick(adj, node_ids, root_id, 0)

    with open(filename, 'w') as f:
        f.write(newick_str)

    print(f"Newick tree written to {filename}")


def create_node_labels(n_nodes: int,
                       is_observed: np.ndarray) -> List[str]:
    """
    Create node labels for tree visualization.

    Args:
        n_nodes: Total number of nodes
        is_observed: Boolean array indicating observed nodes

    Returns:
        List of node labels
    """
    labels = []
    obs_counter = 1
    ins_counter = 1

    for i in range(n_nodes):
        if i == 0:
            labels.append("G.L.")
        elif is_observed[i]:
            labels.append(f"Obs{obs_counter}")
            obs_counter += 1
        else:
            labels.append(f"Inferred{ins_counter}")
            ins_counter += 1

    return labels
