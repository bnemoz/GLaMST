"""
FASTA file I/O operations.

This module provides functions for reading and writing FASTA format files
for sequence data.
"""

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from typing import List, Optional, Tuple
import numpy as np


def read_fasta(filename: str) -> List[str]:
    """
    Read sequences from FASTA file.

    Args:
        filename: Path to FASTA file

    Returns:
        List of sequences as strings
    """
    sequences = []
    for record in SeqIO.parse(filename, "fasta"):
        sequences.append(str(record.seq))
    return sequences


def read_fasta_with_headers(filename: str) -> Tuple[List[str], List[str]]:
    """
    Read sequences and headers from FASTA file.

    Args:
        filename: Path to FASTA file

    Returns:
        Tuple of (headers, sequences)
    """
    headers = []
    sequences = []
    for record in SeqIO.parse(filename, "fasta"):
        headers.append(record.id)
        sequences.append(str(record.seq))
    return headers, sequences


def write_fasta(filename: str,
                sequences: List[str],
                is_observed: np.ndarray,
                labels: Optional[List[str]] = None,
                original_mapping: Optional[dict] = None):
    """
    Write sequences to FASTA file.

    Corresponds to the FASTA writing part of Write_tree_toFile.m

    Args:
        filename: Output FASTA filename
        sequences: List of sequences
        is_observed: Boolean array indicating which sequences were observed
        labels: Optional custom labels for each sequence
        original_mapping: Optional mapping from reconstructed to original sequence names
    """
    records = []

    obs_counter = 1
    ins_counter = 1

    for i, seq in enumerate(sequences):
        # Determine label
        if labels and i < len(labels):
            label = labels[i]
        elif i == 0:
            label = "G.L."
        elif is_observed[i]:
            label = f"Obs{obs_counter}"
            obs_counter += 1
        else:
            label = f"index_{ins_counter}"
            ins_counter += 1

        # Add original sequence names if available
        if original_mapping and i in original_mapping:
            original_names = original_mapping[i]
            if isinstance(original_names, list):
                label = label + "\t" + "\t".join(f'"{name}"' for name in original_names)
            else:
                label = label + f'\t"{original_names}"'

        # Create SeqRecord
        record = SeqRecord(
            Seq(seq),
            id=label,
            description=""
        )
        records.append(record)

    # Write to file
    with open(filename, 'w') as f:
        SeqIO.write(records, f, "fasta")


def write_tree_structure(filename: str,
                        directed_adj: np.ndarray):
    """
    Write tree structure to file.

    Corresponds to the tree file output in Write_tree_toFile.m and main.m

    Each line contains: parent_id child_id1 child_id2 ...

    Args:
        filename: Output tree filename
        directed_adj: Directed adjacency matrix (parent to child)
    """
    with open(filename, 'w') as f:
        for parent in range(directed_adj.shape[0]):
            children = np.where(directed_adj[parent, :] > 0)[0]
            if len(children) > 0:
                # Write parent and all children
                line = f"{parent}"
                for child in children:
                    line += f" {child}"
                f.write(line + "\n")


def write_all_outputs(prefix: str,
                     sequences: List[str],
                     is_observed: np.ndarray,
                     directed_adj: np.ndarray,
                     labels: Optional[List[str]] = None):
    """
    Write all output files (FASTA, tree structure).

    Args:
        prefix: Output file prefix
        sequences: List of all sequences (observed + inferred)
        is_observed: Boolean array indicating observed sequences
        directed_adj: Directed adjacency matrix
        labels: Optional node labels
    """
    # Write FASTA
    write_fasta(f"{prefix}.out.fa", sequences, is_observed, labels)

    # Write tree structure
    write_tree_structure(f"{prefix}.out.tree", directed_adj)

    print(f"Output files written:")
    print(f"  {prefix}.out.fa - Sequences (observed + inferred)")
    print(f"  {prefix}.out.tree - Tree structure")
