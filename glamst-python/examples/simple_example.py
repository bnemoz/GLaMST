"""
Simple example of using GLaMST Python API.

This demonstrates how to use GLaMST programmatically rather than
through the command line.
"""

import numpy as np
from glamst import reconstruct_tree, read_fasta, write_all_outputs


def main():
    print("GLaMST Python API Example")
    print("=" * 70)

    # Option 1: Create sequences programmatically
    print("\n1. Using synthetic sequences:")
    sequences = [
        "ATCGATCGATCG",  # Root/germline (must be first)
        "ATCGATGGATCG",  # Observed sequence 1 (1 mutation)
        "ATCGATGGATCGG", # Observed sequence 2 (1 mutation + 1 insertion)
        "ATCGTTCGATCG",  # Observed sequence 3 (1 mutation)
    ]

    print(f"  Input: {len(sequences)} sequences")
    for i, seq in enumerate(sequences):
        print(f"    {i}: {seq}")

    # Reconstruct tree
    print("\n  Reconstructing tree...")
    all_sequences, mst_adj, is_observed, directed_adj = reconstruct_tree(
        sequences,
        rewire=True
    )

    print(f"\n  Results:")
    print(f"    Total nodes: {len(all_sequences)}")
    print(f"    Observed: {sum(is_observed)}")
    print(f"    Inferred: {len(all_sequences) - sum(is_observed)}")

    # Show all sequences
    print(f"\n  All sequences (observed + inferred):")
    for i, (seq, obs) in enumerate(zip(all_sequences, is_observed)):
        status = "observed" if obs else "inferred"
        print(f"    {i}: {seq} ({status})")

    # Show tree structure
    print(f"\n  Tree structure (parent -> children):")
    for parent in range(len(all_sequences)):
        children = np.where(directed_adj[parent, :] > 0)[0]
        if len(children) > 0:
            print(f"    {parent} -> {list(children)}")

    # Option 2: Load from FASTA file (if available)
    print("\n" + "=" * 70)
    print("2. Loading from FASTA file:")
    try:
        sequences = read_fasta("../demodata/real.fasta")
        print(f"  Loaded {len(sequences)} sequences")

        # Reconstruct (this will take longer)
        print("  Reconstructing tree (this may take a while)...")
        all_sequences, mst_adj, is_observed, directed_adj = reconstruct_tree(
            sequences,
            rewire=False  # Disable rewiring for faster processing
        )

        print(f"\n  Results:")
        print(f"    Total nodes: {len(all_sequences)}")
        print(f"    Observed: {sum(is_observed)}")
        print(f"    Inferred: {len(all_sequences) - sum(is_observed)}")

        # Write outputs
        write_all_outputs("example_output", all_sequences, is_observed, directed_adj)
        print("\n  Output files written with prefix 'example_output'")

    except FileNotFoundError:
        print("  Demo data file not found, skipping this example")

    print("\n" + "=" * 70)
    print("Done!")


if __name__ == "__main__":
    main()
