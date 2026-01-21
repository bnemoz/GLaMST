"""
Command-line interface for GLaMST.

This module provides the main entry point for running GLaMST from the command line.
"""

import argparse
import sys
from pathlib import Path

from .core.tree_reconstruction import reconstruct_tree
from .io.fasta import read_fasta, write_all_outputs
from .io.newick import write_newick, create_node_labels


def main():
    """Main entry point for GLaMST CLI."""
    parser = argparse.ArgumentParser(
        description="GLaMST - Grow Lineages along Minimum Spanning Tree",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  glamst input.fasta
  glamst input.fasta --no-rewire
  glamst input.fasta --output-prefix results/output

Note: The first sequence in the FASTA file must be the root/germline sequence.
"""
    )

    parser.add_argument(
        "fasta_file",
        help="Input FASTA file (first sequence must be root/germline)"
    )

    parser.add_argument(
        "--no-rewire",
        action="store_true",
        help="Disable rewiring optimization (faster but may result in larger tree)"
    )

    parser.add_argument(
        "--output-prefix",
        help="Output file prefix (default: same as input filename)",
        default=None
    )

    parser.add_argument(
        "--version",
        action="version",
        version="GLaMST 0.1.0 (Python)"
    )

    args = parser.parse_args()

    # Check if input file exists
    input_path = Path(args.fasta_file)
    if not input_path.exists():
        print(f"Error: Input file '{args.fasta_file}' not found", file=sys.stderr)
        sys.exit(1)

    # Determine output prefix
    if args.output_prefix:
        output_prefix = args.output_prefix
    else:
        output_prefix = str(input_path)

    print("=" * 70)
    print("GLaMST - Grow Lineages along Minimum Spanning Tree")
    print("=" * 70)
    print(f"Input file: {args.fasta_file}")
    print(f"Output prefix: {output_prefix}")
    print(f"Rewiring: {'disabled' if args.no_rewire else 'enabled'}")
    print("=" * 70)

    try:
        # Read input sequences
        print("\nReading input sequences...")
        sequences = read_fasta(args.fasta_file)

        if len(sequences) == 0:
            print("Error: No sequences found in input file", file=sys.stderr)
            sys.exit(1)

        print(f"  Loaded {len(sequences)} sequences")

        if len(sequences) < 2:
            print("Error: At least 2 sequences required (root + observations)", file=sys.stderr)
            sys.exit(1)

        # Reconstruct tree
        print("\nReconstructing tree...")
        all_sequences, mst_adj, is_observed, directed_adj = reconstruct_tree(
            sequences,
            rewire=not args.no_rewire
        )

        # Create node labels
        node_labels = create_node_labels(len(all_sequences), is_observed)

        # Write outputs
        print("\nWriting output files...")
        write_all_outputs(
            output_prefix,
            all_sequences,
            is_observed,
            directed_adj,
            node_labels
        )

        # Write Newick format
        write_newick(
            f"{output_prefix}.out.newick",
            directed_adj,
            node_labels,
            root_id=0
        )

        # Print summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Input sequences: {len(sequences)}")
        print(f"Total reconstructed nodes: {len(all_sequences)}")
        print(f"  Observed: {sum(is_observed)}")
        print(f"  Inferred: {len(all_sequences) - sum(is_observed)}")
        print("\nOutput files:")
        print(f"  {output_prefix}.out.fa - All sequences")
        print(f"  {output_prefix}.out.tree - Tree structure")
        print(f"  {output_prefix}.out.newick - Newick format")
        print("=" * 70)
        print("\nDone!")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
