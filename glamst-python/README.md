# GLaMST - Python Implementation

Python port of GLaMST (Grow Lineages along Minimum Spanning Tree) for reconstructing phylogenetic lineage trees from B Cell Receptor sequence data.

## Installation

```bash
cd glamst-python
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Usage

```python
from glamst.core.tree_reconstruction import reconstruct_tree
from glamst.io.fasta import read_fasta, write_fasta

# Read sequences (first sequence must be root/germline)
sequences = read_fasta("input.fasta")

# Reconstruct tree
all_sequences, mst_adj, is_observed, directed_adj = reconstruct_tree(sequences, rewire=True)

# Save results
write_fasta("output.fa", all_sequences, is_observed)
```

Command line:
```bash
glamst input.fasta --output-prefix output
```

## Development Status

This is a work-in-progress port from the original MATLAB implementation.

## Original Implementation

See the parent directory for the original MATLAB implementation.
