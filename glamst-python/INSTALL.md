# Installation Guide for GLaMST Python

## Prerequisites

- Python 3.9 or higher
- pip (Python package installer)

## Installation Steps

### 1. Create a virtual environment (recommended)

```bash
cd glamst-python
python -m venv venv

# Activate on Linux/Mac:
source venv/bin/activate

# Activate on Windows:
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install GLaMST in development mode

```bash
pip install -e .
```

This installs the package in "editable" mode, meaning changes to the source code
will be immediately reflected without reinstalling.

### 4. Verify installation

```bash
glamst --version
```

You should see: `GLaMST 0.1.0 (Python)`

## Quick Test

Run the example script:

```bash
cd examples
python simple_example.py
```

Or try the command line interface:

```bash
glamst ../demodata/real.fasta --output-prefix test_output
```

## Running Tests

```bash
cd glamst-python
pytest tests/ -v
```

## Troubleshooting

### Import errors

If you get import errors, make sure you activated the virtual environment and
ran `pip install -e .` from the glamst-python directory.

### Missing dependencies

If specific packages are missing:

```bash
pip install --upgrade -r requirements.txt
```

### Numba compilation issues

Numba requires a C compiler. If you encounter issues:

- **Linux**: Install gcc (`sudo apt-get install build-essential`)
- **Mac**: Install Xcode Command Line Tools (`xcode-select --install`)
- **Windows**: Install Microsoft Visual C++ Build Tools

### Performance issues

For better performance, ensure you have:
- NumPy with Intel MKL (comes with Anaconda)
- Latest version of numba (`pip install --upgrade numba`)

## Development Installation

For development with additional tools:

```bash
pip install -e ".[dev]"
```

This installs additional packages like pytest and matplotlib.
