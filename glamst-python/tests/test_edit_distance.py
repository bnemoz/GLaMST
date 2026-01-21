"""
Tests for edit distance functions.
"""

import pytest
import numpy as np
from glamst.core.edit_distance import edit_distance_only, edit_distance_all


def test_edit_distance_identical():
    """Test edit distance between identical sequences."""
    seq = "ATCG"
    dist, matrix = edit_distance_only(seq, seq)
    assert dist == 0


def test_edit_distance_one_mutation():
    """Test edit distance with one mutation."""
    dist, matrix = edit_distance_only("ATCG", "ATGG")
    assert dist == 1


def test_edit_distance_one_insertion():
    """Test edit distance with one insertion."""
    dist, matrix = edit_distance_only("ATCG", "ATCCG")
    assert dist == 1


def test_edit_distance_one_deletion():
    """Test edit distance with one deletion."""
    dist, matrix = edit_distance_only("ATCG", "ATG")
    assert dist == 1


def test_edit_distance_empty():
    """Test edit distance with empty string."""
    dist, matrix = edit_distance_only("", "ATG")
    assert dist == 3

    dist, matrix = edit_distance_only("ATG", "")
    assert dist == 3


def test_edit_distance_matrix_shape():
    """Test that distance matrix has correct shape."""
    str1 = "ATCG"
    str2 = "ATG"
    dist, matrix = edit_distance_only(str1, str2)
    assert matrix.shape == (len(str1) + 1, len(str2) + 1)


def test_edit_distance_all():
    """Test edit distance with operations extraction."""
    str1 = "ATCG"
    str2 = "ATGG"
    dist, matrix, ops, weights = edit_distance_all(str1, str2)

    assert dist == 1
    assert len(ops) > 0
    assert len(ops) == len(weights)


def test_edit_distance_symmetric():
    """Test that edit distance is symmetric."""
    str1 = "ATCGGTA"
    str2 = "ATCGAA"

    dist1, _ = edit_distance_only(str1, str2)
    dist2, _ = edit_distance_only(str2, str1)

    assert dist1 == dist2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
