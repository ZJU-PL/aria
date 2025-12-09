"""Matrix operations and Howell form algorithms for affine relation domains.

This module implements the fundamental matrix operations and Howell form
algorithms described in Elder et al.'s "Abstract Domains of Affine Relations".
"""

import numpy as np
from typing import Tuple, List, Optional
from fractions import Fraction


class Matrix:
    """Represents a matrix over integers with support for Howell form operations."""

    def __init__(self, data: np.ndarray, modulus: int = 0):
        """Initialize a matrix.

        Args:
            data: 2D numpy array representing the matrix
            modulus: Modulus for modular arithmetic (0 means no modulus)
        """
        self.data = np.array(data, dtype=object)
        self.modulus = modulus
        self.rows, self.cols = self.data.shape

    def copy(self) -> 'Matrix':
        """Create a copy of this matrix."""
        return Matrix(self.data.copy(), self.modulus)

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __repr__(self) -> str:
        return f"Matrix({self.data}, modulus={self.modulus})"

    def row_echelon_form(self) -> 'Matrix':
        """Convert matrix to row echelon form (not full Howell form)."""
        mat = self.copy()
        rows, cols = mat.rows, mat.cols

        # Forward elimination
        for i in range(min(rows, cols)):
            # Find pivot
            pivot_row = i
            for j in range(i + 1, rows):
                if abs(mat[j, i]) > abs(mat[pivot_row, i]):
                    pivot_row = j

            # Swap rows if needed
            if pivot_row != i:
                mat.data[i], mat.data[pivot_row] = mat.data[pivot_row].copy(), mat.data[i].copy()

            # Eliminate below
            if mat[i, i] != 0:
                for j in range(i + 1, rows):
                    if mat[j, i] != 0:
                        factor = mat[j, i] / mat[i, i]
                        for k in range(i, cols):
                            mat[j, k] -= factor * mat[i, k]

        return mat

    def leading_value(self, row: int) -> int:
        """Get the leftmost nonzero value in a row vector."""
        for j in range(self.cols):
            if self[row, j] != 0:
                return self[row, j]
        return 0

    def leading_index(self, row: int) -> int:
        """Get the leading index of a row vector (column index of leftmost nonzero)."""
        for j in range(self.cols):
            if self[row, j] != 0:
                return j
        return self.cols


def howellize(matrix: Matrix) -> Matrix:
    """Put a matrix in Howell form using Algorithm 1 from the paper.

    Args:
        matrix: Input matrix to convert to Howell form

    Returns:
        Matrix in Howell form
    """
    mat = matrix.copy()
    w = matrix.modulus.bit_length() - 1  # Word size from modulus
    j = 0  # Number of already-Howellized rows

    # Algorithm 1: Howellize
    for i in range(1, min(2 * w + 1, mat.cols)):
        # Find all rows with leading index i
        R = []
        for r in range(mat.rows):
            if mat.leading_index(r) == i:
                R.append(r)

        if not R:
            continue

        # Pick row r in R that minimizes rank(r_i)
        r = min(R, key=lambda r: _compute_rank(mat[r, i], w))

        # Pick the odd u and p such that u * 2^p = r_i (with u odd, 1 <= u < 2^w)
        u, p = _find_odd_power_two(mat[r, i], w)

        # For each other row s in R, s != r:
        for s in R:
            if s == r:
                continue

            # Pick odd v and q such that v * 2^q = s_i
            v, q = _find_odd_power_two(mat[s, i], w)

            # Compute adjustment factor
            if u != 0:
                factor = (v * (2**q)) // u
                # Adjust row s: s = s - factor * r
                for col in range(mat.cols):
                    mat[s, col] = (mat[s, col] - factor * mat[r, col]) % mat.modulus

        # Zero out entries above r_i for rows already in Howell form
        for h in range(j):
            if h < mat.rows and mat[h, i] != 0:
                # Compute d = floor(G_h,i / r_i)
                d = mat[h, i] // mat[r, i]
                # G_h = G_h - d * r
                for col in range(mat.cols):
                    mat[h, col] = (mat[h, col] - d * mat[r, col]) % mat.modulus

        # If r_i != 1, add logical consequence 2^w - r_i as new row
        if mat[r, i] != 1:
            # Add new row with leading index > i
            new_row = [0] * mat.cols
            new_row[i] = (mat.modulus - mat[r, i]) % mat.modulus
            # Find appropriate leading index for the new row
            mat.data = np.vstack([mat.data, new_row])

        # Swap r with position j+1
        if j < mat.rows - 1:
            mat.data[r], mat.data[j] = mat.data[j].copy(), mat.data[r].copy()

        j += 1

    return mat


def _compute_rank(value: int, w: int) -> int:
    """Compute the rank of a value (number of trailing zeros in binary representation)."""
    if value == 0:
        return w  # Maximum rank for zero

    rank = 0
    val = value % (2**w)  # Work modulo 2^w
    while val % 2 == 0 and val > 0:
        val >>= 1  # Divide by 2
        rank += 1

    return rank


def _find_odd_power_two(value: int, w: int) -> Tuple[int, int]:
    """Find odd u and p such that u * 2^p = value mod 2^w, with 1 <= u < 2^w and u odd."""
    if value == 0:
        return 0, 0

    p = 0
    val = value % (2**w)  # Work modulo 2^w

    # Count trailing zeros (the power of 2)
    while val % 2 == 0 and val > 0:
        val >>= 1
        p += 1

    # If val is 0 after removing factors of 2, then value was pure power of 2
    if val == 0:
        return 1, p  # Represent as 1 * 2^p

    # val is now odd, ensure it's in range [1, 2^w - 1]
    if val >= 2**w:
        val >>= 1  # Make it smaller, but this changes the representation
        p += 1

    return val, p


def make_explicit(matrix: Matrix) -> Matrix:
    """Transform an AG matrix G in Howell form to near-explicit form.

    Algorithm 2 from the paper.
    """
    G = matrix.copy()
    k = G.rows
    w = 32  # Word size

    for i in range(2, k + 1):
        if G.leading_index(i-1) == i:
            # Row i-1 has leading index i
            rank_ri = G[i-1, i]
            if rank_ri > 0:
                for j in range(1, 2*k + 1):
                    s_i = 1
                    r_j = G[j-1, i] / rank_ri
                    # Build s from r_j with s_i = 1
                    # This needs more detailed implementation

    # Insert all-zero rows for skipped indexes
    for i in range(2, k + 1):
        if G.leading_index(i-1) != i:
            # Insert as the i'th row of G, a new row of all zeros
            pass

    return G
