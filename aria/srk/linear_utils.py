"""
Linear algebra utility functions and helpers.

This module provides utility functions for creating vectors and matrices,
solving linear systems, and other common operations.
"""

from __future__ import annotations
from typing import Dict, List, Set, Tuple, Optional, Union, Any
from fractions import Fraction

from aria.srk.linear import QQVector, QQMatrix, QQ


# Utility functions for creating vectors and matrices
def zero_vector(dimensions: int) -> QQVector:
    """Create a zero vector."""
    return QQVector()


def unit_vector(dim: int, size: int) -> QQVector:
    """Create a unit vector in the given dimension."""
    return QQVector({dim: QQ(1)})


def identity_matrix(size: int) -> QQMatrix:
    """Create an identity matrix."""
    rows = []
    for i in range(size):
        row = QQVector({i: QQ(1)})
        rows.append(row)
    return QQMatrix(rows)


def vector_from_list(values: List[Union[QQ, int]]) -> QQVector:
    """Create a vector from a list of values."""
    entries = {}
    for i, val in enumerate(values):
        if val != 0:
            entries[i] = QQ(val) if not isinstance(val, QQ) else val
    return QQVector(entries)


def matrix_from_lists(rows: List[List[QQ]]) -> QQMatrix:
    """Create a matrix from a list of row lists."""
    vector_rows = [vector_from_list(row) for row in rows]
    return QQMatrix(vector_rows)


def mk_vector(values: List[Union[QQ, int]]) -> QQVector:
    """Create a vector from a list of values."""
    return vector_from_list(values)


def mk_matrix(rows: List[List[Union[QQ, int]]]) -> QQMatrix:
    """Create a matrix from a list of row lists."""
    vector_rows = [vector_from_list(row) for row in rows]
    return QQMatrix(vector_rows)


# Linear term utilities
def linterm_of(srk_context, term) -> QQVector:
    """Convert a term to a linear term representation.

    Args:
        srk_context: SRK context
        term: Term to convert

    Returns:
        QQVector representing the linear term
    """
    from .syntax import (
        destruct,
        mk_const,
        mk_real,
        mk_add,
        mk_mul,
        Const,
        Var,
        Add,
        Mul,
    )

    def _linterm_of_rec(t, const_dim=0):
        """Recursive helper to convert term to linear form."""
        if isinstance(t, Const):
            # Constant term - add to constant dimension
            return QQVector.add_term(QQ.one, const_dim, QQVector.zero())
        elif isinstance(t, Var):
            # Variable term - add to its dimension
            return QQVector.add_term(QQ.one, t.var_id, QQVector.zero())
        elif isinstance(t, Add):
            # Sum of terms
            if not t.args:
                return QQVector.zero()
            result = _linterm_of_rec(t.args[0], const_dim)
            for arg in t.args[1:]:
                result = QQVector.add(result, _linterm_of_rec(arg, const_dim))
            return result
        elif isinstance(t, Mul):
            # Product - check if it's coefficient * variable
            if len(t.args) == 2:
                # Check if first arg is a constant
                const_part, var_part = None, None
                for arg in t.args:
                    if isinstance(arg, (Const, Var)) and hasattr(arg, "symbol"):
                        if isinstance(arg, Const) and arg.symbol.name in [
                            "0",
                            "1",
                            "2",
                            "3",
                            "4",
                            "5",
                            "6",
                            "7",
                            "8",
                            "9",
                        ]:
                            try:
                                const_part = QQ.of_string(arg.symbol.name)
                            except:
                                pass
                        else:
                            var_part = arg
                    elif isinstance(arg, Var):
                        var_part = arg

                if const_part is not None and var_part is not None:
                    # coefficient * variable
                    coeff_vec = QQVector.add_term(
                        const_part, var_part.var_id, QQVector.zero()
                    )
                    return coeff_vec

            # Fallback: treat as sum of linear terms
            if not t.args:
                return QQVector.zero()
            result = _linterm_of_rec(t.args[0], const_dim)
            for arg in t.args[1:]:
                result = QQVector.add(result, _linterm_of_rec(arg, const_dim))
            return result
        else:
            # For other term types, treat as constant 0 for now
            # This is a simplified implementation
            return QQVector.zero()

    return _linterm_of_rec(term)


# Linear algebra utility functions
def solve_linear_system(A: QQMatrix, b: QQVector) -> Optional[QQVector]:
    """Solve Ax = b for x using Gaussian elimination with back substitution.

    Args:
        A: Coefficient matrix (m x n)
        b: Right-hand side vector (m-dimensional)

    Returns:
        Solution vector x if it exists, None otherwise
    """
    if not A.rows:
        return QQVector() if b.is_zero() else None

    m = len(A.rows)  # Number of rows
    n = (
        max((max(row.dimensions()) for row in A.rows if row.dimensions()), default=0)
        + 1
    )  # Number of columns

    # Create augmented matrix [A | b]
    # We'll use column n as the augmented column
    augmented_rows = []
    for i, row in enumerate(A.rows):
        new_row = row.set(n, b.get(i, QQ(0)))
        augmented_rows.append(new_row)

    # Convert to mutable list for Gaussian elimination
    matrix = list(augmented_rows)

    # Forward elimination with partial pivoting
    pivot_row = 0
    for col in range(n):
        if pivot_row >= m:
            break

        # Find pivot (row with largest absolute value in current column)
        max_val = QQ(0)
        max_row = -1
        for row_idx in range(pivot_row, m):
            val = abs(matrix[row_idx].get(col, QQ(0)))
            if val > max_val:
                max_val = val
                max_row = row_idx

        if max_row == -1:
            # No pivot found, skip this column
            continue

        # Swap rows to bring pivot to current position
        if max_row != pivot_row:
            matrix[pivot_row], matrix[max_row] = matrix[max_row], matrix[pivot_row]

        # Get pivot coefficient
        pivot_coeff = matrix[pivot_row].get(col, QQ(0))
        if pivot_coeff == 0:
            continue

        # Normalize pivot row
        matrix[pivot_row] = matrix[pivot_row] * (QQ(1) / pivot_coeff)

        # Eliminate column in rows below
        for row_idx in range(pivot_row + 1, m):
            factor = matrix[row_idx].get(col, QQ(0))
            if factor != 0:
                matrix[row_idx] = matrix[row_idx] - (matrix[pivot_row] * factor)

        pivot_row += 1

    # Check for inconsistency (0 = non-zero)
    for row_idx in range(pivot_row, m):
        # Check if left side is zero but right side is non-zero
        left_zero = all(matrix[row_idx].get(c, QQ(0)) == 0 for c in range(n))
        right_nonzero = matrix[row_idx].get(n, QQ(0)) != 0
        if left_zero and right_nonzero:
            return None  # Inconsistent system

    # Back substitution
    solution_entries = {}

    # Process rows from bottom to top
    for row_idx in range(min(pivot_row, m) - 1, -1, -1):
        row = matrix[row_idx]

        # Find the leading variable (first non-zero column)
        leading_col = None
        for col in range(n):
            if row.get(col, QQ(0)) != 0:
                leading_col = col
                break

        if leading_col is None:
            continue

        # Compute value for this variable
        rhs = row.get(n, QQ(0))  # Right-hand side

        # Subtract known variables
        for col in range(leading_col + 1, n):
            coeff = row.get(col, QQ(0))
            if coeff != 0 and col in solution_entries:
                rhs -= coeff * solution_entries[col]

        # Solve for the leading variable
        leading_coeff = row.get(leading_col, QQ(0))
        if leading_coeff != 0:
            solution_entries[leading_col] = rhs / leading_coeff
        else:
            # Free variable, set to 0
            solution_entries[leading_col] = QQ(0)

    return QQVector(solution_entries)


# Export functions
__all__ = [
    "zero_vector",
    "unit_vector",
    "identity_matrix",
    "vector_from_list",
    "matrix_from_lists",
    "mk_vector",
    "mk_matrix",
    "linterm_of",
    "solve_linear_system",
]
