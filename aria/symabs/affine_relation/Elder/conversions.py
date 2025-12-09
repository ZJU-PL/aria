"""Conversion algorithms between abstract domains.

This module implements the conversion algorithms described in Elder et al.'s
paper for converting between MOS, KS, and AG abstract domains.
"""

import numpy as np
from typing import List
from .matrix_ops import Matrix, howellize, make_explicit
from .mos_domain import MOS
from .ks_domain import KS
from .ag_domain import AG


def mos_to_ks(mos_element: MOS) -> KS:
    """Convert MOS element to KS element.

    Args:
        mos_element: MOS element to convert

    Returns:
        Equivalent KS element

    From Theorem 1: γ_MOS(B) ⊆ γ_AG(G) where G is constructed from B
    """
    if mos_element.is_empty():
        return KS(w=mos_element.w)

    k = mos_element.matrices[0].rows - 1  # Number of variables
    w = mos_element.w

    # For each MOS matrix, create corresponding AG matrix
    ag_matrices = []
    for T in mos_element.matrices:
        # Extract b, M from T = [M b; 0 1]
        M = T.data[:k, :k]
        b = T.data[:k, k]

        # Create AG matrix representing the affine transformation
        # For identity transformation x' = x, we need generators x'_i = x_i for each i
        ag_data = np.zeros((k, 2*k + 1), dtype=object)

        # For each variable i, create generator: x'_i - x_i = 0
        for i in range(k):
            ag_data[i, i] = -1      # -x_i coefficient
            ag_data[i, k + i] = 1   # x'_i coefficient
            # Constant term is 0

        # Also add a constant generator for the affine constant (if any)
        if np.any(b != 0):
            # Add generator for the constant term
            const_row = np.zeros(2*k + 1, dtype=object)
            const_row[2*k] = -1  # -constant coefficient
            for j in range(k):
                const_row[j] = b[j]  # x_j coefficient for constant relation
            ag_data = np.vstack([ag_data, const_row.reshape(1, -1)])

        ag_matrices.append(AG(Matrix(ag_data, 2**w)))

    # Join all AG matrices
    if not ag_matrices:
        return KS(w=w)

    result_ag = ag_matrices[0]
    for ag in ag_matrices[1:]:
        result_ag = result_ag.join(ag)

    # Convert AG to KS
    return ag_to_ks(result_ag)


def ks_to_mos(ks_element: KS) -> MOS:
    """Convert KS element to MOS element.

    Args:
        ks_element: KS element to convert

    Returns:
        Equivalent MOS element
    """
    if ks_element.is_empty():
        return MOS(w=ks_element.w)

    # Convert KS to AG first, then AG to MOS
    ag_element = ks_to_ag(ks_element)
    return ag_to_mos(ag_element)


def ag_to_ks(ag_element: AG) -> KS:
    """Convert AG element to KS element.

    From the paper: AG and KS are equivalent domains, so we can convert
    an AG element to an equivalent KS element with no loss of precision.

    Args:
        ag_element: AG element to convert

    Returns:
        Equivalent KS element
    """
    if ag_element.is_empty():
        return KS(w=ag_element.w)

    # The conversion uses the dualization operation and permutation of columns
    # This is based on the algorithm described in section 3 of the paper

    G = ag_element.matrix
    k = (G.cols - 1) // 2

    # Create KS matrix Z of the form [X X'|b] such that γ_KS(Z) = γ_AG(G)
    # We construct Z by letting [b|X X'] = G ⊥ and permuting columns

    G_dual = ag_element.dualize()
    ks_data = np.zeros((G_dual.rows, 2*k + 1), dtype=object)

    # Copy coefficients and rearrange columns according to the algorithm
    for i in range(G_dual.rows):
        for j in range(2*k + 1):
            if j < k:
                # Pre-state variables
                ks_data[i, j] = G_dual.matrix[i, j]
            elif j < 2*k:
                # Post-state variables
                ks_data[i, j] = G_dual.matrix[i, j]
            else:
                # Constant term
                ks_data[i, 2*k] = G_dual.matrix[i, 2*k]

    return KS(Matrix(ks_data, ag_element.modulus))


def ks_to_ag(ks_element: KS) -> AG:
    """Convert KS element to AG element.

    Args:
        ks_element: KS element to convert

    Returns:
        Equivalent AG element
    """
    if ks_element.is_empty():
        return AG(w=ks_element.w)

    # Reverse the process of ag_to_ks
    # This involves reversing the dualization and column permutation

    X = ks_element.matrix
    k = (X.cols - 1) // 2

    # Create padded matrix and reverse the column permutation
    ag_data = np.zeros((X.rows, 2*k + 1), dtype=object)

    for i in range(X.rows):
        for j in range(2*k + 1):
            if j < k:
                # Pre-state variables (reverse order)
                ag_data[i, k - 1 - j] = X[i, j]
            elif j < 2*k:
                # Post-state variables (reverse order)
                ag_data[i, 2*k - 1 - j] = X[i, j]
            else:
                # Constant term
                ag_data[i, 2*k] = X[i, 2*k]

    ag_element = AG(Matrix(ag_data, ks_element.modulus), ks_element.w)

    # Apply reverse dualization
    return ag_element.dualize()


def ag_to_mos(ag_element: AG) -> MOS:
    """Convert AG element to MOS element using SHATTER operation.

    From Theorem 3: γ_AG(G) = γ_MOS(SHATTER(G))

    Args:
        ag_element: AG element to convert

    Returns:
        Equivalent MOS element
    """
    if ag_element.is_empty():
        return MOS(w=ag_element.w)

    # Apply SHATTER operation (Theorem 3 from paper)
    from .mos_domain import shatter_ag

    shattered_matrices = shatter_ag(ag_element.matrix)
    return MOS(shattered_matrices, ag_element.w)
