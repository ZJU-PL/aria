"""KS (King/Søndergaard) Domain Implementation.

This module implements the KS domain from Elder et al.'s paper "Abstract Domains
of Affine Relations". The KS domain represents two-vocabulary affine relations
using constraints of the form Σ a_i x_i + Σ a'_i x'_i = b.
"""

import numpy as np
import z3
from typing import Dict, List, Set, Tuple, Optional, Union
from .matrix_ops import Matrix


class KS:
    """Represents an element in the KS (King/Søndergaard) domain.

    A KS element is a two-vocabulary matrix whose rows represent constraints on
    a two-vocabulary relation. A KS element is an r-by-(2k + 1) matrix X, with
    0 ≤ r ≤ 2k + 1. The concretization of a KS element X is:

    γ_KS(X) = {(x, x') | [x x'] ∈ Z_{2^w}^k × Z_{2^w}^k | ∀[x x'] ∈ null^t G}

    where G is the matrix obtained by converting X to AG form and then to MOS form.
    """

    def __init__(self, matrix: Optional[Matrix] = None, w: int = 32):
        """Initialize a KS element.

        Args:
            matrix: Two-vocabulary matrix representing the constraints
            w: Word size (default 32 bits)
        """
        if matrix is not None:
            self.matrix = matrix
            self.modulus = matrix.modulus
            self.w = matrix.modulus.bit_length() - 1
            self.rows, self.cols = matrix.rows, matrix.cols
        else:
            # Empty KS element
            self.modulus = 2**w
            self.w = w
            self.matrix = Matrix(np.zeros((0, 1), dtype=object), self.modulus)
            self.rows, self.cols = 0, 1

    def __repr__(self) -> str:
        return f"KS(matrix_shape=({self.rows}, {self.cols}), w={self.w})"

    def is_empty(self) -> bool:
        """Check if this KS element represents the empty relation."""
        return self.rows == 0

    def is_bottom(self) -> bool:
        """Check if this is the bottom element (empty relation)."""
        return self.is_empty()

    def copy(self) -> 'KS':
        """Create a copy of this KS element."""
        return KS(self.matrix.copy(), self.w)

    def concretize(self) -> str:
        """Return a symbolic representation of the concretization."""
        if self.is_empty():
            return "∅"

        constraints = []
        k = (self.cols - 1) // 2  # Number of variables

        for i in range(self.rows):
            row = self.matrix[i, :]
            coeffs = row[:k]  # Coefficients for x variables
            coeffs_prime = row[k:2*k]  # Coefficients for x' variables
            constant = row[2*k]  # Constant term

            terms = []

            # Add x terms
            for j, coeff in enumerate(coeffs):
                if coeff != 0:
                    terms.append(f"{coeff}·x_{j}")

            # Add x' terms
            for j, coeff in enumerate(coeffs_prime):
                if coeff != 0:
                    terms.append(f"{coeff}·x'_{j}")

            if constant != 0 or not terms:
                constraint = " + ".join(terms)
                if constant != 0:
                    if constant > 0:
                        constraint += f" + {constant}"
                    else:
                        constraint += f" - {-constant}"
                constraint += " = 0"
            else:
                constraint = " + ".join(terms) + " = 0"

            constraints.append(constraint)

        return " ∧ ".join(constraints)

    def join(self, other: 'KS') -> 'KS':
        """Compute the join of two KS elements."""
        if self.is_empty():
            return other.copy()
        if other.is_empty():
            return self.copy()

        # Create a new matrix with rows from both elements
        new_data = np.vstack([self.matrix.data, other.matrix.data])
        return KS(Matrix(new_data, self.modulus), self.w)

    def project_pre(self, variables: List[int]) -> 'KS':
        """Project onto pre-state variables (existential quantification over post-state)."""
        if self.is_empty():
            return self.copy()

        k = (self.cols - 1) // 2
        result_data = np.zeros((self.rows, k + 1), dtype=object)

        for i in range(self.rows):
            # Copy pre-state coefficients
            for j in variables:
                result_data[i, j] = self.matrix[i, j]

            # Copy constant term
            result_data[i, k] = self.matrix[i, 2*k]

        return KS(Matrix(result_data, self.modulus), self.w)

    def project_post(self, variables: List[int]) -> 'KS':
        """Project onto post-state variables (existential quantification over pre-state)."""
        if self.is_empty():
            return self.copy()

        k = (self.cols - 1) // 2
        result_data = np.zeros((self.rows, k + 1), dtype=object)

        for i in range(self.rows):
            # Copy post-state coefficients
            for j in variables:
                result_data[i, j] = self.matrix[i, k + j]

            # Copy constant term
            result_data[i, k] = self.matrix[i, 2*k]

        return KS(Matrix(result_data, self.modulus), self.w)

    def compose(self, other: 'KS') -> 'KS':
        """Compose two KS elements (Y ∘ Z where Y ○ Z = {(x,z) | ∃y. (x,y) ∈ Y ∧ (y,z) ∈ Z})."""
        # This is a simplified version - full implementation would use
        # the join algorithm from King & Søndergaard

        k = (self.cols - 1) // 2

        # Create three-vocabulary matrix for intermediate computation
        # Variables: x_pre, y, x'_post
        three_vocab_size = 3 * k + 1

        # Build Y ○ Z computation matrix
        # This is a placeholder - full implementation needs the algorithm
        # from King & Søndergaard paper

        return KS(Matrix(np.zeros((0, three_vocab_size), dtype=object), self.modulus), self.w)

    def __eq__(self, other) -> bool:
        """Check equality of KS elements."""
        if not isinstance(other, KS):
            return False
        if self.rows != other.rows or self.cols != other.cols:
            return False

        return np.array_equal(self.matrix.data, other.matrix.data)


def alpha_ks(phi: z3.ExprRef, pre_vars: List[z3.ExprRef], post_vars: List[z3.ExprRef]) -> KS:
    """Alpha function for KS domain.

    The KS domain represents two-vocabulary relations, so the alpha function
    extracts the strongest two-vocabulary relation that overapproximates
    the concrete semantics defined by phi.

    Args:
        phi: QFBV formula as Z3 expression
        pre_vars: Z3 pre-state variables
        post_vars: Z3 post-state variables

    Returns:
        KS element representing the abstraction of phi
    """
    # First get the MOS abstraction
    from .mos_domain import alpha_mos
    mos_result = alpha_mos(phi, pre_vars, post_vars)

    # Convert MOS to KS
    from .conversions import mos_to_ks
    return mos_to_ks(mos_result)
