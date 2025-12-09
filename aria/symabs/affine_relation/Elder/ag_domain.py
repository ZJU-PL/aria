"""AG (Affine Generator) Domain Implementation.

This module implements the AG domain from Elder et al.'s paper "Abstract Domains
of Affine Relations". The AG domain uses generator form with diagonal decomposition
and provides efficient representations for certain classes of affine relations.
"""

import numpy as np
import z3
from typing import Dict, List, Set, Tuple, Optional, Union
from .matrix_ops import Matrix


class AG:
    """Represents an element in the AG (Affine Generator) domain.

    An AG element is a two-vocabulary matrix whose rows are the affine generators
    of a two-vocabulary relation. An AG element is an r-by-(2k + 1) matrix G,
    with 0 ≤ r ≤ 2k + 1. The concretization of an AG element G is:

    γ_AG(G) = {(x, x') | [x x'] ∈ Z_{2^w}^k × Z_{2^w}^k | [x x'] ∈ row G}

    where "row G" denotes the row space of G.
    """

    def __init__(self, matrix: Optional[Matrix] = None, w: int = 32):
        """Initialize an AG element.

        Args:
            matrix: Generator matrix representing the affine generators
            w: Word size (default 32 bits)
        """
        if matrix is not None:
            self.matrix = matrix
            self.modulus = matrix.modulus
            self.w = matrix.modulus.bit_length() - 1
            self.rows, self.cols = matrix.rows, matrix.cols
        else:
            # Empty AG element
            self.modulus = 2**w
            self.w = w
            self.matrix = Matrix(np.zeros((0, 1), dtype=object), self.modulus)
            self.rows, self.cols = 0, 1

    def __repr__(self) -> str:
        return f"AG(matrix_shape=({self.rows}, {self.cols}), w={self.w})"

    def is_empty(self) -> bool:
        """Check if this AG element represents the empty relation."""
        return self.rows == 0

    def is_bottom(self) -> bool:
        """Check if this is the bottom element (empty relation)."""
        return self.is_empty()

    def copy(self) -> 'AG':
        """Create a copy of this AG element."""
        return AG(self.matrix.copy(), self.w)

    def concretize(self) -> str:
        """Return a symbolic representation of the concretization."""
        if self.is_empty():
            return "∅"

        generators = []
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
                generator = " + ".join(terms)
                if constant != 0:
                    if constant > 0:
                        generator += f" + {constant}"
                    else:
                        generator += f" - {-constant}"
                generator += " = 0"
            else:
                generator = " + ".join(terms) + " = 0"

            generators.append(generator)

        return " ∨ ".join(generators)

    def join(self, other: 'AG') -> 'AG':
        """Compute the join of two AG elements."""
        if self.is_empty():
            return other.copy()
        if other.is_empty():
            return self.copy()

        # For AG domain, join is matrix concatenation (row-wise)
        new_data = np.vstack([self.matrix.data, other.matrix.data])
        return AG(Matrix(new_data, self.modulus), self.w)

    def diagonal_decomposition(self) -> Tuple[Matrix, Matrix, Matrix]:
        """Perform diagonal decomposition of a square matrix (Definition 2 from paper).

        Returns:
            Tuple of (L, D, R) where M = L·D·R, L and R are invertible,
            and D is diagonal with entries that are either 0 or powers of 2.
        """
        if self.rows != self.cols:
            raise ValueError("Diagonal decomposition requires square matrix")

        M = self.matrix
        n = M.rows
        w = self.w

        # Initialize L and R as identity matrices
        L_data = np.eye(n, dtype=object)
        R_data = np.eye(n, dtype=object)
        D_data = M.data.copy()

        # Gaussian elimination adapted for modular arithmetic
        for i in range(n):
            # Find pivot (row with smallest rank in column i)
            pivot_row = i
            min_rank = float('inf')

            for j in range(i, n):
                if D_data[j, i] != 0:
                    rank_val = self._compute_rank(D_data[j, i])
                    if rank_val < min_rank:
                        min_rank = rank_val
                        pivot_row = j

            # Swap rows if needed
            if pivot_row != i:
                D_data[i], D_data[pivot_row] = D_data[pivot_row].copy(), D_data[i].copy()
                L_data[i], L_data[pivot_row] = L_data[pivot_row].copy(), L_data[i].copy()

            # If pivot is zero, continue (singular matrix)
            if D_data[i, i] == 0:
                continue

            # Eliminate below and above
            for j in range(n):
                if j == i:
                    continue

                if D_data[j, i] != 0:
                    # Compute factor: need to handle modular inverse
                    pivot_val = D_data[i, i]
                    elim_val = D_data[j, i]

                    # For modular arithmetic, we need to solve: pivot_val * factor ≡ elim_val mod modulus
                    # This requires computing the inverse of pivot_val mod modulus
                    try:
                        pivot_inv = self._modular_inverse(pivot_val, self.modulus)
                        factor = (elim_val * pivot_inv) % self.modulus

                        # Update D: D[j,:] = D[j,:] - factor * D[i,:]
                        for k in range(n):
                            D_data[j, k] = (D_data[j, k] - factor * D_data[i, k]) % self.modulus

                        # Update L: L[j,:] = L[j,:] - factor * L[i,:]
                        for k in range(n):
                            L_data[j, k] = (L_data[j, k] - factor * L_data[i, k]) % self.modulus

                    except ValueError:
                        # Pivot has no inverse, skip elimination
                        pass

        return Matrix(L_data, self.modulus), Matrix(D_data, self.modulus), Matrix(R_data, self.modulus)

    def dualize(self) -> 'AG':
        """Compute the dual of this AG element (Definition 3 from paper).

        The dual M⊥ is defined such that:
        - PAD(M) is the (2k+1)-by-(2k+1) matrix TSL(PAD(M))
        - L, D, R is the diagonal decomposition of PAD(M)
        - T is the diagonal matrix with T_{i,i} = 2^{w-rank(D_{i,i})}
        - M⊥ = (L^{-1})^t · T · (R^{-1})^t

        Returns:
            Dual AG element
        """
        k = (self.cols - 1) // 2
        w = self.w

        # Step 1: Compute PAD(M) - pad to (2k+1) x (2k+1)
        pad_size = 2 * k + 1
        pad_matrix = np.zeros((pad_size, pad_size), dtype=object)

        # Copy original matrix into PAD form
        for i in range(self.rows):
            for j in range(self.cols):
                pad_matrix[i, j] = self.matrix[i, j]

        pad_ag = AG(Matrix(pad_matrix, self.modulus), self.w)

        # Step 2: Diagonal decomposition
        L, D, R = pad_ag.diagonal_decomposition()

        # Step 3: Compute T matrix
        T = np.zeros((pad_size, pad_size), dtype=object)
        for i in range(pad_size):
            if D[i, i] != 0:
                rank_val = self._compute_rank(D[i, i])
                T[i, i] = 2**(w - rank_val)
            else:
                T[i, i] = 0

        # Step 4: Compute inverses of L and R
        # For now, assume L and R are diagonal (which they should be after decomposition)
        L_inv_data = np.zeros((pad_size, pad_size), dtype=object)
        R_inv_data = np.zeros((pad_size, pad_size), dtype=object)

        for i in range(pad_size):
            if L[i, i] != 0:
                L_inv_data[i, i] = self._modular_inverse(L[i, i], self.modulus)
            if R[i, i] != 0:
                R_inv_data[i, i] = self._modular_inverse(R[i, i], self.modulus)

        L_inv = Matrix(L_inv_data, self.modulus)
        R_inv = Matrix(R_inv_data, self.modulus)

        # Compute (L^{-1})^t · T · (R^{-1})^t
        L_inv_T = L_inv.data.T
        R_inv_T = R_inv.data.T

        temp = np.dot(L_inv_T, T)
        dual_matrix_data = np.dot(temp, R_inv_T) % self.modulus

        return AG(Matrix(dual_matrix_data, self.modulus), self.w)

    def _compute_rank(self, value: int) -> int:
        """Compute the rank (number of trailing zeros) of a value."""
        if value == 0:
            return self.w
        rank = 0
        val = abs(value)
        while val % 2 == 0 and val > 0:
            val //= 2
            rank += 1
        return rank

    def _modular_inverse(self, a: int, m: int) -> int:
        """Compute modular inverse of a modulo m using extended Euclidean algorithm."""
        if a == 0:
            raise ValueError("No inverse for 0")

        # Extended Euclidean algorithm
        m0 = m
        y = 0
        x = 1

        if m == 1:
            return 0

        while a > 1:
            # q is quotient
            q = a // m
            t = m

            # m is remainder now, process same as Euclid's algo
            m = a % m
            a = t
            t = y

            # Update y and x
            y = x - q * y
            x = t

        # Make x positive
        if x < 0:
            x += m0

        if x >= m0:
            raise ValueError(f"No inverse for {a} mod {m0}")

        return x

    def __eq__(self, other) -> bool:
        """Check equality of AG elements."""
        if not isinstance(other, AG):
            return False
        if self.rows != other.rows or self.cols != other.cols:
            return False

        return np.array_equal(self.matrix.data, other.matrix.data)


def alpha_ag(phi: z3.ExprRef, pre_vars: List[z3.ExprRef], post_vars: List[z3.ExprRef]) -> AG:
    """Alpha function for AG domain.

    The AG domain represents relations as generators, so the alpha function
    extracts the strongest generator-based relation that overapproximates
    the concrete semantics defined by phi.

    Args:
        phi: QFBV formula as Z3 expression
        pre_vars: Z3 pre-state variables
        post_vars: Z3 post-state variables

    Returns:
        AG element representing the abstraction of phi
    """
    # First get the MOS abstraction
    from .mos_domain import alpha_mos
    mos_result = alpha_mos(phi, pre_vars, post_vars)

    # Convert MOS to KS, then KS to AG
    from .conversions import mos_to_ks, ks_to_ag
    ks_result = mos_to_ks(mos_result)
    return ks_to_ag(ks_result)
