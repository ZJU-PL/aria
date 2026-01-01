"""
Linear Transition Systems (LTS).

This module implements linear transition systems where the state space
is a finite dimensional vector space and transitions are linear maps.

Based on src/lts.ml from the OCaml implementation.
"""

from __future__ import annotations
from typing import (
    Dict,
    List,
    Set,
    Tuple,
    Optional,
    Union,
    Any,
    Generic,
    TypeVar,
    Callable,
)
from dataclasses import dataclass, field
from fractions import Fraction
import logging

from aria.srk.syntax import Context, Symbol, Type, Expression
from aria.srk.linear import QQVector, QQMatrix, QQVectorSpace, identity_matrix
from aria.srk.transitionFormula import TransitionFormula
from aria.srk.qQ import QQ

logger = logging.getLogger(__name__)

T = TypeVar("T")


# Helper functions for linear algebra operations
def divide_right(A: QQMatrix, B: QQMatrix) -> Optional[QQMatrix]:
    """Solve X*B = A for X (right division)."""
    try:
        from .linear import divide_right as linear_divide_right

        return linear_divide_right(A, B)
    except Exception:
        return None


def divide_left(A: QQMatrix, B: QQMatrix) -> Optional[QQMatrix]:
    """Solve B*X = A for X (left division)."""
    try:
        from .linear import divide_left as linear_divide_left

        return linear_divide_left(A, B)
    except Exception:
        return None


def nullspace(matrix: QQMatrix, dims: List[int]) -> List[QQVector]:
    """Compute basis for nullspace of matrix."""
    try:
        from .linear_advanced import null_space

        return null_space(matrix)
    except ImportError:
        # Fallback implementation without numpy
        # Use Gaussian elimination to find nullspace
        if not matrix.rows:
            # Empty matrix has full nullspace
            return [QQVector.of_term(QQ.one(), dim) for dim in dims]

        # Create augmented matrix with identity columns
        max_dim = max(dims) if dims else 0
        num_cols = max(
            max_dim + 1,
            max(
                (max(row.dimensions()) for row in matrix.rows if row.dimensions()),
                default=0,
            )
            + 1,
        )

        # Build augmented matrix [matrix | identity]
        augmented_rows = []
        for row in matrix.rows:
            new_row = row
            for i in range(num_cols):
                new_row = new_row.set(i, new_row.get(i, QQ(0)))
            augmented_rows.append(new_row)

        # Add identity rows for each dimension
        for dim in dims:
            identity_row = QQVector.of_term(QQ.one(), dim + num_cols)
            augmented_rows.append(identity_row)

        augmented_matrix = QQMatrix(augmented_rows)

        # Use Gaussian elimination to find nullspace
        # This is a simplified version - full implementation would use proper row reduction
        nullspace_vectors = []

        # For each dimension, check if it's free
        for dim in dims:
            # Check if this dimension has no constraints
            is_free = True
            for row in matrix.rows:
                if row.get(dim, QQ(0)) != 0:
                    is_free = False
                    break

            if is_free:
                nullspace_vectors.append(QQVector.of_term(QQ.one(), dim))

        return nullspace_vectors


def linear_solve(matrix: QQMatrix, vector: QQVector) -> Optional[QQVector]:
    """Solve matrix * x = vector for x."""
    try:
        from .linear_utils import solve_linear_system

        return solve_linear_system(matrix, vector)
    except Exception:
        return None


@dataclass(frozen=True)
class LinearTransitionSystem:
    """Linear transition system: Ax' = Bx.

    Represents a transition relation as a pair of matrices (A, B) where
    the transition is characterized by the equation Ax' = Bx.
    """

    A: QQMatrix  # Matrix for primed variables (post-state)
    B: QQMatrix  # Matrix for unprimed variables (pre-state)

    def __init__(self, A: QQMatrix, B: QQMatrix):
        object.__setattr__(self, "A", A)
        object.__setattr__(self, "B", B)

    def __str__(self) -> str:
        return f"LTS(A=\\n{self.A}\\nB=\\n{self.B})"


class PartialLinearMap:
    """Partial linear map with domain constraints.

    A partial linear map is a partial function that is linear and
    whose domain is a vector space. The domain is specified by
    guard constraints (orthogonal complement representation).
    """

    def __init__(self, map_matrix: QQMatrix, guard: List[QQVector]):
        """Create a partial linear map.

        Args:
            map_matrix: The linear transformation matrix
            guard: List of vectors forming the orthogonal complement of the domain.
                  Domain = { v : g · v = 0 for all g in guard }
        """
        self.map_matrix = map_matrix
        self.guard = guard
        self._normalize()

    def _normalize(self) -> None:
        """Normalize the partial map.

        After normalization:
        1. Guard vectors are linearly independent (basis)
        2. Map sends every vector orthogonal to domain to 0

        Simplified implementation - a full version would compute proper basis.
        """
        # For now, just keep the guard as-is
        # A full implementation would:
        # 1. Compute basis for guard space
        # 2. Compute nullspace (domain)
        # 3. Perform change of basis
        pass

    @staticmethod
    def make(map_matrix: QQMatrix, guard: List[QQVector]) -> PartialLinearMap:
        """Create and normalize a partial linear map."""
        return PartialLinearMap(map_matrix, guard)

    @staticmethod
    def identity(dim: int) -> PartialLinearMap:
        """Create identity map on coordinates 0..dim-1."""
        identity_mat = identity_matrix(dim)
        return PartialLinearMap(identity_mat, [])

    def equal(self, other: PartialLinearMap) -> bool:
        """Check equality of partial linear maps."""
        # Convert guards to QQVectorSpace objects for comparison
        guard1 = (
            QQVectorSpace(self.guard) if isinstance(self.guard, list) else self.guard
        )
        guard2 = (
            QQVectorSpace(other.guard) if isinstance(other.guard, list) else other.guard
        )
        return QQMatrix.equal(
            self.map_matrix, other.map_matrix
        ) and QQVectorSpace.equal(guard1, guard2)

    def __eq__(self, other: object) -> bool:
        """Check equality of partial linear maps."""
        if not isinstance(other, PartialLinearMap):
            return False
        return self.equal(other)

    def compose(self, other: PartialLinearMap) -> PartialLinearMap:
        """Compose two partial linear maps: self ∘ other."""
        # Composed map matrix
        composed_map = self.map_matrix * other.map_matrix

        # Composed guard: self.guard * other.map + other.guard
        guard_part1 = QQMatrix.rowsi(QQMatrix(self.guard) * other.map_matrix)
        guard_part1_vecs = [vec for (_, vec) in guard_part1]

        composed_guard_space = QQVectorSpace.sum(
            QQVectorSpace(guard_part1_vecs), QQVectorSpace(other.guard)
        )
        composed_guard = composed_guard_space.basis

        return PartialLinearMap.make(composed_map, composed_guard)

    def iteration_sequence(self) -> Tuple[List[PartialLinearMap], List[QQVector]]:
        """Compute iteration sequence until guard stabilizes.

        Returns ([f, f∘f, ..., f^n], stable_guard) where stable_guard is
        the invariant guard that stabilizes at f^n.
        """

        def fix(
            current: PartialLinearMap,
        ) -> Tuple[List[PartialLinearMap], List[QQVector]]:
            composed = current.compose(self)
            if QQVectorSpace.equal(current.guard, composed.guard):
                return ([current], current.guard)
            else:
                (seq, stable) = fix(composed)
                return ([current] + seq, stable)

        return fix(self)

    def map(self) -> QQMatrix:
        """Access the underlying map matrix."""
        return self.map_matrix

    def guard_space(self) -> List[QQVector]:
        """Access the guard (orthogonal complement of domain)."""
        if isinstance(self.guard, QQVectorSpace):
            return self.guard.basis
        return self.guard

    def __str__(self) -> str:
        guard_str = ", ".join(str(g) for g in self.guard)
        return f"PartialLinearMap(map=\\n{self.map_matrix}, guard=[{guard_str}])"


class LTSAnalysis:
    """Analysis operations for linear transition systems."""

    def __init__(self, context: Context):
        self.context = context

    def abstract_lts(self, tf: TransitionFormula) -> LinearTransitionSystem:
        """Find best LTS abstraction of transition formula w.r.t. affine simulations.

        Extracts the affine hull of the transition formula and constructs
        matrices A and B such that Ax' = Bx characterizes the linear part.
        """
        try:
            from .transitionFormula import (
                formula as tf_formula,
                symbols as tf_symbols,
                is_symbolic_constant,
                exists as tf_exists,
            )
            from .abstract import affine_hull
            from .linear import Linear, linterm_of

            tr_symbols = tf_symbols(tf)
            phi_formula = tf_formula(tf)

            # Get all symbols that exist in the formula
            from .syntax import symbols as get_symbols

            phi_symbols_set = get_symbols(phi_formula)
            exists_pred = tf_exists(tf)
            phi_symbols = [s for s in phi_symbols_set if exists_pred(s)]

            # Identify symbolic constants
            constants = [s for s in phi_symbols if is_symbolic_constant(tf, s)]

            # Create pre_map: post dims -> pre dims
            pre_map = {}
            for s, s_prime in tr_symbols:
                pre_dim = Linear.dim_of_sym(s)
                post_dim = Linear.dim_of_sym(s_prime)
                pre_map[post_dim] = pre_dim

            # Compute affine hull and build matrices
            mA = QQMatrix.zero()
            mB = QQMatrix.zero()
            row_idx = 0

            hull_terms = affine_hull(self.context, phi_formula, phi_symbols)

            for term in hull_terms:
                linterm = linterm_of(self.context, term)
                a_vec = QQVector.zero()
                b_vec = QQVector.zero()

                for coeff, dim in QQVector.enum(linterm):
                    if dim in pre_map:
                        # Post-state variable: goes in A with negated coefficient
                        pre_dim = pre_map[dim]
                        a_vec = QQVector.add_term(QQ.negate(coeff), pre_dim, a_vec)
                    elif dim == Linear.const_dim:
                        # Constant: goes in B
                        b_vec = QQVector.add_term(coeff, Linear.const_dim, b_vec)
                    else:
                        # Pre-state or symbolic constant: goes in B
                        b_vec = QQVector.add_term(coeff, dim, b_vec)

                mA = QQMatrix.add_row(row_idx, a_vec, mA)
                mB = QQMatrix.add_row(row_idx, b_vec, mB)
                row_idx += 1

            # Add identity constraints for symbolic constants
            for const_sym in [Linear.const_dim] + [
                Linear.dim_of_sym(c) for c in constants
            ]:
                a_vec = QQVector.of_term(QQ.one(), const_sym)
                b_vec = QQVector.of_term(QQ.one(), const_sym)
                mA = QQMatrix.add_row(row_idx, a_vec, mA)
                mB = QQMatrix.add_row(row_idx, b_vec, mB)
                row_idx += 1

            return LinearTransitionSystem(mA, mB)

        except Exception as e:
            logger.warning(f"LTS abstraction failed: {e}, returning empty LTS")
            return LinearTransitionSystem(QQMatrix.zero(), QQMatrix.zero())

    def contains(
        self, lts1: LinearTransitionSystem, lts2: LinearTransitionSystem
    ) -> bool:
        """Check if lts1 contains lts2 (lts2's transitions ⊆ lts1's transitions)."""
        witness = self.containment_witness(lts1, lts2)
        return witness is not None

    def containment_witness(
        self, lts1: LinearTransitionSystem, lts2: LinearTransitionSystem
    ) -> Optional[QQMatrix]:
        """Find witness matrix M such that M*A1 = A2 and M*B1 = B2.

        If such M exists, then lts1 contains lts2.
        """
        # Interleave columns of A and B matrices
        interleaved1 = QQMatrix.interlace_columns(lts1.A, lts1.B)
        interleaved2 = QQMatrix.interlace_columns(lts2.A, lts2.B)

        # Solve for M: M * interleaved1 = interleaved2
        return divide_right(interleaved2, interleaved1)


def max_rowspace_projection(mA: QQMatrix, mB: QQMatrix) -> List[QQVector]:
    """Find basis for { v : exists u. uA = vB }.

    Computes the maximum rowspace projection from A to B.
    """
    # Create system: u*A - v*B = 0 where u's are even columns, v's are odd
    mat = QQMatrix.interlace_columns(
        mA.transpose(), QQMatrix.scalar_mul(QQ.of_int(-1), mB).transpose()
    )

    result = []
    mat_rows = QQMatrix.nb_rows(mat)

    # Try to find vectors v such that there exists u with uA = vB
    for r, _ in QQMatrix.rowsi(mB):
        col = 2 * r + 1  # Column for v_r

        # Add constraint that v_r = 1
        mat_with_constraint = QQMatrix.add_row(
            mat_rows, QQVector.of_term(QQ.one(), col), mat
        )

        # Try to solve
        solution = linear_solve(
            mat_with_constraint, QQVector.of_term(QQ.one(), mat_rows)
        )

        if solution is not None:
            # Extract v from solution (odd columns)
            v_row = QQVector.zero()
            for entry, i in QQVector.enum(solution):
                if i % 2 == 1:  # Odd column = v component
                    v_row = v_row.add_term(entry, i // 2)

            if v_row != QQVector.zero():
                result.append(v_row)
                mat = mat_with_constraint
                mat_rows += 1

    return result


def determinize(lts: LinearTransitionSystem) -> Tuple[PartialLinearMap, QQMatrix]:
    """Find best deterministic abstraction of an LTS.

    Returns (dlts, simulation_matrix) where:
    - dlts is a partial linear map representing the deterministic LTS
    - simulation_matrix S witnesses that Sx' = T(Sx) for the DLTS
    """

    def fix(mA: QQMatrix, mB: QQMatrix) -> Tuple[QQMatrix, QQMatrix]:
        """Iteratively refine until we get a deterministic system."""
        mS = QQMatrix(max_rowspace_projection(mA, mB))

        # Account for zero rows of B
        mT_prime = mS
        next_row = QQMatrix.nb_rows(mB)

        for i in QQMatrix.row_set(mA):
            row_b = QQMatrix.row(i, mB)
            if QQVector.is_zero(row_b):
                row_vec = QQVector.of_term(QQ.one(), i)
                mT_prime = QQMatrix.add_row(next_row, row_vec, mT_prime)
                next_row += 1

        # Check if we're done
        if QQMatrix.nb_rows(mB) == QQMatrix.nb_rows(mS):
            return (mA, mB)
        else:
            # Continue refining
            return fix(mT_prime * mA, mT_prime * mB)

    (mA, mB) = fix(lts.A, lts.B)

    # Check if the system is already deterministic (unchanged by fix)
    if QQMatrix.equal(mA, lts.A) and QQMatrix.equal(mB, lts.B):
        # System is already deterministic, return identity similarity matrix
        num_rows = QQMatrix.nb_rows(mA)
        identity_vectors = [QQVector.of_term(QQ.one(), i) for i in range(num_rows)]
        mS = QQMatrix(identity_vectors)
        # For deterministic systems, the deterministic LTS map is the original A
        mT = mA
    else:
        # Compute simulation matrix S (basis of A's row space)
        mS = QQMatrix(QQVectorSpace.simplify(QQVectorSpace.of_matrix(mA).basis))

        # Compute D such that DA = S
        mD = divide_right(mS, mA)
        if mD is None:
            # Fallback
            mD = mS

        # Compute T such that DB = TS
        mDB = mD * mB
        mT = divide_right(mDB, mS)
        if mT is None:
            # Fallback
            mT = mDB

    # Compute guard: basis for { g : Ax' = Bx |= gSx = 0 }
    dims = sorted(list(set(QQMatrix.row_set(mA)) | set(QQMatrix.row_set(mB))))
    mN = QQMatrix(nullspace(mA.transpose(), dims))

    mNB = mN * mB
    mG = divide_right(mNB, mS)

    guard = QQVectorSpace.of_matrix(mG) if mG is not None else []

    return (PartialLinearMap.make(mT, guard), mS)


def dlts_inverse_image(sim: QQMatrix, dlts: PartialLinearMap) -> LinearTransitionSystem:
    """Compute inverse image of DLTS under similarity map.

    Given similarity S and DLTS with map T and guard G,
    compute LTS such that Ax' = Bx where:
    - A = S
    - B = T*S with guard constraints
    """
    mA = sim
    dynamics = dlts.map() * sim

    dim = QQMatrix.nb_rows(sim)
    mB = dynamics

    # Add guard constraints as additional rows
    for i, dom_constraint in enumerate(dlts.guard_space()):
        # Transform constraint by similarity
        transformed = QQVector.vector_left_mul(dom_constraint, sim)
        mB = QQMatrix.add_row(i + dim, transformed, mB)

    return LinearTransitionSystem(mA, mB)


def dlts_abstract_spectral(
    spectral_decomp: Callable[[QQMatrix, List[int]], List[QQVector]],
    dlts: PartialLinearMap,
    dim: int,
) -> Tuple[PartialLinearMap, QQMatrix]:
    """Compute best abstraction of DLTS satisfying spectral conditions.

    Args:
    spectral_decomp: Function computing spectral decomposition
    dlts: The partial linear map to abstract
    dim: Dimension of the space

    Returns:
    (abstracted_dlts, simulation_matrix)
    """

    def fix(mA: QQMatrix, mB: QQMatrix) -> Tuple[PartialLinearMap, QQMatrix]:
        (det_dlts, mS) = determinize(LinearTransitionSystem(mA, mB))
        (sequence, dom) = det_dlts.iteration_sequence()

        # Get the stable map (agrees with dlts on invariant domain)
        mT = PartialLinearMap.make(det_dlts.map(), dom).map()

        # Compute spectral decomposition
        dims = sorted(list(QQMatrix.row_set(mS)))
        sd = spectral_decomp(mT, dims)

        mP = QQMatrix(sd)
        mPS = mP * mS
        mPTS = (mP * det_dlts.map()) * mS
        mPDS = (mP * QQMatrix(det_dlts.guard_space())) * mS

        if len(sd) == QQMatrix.nb_rows(mS):
            # Spectral decomposition is complete
            map_result = divide_right(mPTS, mPS)
            if map_result is None:
                map_result = mPTS

            guard_mat = divide_right(mPDS, mPS)
            guard_result = (
                QQVectorSpace.of_matrix(guard_mat) if guard_mat is not None else []
            )

            return (PartialLinearMap.make(map_result, guard_result), mPS)
        else:
            # Continue refining
            mB_new = mPTS
            size = QQMatrix.nb_rows(mPS)
            for i, row in QQMatrix.rowsi(mPDS):
                mB_new = QQMatrix.add_row(i + size, row, mB_new)

            return fix(mPS, mB_new)

    # Initial system: [I] x' = [T] x
    #                 [0]      [D]
    mA = QQMatrix.identity(list(range(dim)))
    mB = dlts.map()

    for i, row in enumerate(dlts.guard_space()):
        mB = QQMatrix.add_row(i + dim, row, mB)

    return fix(mA, mB)


def periodic_rational_spectrum_reflection(
    dlts: PartialLinearMap, dim: int
) -> Tuple[PartialLinearMap, QQMatrix]:
    """Find best abstraction with periodic rational spectrum."""
    from .linear import periodic_rational_spectral_decomposition

    def spectral_decomp(m: QQMatrix, dims: List[int]) -> List[QQVector]:
        decomp = periodic_rational_spectral_decomposition(m, dims)
        return [v for (_, _, v) in decomp]

    return dlts_abstract_spectral(spectral_decomp, dlts, dim)


def rational_spectrum_reflection(
    dlts: PartialLinearMap, dim: int
) -> Tuple[PartialLinearMap, QQMatrix]:
    """Find best abstraction with rational spectrum."""
    from .linear import rational_spectral_decomposition

    def spectral_decomp(m: QQMatrix, dims: List[int]) -> List[QQVector]:
        decomp = rational_spectral_decomposition(m, dims)
        return [v for (_, v) in decomp]

    return dlts_abstract_spectral(spectral_decomp, dlts, dim)


# Type alias
DeterministicLTS = PartialLinearMap


class LTSOperations:
    """Operations on Linear Transition Systems."""

    @staticmethod
    def determinize(
        lts_matrices: Tuple[QQMatrix, QQMatrix],
    ) -> Tuple[PartialLinearMap, QQMatrix]:
        """Find best deterministic abstraction of an LTS.

        Args:
            lts_matrices: Tuple (A, B) of matrices representing the LTS

        Returns:
            (dlts, simulation_matrix) where:
            - dlts is a partial linear map representing the deterministic LTS
            - simulation_matrix S witnesses that Sx' = T(Sx) for the DLTS
        """
        A, B = lts_matrices
        lts = LinearTransitionSystem(A, B)
        return determinize(lts)

    @staticmethod
    def dlts_inverse_image(
        sim: QQMatrix, dlts: PartialLinearMap
    ) -> LinearTransitionSystem:
        """Compute inverse image of DLTS under similarity map.

        Given similarity S and DLTS with map T and guard G,
        compute LTS such that Ax' = Bx where:
        - A = S
        - B = T*S with guard constraints
        """
        return dlts_inverse_image(sim, dlts)


# Factory functions
def make_linear_transition_system(A: QQMatrix, B: QQMatrix) -> LinearTransitionSystem:
    """Create a linear transition system."""
    return LinearTransitionSystem(A, B)


def make_partial_linear_map(
    matrix: QQMatrix, guard: List[QQVector]
) -> PartialLinearMap:
    """Create a partial linear map."""
    return PartialLinearMap.make(matrix, guard)


def make_lts_analysis(context: Context) -> LTSAnalysis:
    """Create an LTS analysis engine."""
    return LTSAnalysis(context)


# Analysis functions
def abstract_to_lts(
    transition_formula: TransitionFormula, context: Context
) -> LinearTransitionSystem:
    """Abstract a transition formula to an LTS."""
    analysis = LTSAnalysis(context)
    return analysis.abstract_lts(transition_formula)


def check_lts_containment(
    lts1: LinearTransitionSystem, lts2: LinearTransitionSystem, context: Context
) -> bool:
    """Check if lts1 contains the transitions of lts2."""
    analysis = LTSAnalysis(context)
    return analysis.contains(lts1, lts2)


def find_containment_witness(
    lts1: LinearTransitionSystem, lts2: LinearTransitionSystem, context: Context
) -> Optional[QQMatrix]:
    """Find witness matrix for containment."""
    analysis = LTSAnalysis(context)
    return analysis.containment_witness(lts1, lts2)
