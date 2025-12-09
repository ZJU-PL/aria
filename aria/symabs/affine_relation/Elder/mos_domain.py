"""MOS (Müller-Olm/Seidl) Domain Implementation.

This module implements the MOS domain from Elder et al.'s paper "Abstract Domains
of Affine Relations". The MOS domain represents affine transformers as sets of
matrices and provides operations for computing the α function symbolically.
"""

import z3
import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Union
from .matrix_ops import Matrix


class MOS:
    """Represents an element in the MOS (Müller-Olm/Seidl) domain.

    An MOS element is a set of (k+1)-by-(k+1) matrices over Z_{2^w}, where each
    matrix T represents a one-vocabulary transformer of the form x' = x·M + b,
    or equivalently, [x' x] = [x 1]·T.

    The concretization of an MOS element B is the union of the affine spaces
    represented by the transformers in B.
    """

    def __init__(self, matrices: Optional[List[Matrix]] = None, w: int = 32):
        """Initialize an MOS element.

        Args:
            matrices: List of matrices representing the affine transformers
            w: Word size (default 32 bits)
        """
        self.w = w
        self.matrices = matrices if matrices is not None else []
        self.modulus = 2**w

    def __repr__(self) -> str:
        return f"MOS(matrices={len(self.matrices)}, w={self.w})"

    def is_empty(self) -> bool:
        """Check if this MOS element represents the empty set."""
        return len(self.matrices) == 0

    def is_bottom(self) -> bool:
        """Check if this is the bottom element (empty set)."""
        return self.is_empty()

    def is_top(self) -> bool:
        """Check if this represents the top element (universal set)."""
        # Top element would be all possible affine transformers
        # For practical purposes, we consider a non-empty set as not top
        return False

    def add_matrix(self, matrix: Matrix) -> None:
        """Add a matrix to this MOS element."""
        self.matrices.append(matrix)

    def union(self, other: 'MOS') -> 'MOS':
        """Compute the union of two MOS elements."""
        if self.is_empty():
            return other.copy()
        if other.is_empty():
            return self.copy()

        result = self.copy()
        for matrix in other.matrices:
            if matrix not in result.matrices:
                result.add_matrix(matrix)
        return result

    def copy(self) -> 'MOS':
        """Create a copy of this MOS element."""
        new_matrices = [m.copy() for m in self.matrices]
        return MOS(new_matrices, self.w)

    def concretize(self) -> str:
        """Return a symbolic representation of the concretization."""
        if self.is_empty():
            return "∅"

        terms = []
        for i, matrix in enumerate(self.matrices):
            # Extract the affine transformation from the matrix
            # For a matrix T = [M b; 0 1], we have x' = x·M + b
            k = matrix.rows - 1
            if k <= 0:
                continue

            M = matrix.data[:k, :k]
            b = matrix.data[:k, k]

            term = f"T_{i}: x' = x·{M}"
            if np.any(b != 0):
                term += f" + {b}"
            terms.append(term)

        return " ∪ ".join(terms)

    def __eq__(self, other) -> bool:
        """Check equality of MOS elements."""
        if not isinstance(other, MOS):
            return False
        if len(self.matrices) != len(other.matrices):
            return False

        # Simple check - in practice would need canonical representation
        return set(str(m.data) for m in self.matrices) == set(str(m.data) for m in other.matrices)


def alpha_mos(phi: z3.ExprRef, pre_vars: List[z3.ExprRef], post_vars: List[z3.ExprRef]) -> MOS:
    """Symbolic implementation of the α function for MOS.

    Uses CEGIS (CounterExample-Guided Inductive Synthesis) to compute the strongest
    affine relation that overapproximates the concrete semantics defined by phi.

    This implements a proper abstraction algorithm inspired by the existing KS
    implementation, using counterexamples to iteratively refine the abstraction.

    Args:
        phi: QFBV formula as Z3 expression
        pre_vars: Z3 pre-state variables
        post_vars: Z3 post-state variables

    Returns:
        MOS element representing the strongest affine abstraction of phi
    """
    import z3

    w = 32  # Word size - should be configurable

    # Use CEGIS approach to find the strongest MOS abstraction
    return _cegis_alpha_mos(phi, pre_vars, post_vars, w)


def _cegis_alpha_mos(formula, pre_vars, post_vars, w: int) -> MOS:
    """CEGIS-based alpha function for MOS domain.

    Uses CounterExample-Guided Inductive Synthesis to find the strongest
    set of affine transformations that overapproximates the formula.

    Algorithm:
    1. Start with empty MOS (no transformations)
    2. Check if current MOS correctly overapproximates formula
    3. If counterexample found, extract implied transformation and add it
    4. Repeat until no more counterexamples

    Args:
        formula: Z3 formula representing the concrete semantics
        pre_vars: Pre-state variables
        post_vars: Post-state variables
        w: Word size

    Returns:
        MOS element representing the strongest abstraction
    """
    import z3

    k = len(pre_vars)
    matrices = []  # List of transformation matrices

    # CEGIS loop
    max_iterations = 50  # Prevent infinite loops
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        # Check if current abstraction correctly overapproximates the formula
        # A model of the formula violates the abstraction if it doesn't satisfy
        # any of the transformation constraints

        check_solver = z3.Solver()
        check_solver.add(formula)

        # If there are transformations, the model must violate ALL of them
        # A transform T is violated if any of its component equations is violated
        if matrices:
            per_transform_violation = []
            for matrix in matrices:
                M = matrix.data[:k, :k]
                b = matrix.data[:k, k]
                component_violations = []
                for i in range(k):
                    lhs = post_vars[i]
                    for j in range(k):
                        if M[i, j] != 0:
                            lhs = lhs - (M[i, j] * pre_vars[j])
                    lhs = lhs - b[i]
                    component_violations.append(lhs != 0)
                per_transform_violation.append(z3.Or(component_violations))
            check_solver.add(z3.And(per_transform_violation))

        # Check if there's a counterexample
        if check_solver.check() == z3.sat:
            # Found a counterexample not covered by existing transforms
            # Use SMT-based synthesis to find a new affine transform T: x' = x*M + b
            model = check_solver.model()
            seed_pre = [model.eval(var, model_completion=True).as_long() for var in pre_vars]
            seed_post = [model.eval(var, model_completion=True).as_long() for var in post_vars]

            new_matrix = _synthesize_affine_transform_smt(
                formula=formula,
                pre_vars=pre_vars,
                post_vars=post_vars,
                w=w,
                seed_sample=(seed_pre, seed_post),
            )
            if new_matrix is not None:
                matrices.append(new_matrix)
            else:
                # If we can't extract a transformation, we're stuck
                # In practice, this shouldn't happen for valid affine formulas
                break
        else:
            # No counterexample found - current abstraction is correct
            break

    # Apply Howell form to canonicalize all matrices
    canonical_matrices = []
    for matrix in matrices:
        # TODO: Fix Howell form implementation for transformation matrices
        # For now, use the original matrix
        canonical_matrices.append(matrix)

    return MOS(canonical_matrices, w)


def _synthesize_affine_transform_smt(
    formula: z3.ExprRef,
    pre_vars: List[z3.ExprRef],
    post_vars: List[z3.ExprRef],
    w: int,
    seed_sample: Optional[Tuple[List[int], List[int]]] = None,
    max_samples: int = 8,
    max_iters: int = 20,
    timeout_ms: Optional[int] = None,
) -> Optional[Matrix]:
    """Synthesize an affine transform T (M, b) such that phi => x' = x*M + b.

    Uses an inner CEGIS loop:
      1) Fit unknown coefficients (M, b) to a set of concrete samples satisfying phi
      2) Validate: search for a model of phi that violates the current transform
      3) If found, add as a new sample and refit; otherwise return the transform
    """
    import z3

    k = len(pre_vars)
    modulus = 2**w
    BV = lambda v: z3.BitVecVal(int(v) % modulus, w)

    # Sample store: list of tuples (pre_vals, post_vals), both lists of ints
    samples: List[Tuple[List[int], List[int]]] = []
    if seed_sample is not None:
        samples.append(seed_sample)

    def fit_transform(samples_local: List[Tuple[List[int], List[int]]]) -> Optional[Tuple[List[List[int]], List[int]]]:
        """Fit M (k x k) and b (k) over given samples using bit-vector arithmetic."""
        s = z3.Solver()
        if timeout_ms is not None:
            s.set(timeout=timeout_ms)

        M = [[z3.BitVec(f"M_{i}_{j}", w) for j in range(k)] for i in range(k)]
        b = [z3.BitVec(f"b_{i}", w) for i in range(k)]

        # Constrain coefficients to be within modulus automatically (BitVec ensures wrap-around)
        # Add constraints per sample
        for (pre_vals, post_vals) in samples_local:
            x = [BV(v) for v in pre_vals]
            xp = [BV(v) for v in post_vals]
            for i in range(k):
                acc = BV(0)
                for j in range(k):
                    acc = acc + (M[i][j] * x[j])
                acc = acc + b[i]
                s.add(acc == xp[i])

        if s.check() != z3.sat:
            return None
        m = s.model()

        # Extract concrete coefficients
        M_int: List[List[int]] = [[m.eval(M[i][j], model_completion=True).as_long() for j in range(k)] for i in range(k)]
        b_int: List[int] = [m.eval(b[i], model_completion=True).as_long() for i in range(k)]
        return M_int, b_int

    def build_matrix(M_int: List[List[int]], b_int: List[int]) -> Matrix:
        data = np.zeros((k + 1, k + 1), dtype=object)
        for i in range(k):
            for j in range(k):
                data[i, j] = M_int[i][j] % modulus
            data[i, k] = b_int[i] % modulus
        data[k, k] = 1
        return Matrix(data, modulus)

    def transform_violated_expr(M_int: List[List[int]], b_int: List[int]) -> z3.ExprRef:
        # Build violation: Or_i (x'_i != sum_j M[i][j]*x_j + b[i])
        comps = []
        for i in range(k):
            lhs = post_vars[i]
            for j in range(k):
                coeff = BV(M_int[i][j])
                lhs = lhs - (coeff * pre_vars[j])
            lhs = lhs - BV(b_int[i])
            comps.append(lhs != BV(0))
        return z3.Or(comps)

    # Inner CEGIS loop
    iters = 0
    while iters < max_iters and len(samples) <= max_samples:
        iters += 1
        fit = fit_transform(samples)
        if fit is None:
            # If we cannot fit with current samples, try to add a new sample
            # by simply drawing any model of the formula (diversify with disequalities)
            s = z3.Solver()
            s.add(formula)
            if s.check() != z3.sat:
                return None
            m = s.model()
            pre_vals = [m.eval(v, model_completion=True).as_long() for v in pre_vars]
            post_vals = [m.eval(v, model_completion=True).as_long() for v in post_vars]
            samples.append((pre_vals, post_vals))
            continue

        M_int, b_int = fit

        # Validate: phi ∧ violated(M,b) ? If unsat, we are done
        vsol = z3.Solver()
        if timeout_ms is not None:
            vsol.set(timeout=timeout_ms)
        vsol.add(formula)
        vsol.add(transform_violated_expr(M_int, b_int))
        if vsol.check() == z3.sat:
            # Add counterexample to samples and continue
            vm = vsol.model()
            pre_vals = [vm.eval(v, model_completion=True).as_long() for v in pre_vars]
            post_vals = [vm.eval(v, model_completion=True).as_long() for v in post_vars]
            # Avoid duplicate samples
            if (pre_vals, post_vals) not in samples:
                samples.append((pre_vals, post_vals))
            else:
                # If duplicate, break to avoid looping
                break
            continue

        # Success: return matrix
        return build_matrix(M_int, b_int)

    return None

def create_z3_variables(variable_names: List[str], w: int = 32) -> Tuple[List[z3.ExprRef], List[z3.ExprRef]]:
    """Create Z3 variables for pre-state and post-state.

    Args:
        variable_names: List of variable names
        w: Word size in bits (default 32)

    Returns:
        Tuple of (pre_vars, post_vars) as lists of Z3 BitVec expressions
    """
    import z3

    k = len(variable_names)
    pre_vars = [z3.BitVec(f"{name}", w) for name in variable_names]
    post_vars = [z3.BitVec(f"{name}'", w) for name in variable_names]

    return pre_vars, post_vars

def _is_identity_relation(ag_matrix: Matrix) -> bool:
    """Check if an AG matrix represents an identity relation."""
    k = (ag_matrix.cols - 1) // 2

    # For identity relation, we expect exactly k rows
    if ag_matrix.rows != k:
        return False

    # Each row should represent x'_i = x_i for some i
    for i in range(k):
        row = ag_matrix[i, :]

        # Check if this row represents x'_i = x_i
        # The pattern should be: -1*x_i + 1*x'_i + 0 for other variables = 0
        found_identity_for_i = False
        for j in range(k):
            a_j = row[j]        # coefficient of x_j
            b_j = row[k + j]    # coefficient of x'_j
            constant = row[2*k]

            if (j == i and a_j == -1 and b_j == 1 and constant == 0 and
                all(row[m] == 0 for m in range(k) if m != j) and
                all(row[k + m] == 0 for m in range(k) if m != j)):
                found_identity_for_i = True
                break

        if not found_identity_for_i:
            return False

    return True


def shatter_ag(ag_matrix: Matrix) -> List[Matrix]:
    """SHATTER operation for AG domain (Theorem 3 from the paper).

    Converts an AG element G to a set of MOS matrices representing the
    individual affine transformations that make up the concretization of G.

    Args:
        ag_matrix: AG matrix where each row is a generator [coeffs_x, coeffs_x', constant]

    Returns:
        List of MOS matrices representing the shattered AG element
    """
    k = (ag_matrix.cols - 1) // 2  # Number of variables
    w = ag_matrix.modulus.bit_length() - 1
    result = []

    # For each generator row, try to extract transformation matrices
    for i in range(ag_matrix.rows):
        row = ag_matrix[i, :]

        # Extract coefficients: [x_coeffs, x'_coeffs, constant]
        x_coeffs = row[:k]
        xp_coeffs = row[k:2*k]
        constant = row[2*k]

        # Try to interpret this generator as affine transformation constraints
        # A generator Σ a_i x_i + Σ b_i x'_i + c = 0 represents that
        # for all (x, x') satisfying the relation, we have this equation

        # For simple cases like x'_j - x_j = 0, we can extract x'_j = x_j
        transformations_found = []

        for j in range(k):
            a_j = row[j]        # coefficient of x_j
            b_j = row[k + j]    # coefficient of x'_j

            # Check if this represents x'_j = x_j (i.e., -x_j + x'_j = 0)
            if (a_j == -1 and b_j == 1 and
                all(row[m] == 0 for m in range(k) if m != j) and
                all(row[k + m] == 0 for m in range(k) if m != j) and
                constant == 0):
                # This represents x'_j = x_j, create identity transformation
                matrix_data = np.eye(k + 1, dtype=object)
                transformations_found.append(Matrix(matrix_data, ag_matrix.modulus))

        # For more complex generators, we would need to solve a system of equations
        # to find the possible transformations. For now, we'll handle only simple cases.
        if transformations_found:
            result.extend(transformations_found)

        # Special case: if this looks like part of an identity relation,
        # check if all rows together represent identity transformations
        elif _is_identity_relation(ag_matrix):
            # Create a single identity transformation matrix
            matrix_data = np.eye(k + 1, dtype=object)
            result.append(Matrix(matrix_data, ag_matrix.modulus))

    # Remove duplicates (matrices that represent the same transformation)
    seen = set()
    unique_result = []
    for matrix in result:
        matrix_str = str(matrix.data.tolist())
        if matrix_str not in seen:
            seen.add(matrix_str)
            unique_result.append(matrix)

    return unique_result


def matrix_multiply(T1: Matrix, T2: Matrix) -> Matrix:
    """Multiply two transformation matrices with proper modular arithmetic."""
    if T1.cols != T2.rows:
        raise ValueError("Matrix dimensions don't match for multiplication")

    result_data = np.zeros((T1.rows, T2.cols), dtype=object)

    for i in range(T1.rows):
        for j in range(T2.cols):
            sum_val = 0
            for k in range(T1.cols):
                # Modular multiplication
                prod = (T1[i, k] * T2[k, j]) % T1.modulus
                sum_val = (sum_val + prod) % T1.modulus
            result_data[i, j] = sum_val

    return Matrix(result_data, T1.modulus)
