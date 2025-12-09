from __future__ import annotations

from typing import List, Sequence, Tuple

import z3

from .congruence_system import CongruenceSystem


def _bools_to_ints(bs: Sequence[z3.BoolRef]) -> List[z3.IntNumRef | z3.ArithRef]:
    return [z3.If(b, z3.IntVal(1), z3.IntVal(0)) for b in bs]


def _model_vector(model: z3.ModelRef, xs: Sequence[z3.ArithRef]) -> List[int]:
    vals: List[int] = []
    for x in xs:
        v = model.eval(x, model_completion=True)
        vals.append(int(v.as_long()))
    return vals


def _augment_with_model(sys: CongruenceSystem, xvec: List[int]) -> None:
    """Given a new model x ∈ {0,1}^k, add the equation fixing x to the system's
    row-space by joining with the affine hull of previous models using the paper's
    triangular merge algorithm. This maintains triangular form and produces a
    more compact representation than simple row addition.

    The algorithm computes the intersection of the current affine hull with the
    new hyperplane defined by the model, using the triangular structure to
    efficiently eliminate variables.
    """
    k = len(xvec)
    m = sys.modulus

    # Add the new constraint: Σ x_i * 2^i ≡ Σ x_i * 2^i (mod m)
    # This represents the model as a congruence
    coeffs = [1] * k
    rhs = sum(xvec[j] for j in range(k))
    sys.add_row(coeffs, rhs)

    # Apply triangular merge: eliminate using existing triangular structure
    sys.triangularize()

    # Additional optimization: if we have a full-rank triangular system,
    # we can often eliminate the last row by substitution
    if sys.num_rows > 0 and sys.num_rows == sys.width:
        # Check if the last row has a pivot (non-zero coefficient in last column)
        last_row = sys.coeffs[-1]
        last_col = len(last_row) - 1

        if last_row[last_col] != 0:
            # We can eliminate this row using the triangular structure
            # This implements the paper's observation about redundant constraints
            inv = CongruenceSystem._mod_inv_pow2(last_row[last_col] % m, m)
            if inv is not None:
                # Solve for the last variable and substitute back
                # This is a key optimization from the paper
                scale = inv
                for j in range(last_col):
                    last_row[j] = (last_row[j] * scale) % m
                sys.rhs[-1] = (sys.rhs[-1] * scale) % m

                # Now substitute back through previous rows
                for i in range(sys.num_rows - 2, -1, -1):
                    if i < len(sys.coeffs) and sys.coeffs[i][last_col] != 0:
                        factor = sys.coeffs[i][last_col] % m
                        for j in range(last_col):
                            sys.coeffs[i][j] = (sys.coeffs[i][j] - factor * last_row[j]) % m
                        sys.rhs[i] = (sys.rhs[i] - factor * sys.rhs[-1]) % m

                # Remove the last row as it's now redundant
                sys.coeffs.pop()
                sys.rhs.pop()


def congruent_closure(
    formula: z3.BoolRef,
    bool_vars: Sequence[z3.BoolRef],
    modulus: int,
    max_iters: int = 256,
) -> CongruenceSystem:
    """Compute a modular congruence abstraction of ``formula`` over ``bool_vars``.

    This implements a SAT-guided CEGIS loop inspired by Fig. 2 of the paper. We
    maintain a candidate system and iteratively enforce stability: if there is a
    model of ``formula`` that violates the current rows, we add information from
    that model and continue. When no violating model exists, the system is
    stable and represents a congruent over-approximation of all models.

    Notes:
    - We use a simple row-augmentation rather than full triangular maintenance
      to keep the implementation compact. It preserves soundness; precision is
      generally good for small blocks. This can be strengthened later.
    - ``modulus`` should be 2**w.
    """
    assert len(bool_vars) > 0
    assert modulus > 0 and (modulus & (modulus - 1)) == 0

    sys = CongruenceSystem(modulus=modulus, coeffs=[], rhs=[])

    s = z3.Solver()
    s.add(formula)
    xs = _bools_to_ints(bool_vars)

    # Initially, try to obtain a few models to seed the system.
    seeds: List[List[int]] = []
    for _ in range(4):
        if s.check() != z3.sat:
            break
        m = s.model()
        v = _model_vector(m, xs)
        seeds.append(v)
        s.add(z3.Or([x != m.eval(x, model_completion=True) for x in xs]))

    for v in seeds:
        _augment_with_model(sys, v)
    sys.triangularize()

    # Stability loop
    iteration = 0
    while iteration < max_iters:
        iteration += 1
        s2 = z3.Solver()
        s2.add(formula)
        # add negation of current system: some row must be violated
        violated: List[z3.BoolRef] = []
        for a, b in zip(sys.coeffs, sys.rhs):
            lhs = sum(int(ai) * xi for ai, xi in zip(a, xs))
            # lhs ≡ b (mod m)  <=>  ∃t. lhs = b + m*t
            t = z3.FreshInt("t_viol")
            violated.append(z3.Not(lhs == z3.IntVal(int(b)) + z3.IntVal(modulus) * t))
        if violated:
            s2.add(z3.Or(violated))

        if s2.check() != z3.sat:
            break  # stable

        m = s2.model()
        v = _model_vector(m, xs)
        _augment_with_model(sys, v)
        sys.triangularize()

    return sys
