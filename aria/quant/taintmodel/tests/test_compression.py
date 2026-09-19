"""Tests for certified failure-conditioned counterexample compression."""

import pytest
from z3 import *

from aria.quant.taintmodel.compression import (
    CounterexampleCompression,
    certify_compression,
    propose_joint_compression,
)
from aria.quant.taintmodel.solver import QuantSolver


def _assert_equiv(lhs, rhs):
    solver = Solver()
    solver.add(lhs != rhs)
    assert solver.check() == unsat


class _QFCheckingSolver(QuantSolver):
    def _check_qf(self, formula):
        assert not self._contains_quantifier(formula)
        return super()._check_qf(formula)


def _truth_table_formula(x, y, table, values):
    rows = []
    for x_index, x_value in enumerate(values):
        for y_index, y_value in enumerate(values):
            bit = 1 << (2 * x_index + y_index)
            if table & bit:
                rows.append(And(x == x_value, y == y_value))
    return Or(*rows)


def _truth_table_has_exists_forall_solution(table):
    return any(
        all(table & (1 << (2 * x_index + y_index)) for y_index in range(2))
        for x_index in range(2)
    )


@pytest.mark.parametrize("use_bitvectors", [False, True])
def test_exhaustive_one_bit_formulas_match_truth_tables(use_bitvectors):
    if use_bitvectors:
        x, y = BitVecs("truth_x truth_y", 1)
        values = [BitVecVal(0, 1), BitVecVal(1, 1)]
    else:
        x, y = Bools("truth_x truth_y")
        values = [BoolVal(False), BoolVal(True)]

    for table in range(16):
        matrix = _truth_table_formula(x, y, table, values)
        status, _ = _QFCheckingSolver().solve(Exists([x], ForAll([y], matrix)))
        expected = "sat" if _truth_table_has_exists_forall_solution(table) else "unsat"
        assert status == expected, f"truth table 0x{table:x}"


def test_identity_compression_is_always_certified():
    x, y = BitVecs("x y", 4)
    matrix = ULT(x + y, x ^ y)
    identity = CounterexampleCompression.identity([y])

    status, model = certify_compression(matrix, [x], identity)

    assert status == "valid"
    assert model is None
    assert identity.residual_bit_count == 4


def test_joint_mask_reconstruction_eliminates_the_universal_word():
    a, b, mask, cutoff, y = BitVecs("a b mask cutoff y", 8)
    matrix = Or(
        (y & mask) != (a & mask),
        (y & ~mask) != (b & ~mask),
        ULT(y, cutoff),
    )

    compression = propose_joint_compression(
        matrix,
        [a, b, mask, cutoff],
        [y],
    )

    assert compression is not None
    assert compression.residual_bit_count == 0
    _assert_equiv(
        compression.reconstruction[0],
        (a & mask) | (b & ~mask),
    )
    assert (
        certify_compression(
            matrix,
            [a, b, mask, cutoff],
            compression,
        )[0]
        == "valid"
    )


def test_joint_slice_reconstruction_assembles_a_universal_word():
    high, low = BitVecs("high low", 4)
    cutoff, y = BitVecs("slice_cutoff slice_y", 8)
    matrix = Or(
        Extract(7, 4, y) != high,
        Extract(3, 0, y) != low,
        ULT(y, cutoff),
    )

    compression = propose_joint_compression(matrix, [high, low, cutoff], [y])

    assert compression is not None
    assert compression.residual_bit_count == 0
    _assert_equiv(compression.reconstruction[0], Concat(high, low))
    assert certify_compression(matrix, [high, low, cutoff], compression)[0] == "valid"


def test_dependency_cycle_keeps_one_residual_word():
    x, y0, y1 = BitVecs("x y0 y1", 8)
    matrix = Or(y0 != y1, ULT(y0, x))

    compression = propose_joint_compression(matrix, [x], [y0, y1])

    assert compression is not None
    assert compression.residual_bit_count == 8
    assert certify_compression(matrix, [x], compression)[0] == "valid"


def test_whole_formula_certificate_rejects_a_missed_failure_path():
    x, y = BitVecs("x y", 2)
    matrix = Or(y == x, y == ~x)
    wrong = CounterexampleCompression(
        guard=BoolVal(True),
        universals=(y,),
        residuals=(),
        reconstruction=(x,),
        projection=(),
        name="deliberately-wrong",
    )

    status, model = certify_compression(matrix, [x], wrong)

    assert status == "counterexample"
    assert model is not None


def test_guarded_certificate_needs_to_hold_only_inside_the_guard():
    x, y = Bools("guard_x guard_y")
    matrix = If(x, y, Not(y))
    guarded = CounterexampleCompression(
        guard=x,
        universals=(y,),
        residuals=(),
        reconstruction=(BoolVal(False),),
        projection=(),
    )
    unguarded = CounterexampleCompression(
        guard=BoolVal(True),
        universals=(y,),
        residuals=(),
        reconstruction=(BoolVal(False),),
        projection=(),
    )

    assert certify_compression(matrix, [x], guarded)[0] == "valid"
    assert certify_compression(matrix, [x], unguarded)[0] == "counterexample"


def test_hidden_eliminated_dependency_is_rejected_structurally():
    x, y = BitVecs("x y", 4)
    matrix = y == x
    malformed = CounterexampleCompression(
        guard=BoolVal(True),
        universals=(y,),
        residuals=(),
        reconstruction=(y,),
        projection=(),
    )

    with pytest.raises(ValueError, match="hidden dependency"):
        certify_compression(matrix, [x], malformed)


def test_solver_learns_and_uses_zero_residual_compression():
    a, b, mask, cutoff, y = BitVecs("a b mask cutoff y", 8)
    matrix = Or(
        (y & mask) != (a & mask),
        (y & ~mask) != (b & ~mask),
        ULT(y, cutoff),
    )
    solver = QuantSolver()

    status, model = solver.solve(ForAll([y], matrix))

    assert status == "sat"
    assert model is not None
    reconstructed = (a & mask) | (b & ~mask)
    assert is_true(
        simplify(model.eval(ULT(reconstructed, cutoff), model_completion=True))
    )
    assert solver.last_statistics["compressions_certified"] == 1
    assert solver.last_statistics["compressed_verifications"] >= 1


def test_qf_cegis_exhausts_a_finite_bv_candidate_space():
    x, y = BitVecs("x y", 2)
    solver = QuantSolver(enable_compression=False)

    status, model = solver.solve(Exists([x], ForAll([y], x != y)))

    assert status == "unsat"
    assert model is None
    assert solver.last_statistics["refinements"] == 4
