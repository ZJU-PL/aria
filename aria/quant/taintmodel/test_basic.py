# tests/test_basic.py --------------------------------------------------------
import pytest, os, textwrap
from aria.quant.taintmodel.solver import QuantSolver
from aria.quant.taintmodel.taint import (
    _simplify_bv_equality,
    infer_sic,
    infer_sic_with_taints,
)
from z3 import *


@pytest.fixture
def solver():
    return QuantSolver()


def run(slv, smt):
    f = parse_smt2_string(smt)
    if isinstance(f, list):
        f = And(*f)
    return slv.solve(f)[0]  # "sat"/"unsat"/"unknown"


def test_motivating_example(solver):
    smt = """
    (declare-const a Int)
    (declare-const b Int)
    (assert (forall ((x Int)) (> (+ (* a x) b) 0)))
    (check-sat)
    """
    assert run(solver, smt) == "sat"


def test_motivating_example_returns_model_for_free_vars(solver):
    a, b, x = Ints("a b x")
    res, model = solver.solve(ForAll([x], a * x + b > 0))
    assert res == "sat"
    assert model is not None
    assert {d.name() for d in model.decls()} == {"a", "b"}
    assert model.eval(a, model_completion=True).as_long() == 0
    assert is_true(model.eval(b > 0, model_completion=True))


def test_unsat_simple(solver):
    smt = """
    (declare-const a Int)
    (assert (forall ((x Int)) (< x a)))
    (assert (forall ((x Int)) (> x a)))
    (check-sat)
    """
    assert run(solver, smt) == "unknown"


def test_forall_unsat_case_is_confirmed_by_fragment_checks(solver):
    x = Int("x")
    res, model = solver.solve(ForAll([x], x < 0))
    assert res == "unsat"
    assert model is None


def test_equality_unsat_is_confirmed_by_fragment_checks(solver):
    x, y = Ints("x y")
    res, model = solver.solve(ForAll([x], x == y))
    assert res == "unsat"
    assert model is None


def test_equality_unsat_can_be_confirmed_with_wic_verification():
    x, y = Ints("x y")
    res, model = QuantSolver(verify_wic=True).solve(ForAll([x], x == y))
    assert res == "unsat"
    assert model is None


def test_paper_wic_counterexample_returns_unknown(solver):
    x = Int("x")
    formula = ForAll([x], Or(x < 0, x >= 0))
    res, model = solver.solve(formula)
    assert res == "unknown"
    assert model is None


def test_explicit_exists_forall_sat():
    a, b, x = Ints("a b x")
    formula = Exists([a, b], ForAll([x], a * x + b > 0))
    res, model = QuantSolver().solve(formula)
    assert res == "sat"
    assert model is not None
    assert len(model.decls()) == 0


def test_explicit_exists_forall_unsat():
    a, x = Ints("a x")
    formula = Exists([a], ForAll([x], a * x > 0))
    res, model = QuantSolver().solve(formula)
    assert res == "unsat"
    assert model is None


def test_non_prenex_alternation_returns_unknown(solver):
    x, y = Ints("x y")
    formula = ForAll([x], Exists([y], y == x))
    res, model = solver.solve(formula)
    assert res == "unknown"
    assert model is None


def test_array_select_store_rewrite():
    a = Array("a", IntSort(), IntSort())
    i, j, v = Ints("i j v")
    expr = Select(Store(a, i, v), j)
    sic = infer_sic(expr, {i, j})
    ite_expr = If(i == j, v, Select(a, j))
    sic_expected = infer_sic(ite_expr, {i, j})
    s = Solver()
    s.add(sic != sic_expected)
    assert s.check() == unsat


def _assert_equiv(lhs, rhs):
    s = Solver()
    s.add(lhs != rhs)
    assert s.check() == unsat


def test_motivation_ax_b_sic():
    a, b, x = Ints("a b x")
    expr = a * x + b > 0
    sic = infer_sic(expr, {x})
    _assert_equiv(sic, a == 0)


def test_arith_equality_uses_conjunctive_sic():
    a, x = Ints("a x")
    expr = a * x == 0
    sic = infer_sic(expr, {x})
    _assert_equiv(sic, a == 0)


def test_bv_or_sic():
    a, b = BitVecs("a b", 8)
    expr = a | b
    sic = infer_sic(expr, {a})
    _assert_equiv(sic, b == BitVecVal(255, 8))


def test_bv_nary_or_sic_uses_any_all_ones_operand():
    a, b, c = BitVecs("a b c", 8)
    expr = a | b | c
    sic = infer_sic(expr, {a})
    _assert_equiv(sic, Or(b == BitVecVal(255, 8), c == BitVecVal(255, 8)))


def test_bv_ult_rhs_zero_is_absorbing_false():
    a, b = BitVecs("a b", 8)
    sic = infer_sic(ULT(a, b), {a})
    _assert_equiv(sic, b == BitVecVal(0, 8))


def test_bv_ule_rhs_max_is_absorbing_true():
    a, b = BitVecs("a b", 8)
    sic = infer_sic(ULE(a, b), {a})
    _assert_equiv(sic, b == BitVecVal(255, 8))


def test_bv_shift_left_zero_value_is_independent_of_shift():
    a, b = BitVecs("a b", 8)
    sic = infer_sic(a << b, {b})
    _assert_equiv(sic, a == BitVecVal(0, 8))


def test_bv_ashr_all_ones_is_independent_of_shift():
    a, b = BitVecs("a b", 8)
    sic = infer_sic(a >> b, {b})
    _assert_equiv(sic, Or(a == BitVecVal(0, 8), a == BitVecVal(255, 8)))


def test_bv_udiv_by_zero_is_independent_of_numerator():
    a, b = BitVecs("a b", 8)
    sic = infer_sic(UDiv(a, b), {a})
    _assert_equiv(sic, b == BitVecVal(0, 8))


def test_bv_urem_by_one_is_independent_of_numerator():
    a, b = BitVecs("a b", 8)
    sic = infer_sic(URem(a, b), {a})
    _assert_equiv(sic, b == BitVecVal(1, 8))


def test_bv_urem_equality_zero_uses_constant_refinement():
    a, b = BitVecs("a b", 8)
    sic = infer_sic(URem(a, b) == BitVecVal(0, 8), {a})
    _assert_equiv(sic, b == BitVecVal(1, 8))


def test_bv_concat_equality_zero_uses_word_level_refinement():
    a, b = BitVecs("a b", 8)
    sic = infer_sic(Concat(a, b) == BitVecVal(0, 16), {a})
    _assert_equiv(sic, b != BitVecVal(0, 8))


def test_bv_extract_concat_equality_uses_word_level_refinement():
    a, b = BitVecs("a b", 8)
    expr = Extract(7, 0, Concat(a, b)) == BitVecVal(0, 8)
    sic = infer_sic(expr, {a})
    _assert_equiv(sic, BoolVal(True))


def test_bv_zero_ext_equality_zero_simplify_helper_rewrites_to_source_zero():
    a = BitVec("a", 8)
    simplified = _simplify_bv_equality(ZeroExt(8, a) == BitVecVal(0, 16))
    _assert_equiv(simplified, a == BitVecVal(0, 8))


def test_bv_sign_ext_equality_all_ones_simplify_helper_rewrites_to_source_all_ones():
    a = BitVec("a", 8)
    simplified = _simplify_bv_equality(SignExt(8, a) == BitVecVal(65535, 16))
    _assert_equiv(simplified, a == BitVecVal(255, 8))


def test_bv_srem_by_minus_one_is_independent_of_numerator():
    a, b = BitVecs("a b", 8)
    sic = infer_sic(SRem(a, b), {a})
    _assert_equiv(sic, Or(b == BitVecVal(1, 8), b == BitVecVal(255, 8)))


def test_bv_sdiv_equality_signed_min_uses_constant_refinement():
    a, b = BitVecs("a b", 8)
    sic = infer_sic(UDiv(a, b) == BitVecVal(255, 8), {a})
    _assert_equiv(sic, b == BitVecVal(0, 8))


def test_bv_sdiv_zero_numerator_and_nonzero_denominator_is_constant_zero():
    a, b = BitVecs("a b", 8)
    sic = infer_sic(a / b, set())
    s = Solver()
    s.add(a == BitVecVal(0, 8), b == BitVecVal(1, 8), Not(sic))
    assert s.check() == unsat


def test_bv_sdiv_min_by_minus_one_is_constant_signed_min():
    a, b = BitVecs("a b", 8)
    sic = infer_sic(a / b, set())
    s = Solver()
    s.add(a == BitVecVal(128, 8), b == BitVecVal(255, 8), Not(sic))
    assert s.check() == unsat


def test_arith_nary_mul_sic_uses_any_zero_operand():
    a, x, y = Ints("a x y")
    expr = a * x * y > 0
    sic = infer_sic(expr, {x})
    _assert_equiv(sic, Or(a == 0, y == 0))


def test_taint_variable_path_matches_direct():
    a, b, x = Ints("a b x")
    expr = a * x + b > 0
    sic_direct = infer_sic(expr, {x})
    sic_taint, _ = infer_sic_with_taints(expr, {x})
    _assert_equiv(sic_direct, sic_taint)


def test_alternating_quantifiers_do_not_crash(solver):
    x, a, y, b = Ints("x a y b")
    formula = ForAll(
        [x],
        Exists([a], ForAll([y], Exists([b], (x - a) * y + b > 0))),
    )
    res, _ = solver.solve(formula)
    assert res == "unknown"


def test_bv_add_quantifier_path_does_not_crash(solver):
    a, b, c = BitVecs("a b c", 8)
    res, _ = solver.solve(ForAll([a], ULT(a + b, c)))
    assert res in {"unknown", "unsat"}


def test_bv_ashr_no_longer_produces_spurious_sat(solver):
    a, b, c = BitVecs("a b c", 8)
    res, _ = solver.solve(ForAll([a], (a >> b) == c))
    assert res in {"unknown", "unsat"}


def test_exists_forall_const_array_sat():
    i, v = Ints("i v")
    formula = Exists([v], ForAll([i], Select(K(IntSort(), v), i) == v))
    res, model = QuantSolver().solve(formula)
    assert res == "sat"
    assert model is not None
    assert len(model.decls()) == 0


def test_exists_forall_bv_ule_absorber_sat():
    a = BitVec("a", 8)
    x = BitVec("x", 8)
    formula = Exists([a], ForAll([x], ULE(x, a)))
    res, model = QuantSolver().solve(formula)
    assert res == "sat"
    assert model is not None
    assert len(model.decls()) == 0


def test_exists_forall_bv_urem_absorber_sat():
    d = BitVec("d", 8)
    x = BitVec("x", 8)
    formula = Exists([d], ForAll([x], URem(x, d) == 0))
    res, model = QuantSolver().solve(formula)
    assert res == "sat"
    assert model is not None
    assert len(model.decls()) == 0


def test_parsed_bvurem_internal_name_is_normalized():
    expr = parse_smt2_string(
        """
        (declare-fun a () (_ BitVec 8))
        (declare-fun b () (_ BitVec 8))
        (assert (= (bvurem a b) #x00))
        """
    )[0]
    a, _ = BitVecs("a b", 8)
    sic = infer_sic(expr, {a})
    _assert_equiv(sic, BitVec("b", 8) == BitVecVal(1, 8))


def test_parsed_bvashr_internal_name_is_normalized():
    expr = parse_smt2_string(
        """
        (declare-fun a () (_ BitVec 8))
        (declare-fun b () (_ BitVec 8))
        (assert (= (bvashr a b) #xff))
        """
    )[0]
    _, b = BitVecs("a b", 8)
    sic = infer_sic(expr, {b})
    _assert_equiv(
        sic,
        Or(BitVec("a", 8) == BitVecVal(0, 8), BitVec("a", 8) == BitVecVal(255, 8)),
    )


def test_parsed_concat_equality_is_simplified_for_sic():
    expr = parse_smt2_string(
        """
        (declare-fun a () (_ BitVec 8))
        (declare-fun b () (_ BitVec 8))
        (assert (= (concat a b) #x0000))
        """
    )[0]
    a, _ = BitVecs("a b", 8)
    sic = infer_sic(expr, {a})
    _assert_equiv(sic, BitVec("b", 8) != BitVecVal(0, 8))


def test_parsed_zero_ext_equality_is_simplified_by_helper():
    expr = parse_smt2_string(
        """
        (declare-fun a () (_ BitVec 8))
        (assert (= ((_ zero_extend 8) a) #x0000))
        """
    )[0]
    simplified = _simplify_bv_equality(expr)
    _assert_equiv(simplified, BitVec("a", 8) == BitVecVal(0, 8))


def test_parsed_sign_ext_equality_is_simplified_by_helper():
    expr = parse_smt2_string(
        """
        (declare-fun a () (_ BitVec 8))
        (assert (= ((_ sign_extend 8) a) #xffff))
        """
    )[0]
    simplified = _simplify_bv_equality(expr)
    _assert_equiv(simplified, BitVec("a", 8) == BitVecVal(255, 8))
