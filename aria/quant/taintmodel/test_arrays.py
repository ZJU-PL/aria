"""Utilities and definitions for aria.quant.taintmodel.test_arrays."""

import pytest
from z3 import *
from aria.quant.taintmodel.taint import infer_sic


def _assert_equiv(lhs, rhs):
    s = Solver()
    s.add(lhs != rhs)
    assert s.check() == unsat


def test_array_store_select_eliminates_target():
    a = Array("a", IntSort(), IntSort())
    i, j, v = Ints("i j v")
    expr = Select(Store(a, i, v), j)
    sic = infer_sic(expr, {i, j})
    expected = infer_sic(If(i == j, v, Select(a, j)), {i, j})
    _assert_equiv(sic, expected)


def test_array_equality_sic_all_targets():
    a = Array("a", IntSort(), IntSort())
    b = Array("b", IntSort(), IntSort())
    i, v = Ints("i v")
    expr = Store(a, i, v) == b
    sic = infer_sic(expr, {i, v})
    _assert_equiv(sic, BoolVal(False))


def test_store_independence_on_parts():
    a = Array("a", IntSort(), IntSort())
    i, v = Ints("i v")
    expr = Store(a, i, v)
    sic = infer_sic(expr, {i, v})
    _assert_equiv(sic, BoolVal(False))


def test_select_const_array_inherits_value_sic():
    i, v = Ints("i v")
    expr = Select(K(IntSort(), v), i)
    sic = infer_sic(expr, {i})
    _assert_equiv(sic, BoolVal(True))


def test_select_array_ite_of_const_arrays_is_index_independent():
    c = Bool("c")
    i, v, w = Ints("i v w")
    expr = Select(If(c, K(IntSort(), v), K(IntSort(), w)), i)
    sic = infer_sic(expr, {i})
    _assert_equiv(sic, BoolVal(True))
