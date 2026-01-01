import pytest

z3 = pytest.importorskip("z3")

from pc_dsl.easy_z3 import Solver, Concat, Length, Store, ForAll


class ArraysStrings(Solver):
    s: str
    t: str

    assert Concat(s, t) == "ab"
    assert Length(s) == 1


def test_arrays_and_strings():
    assert ArraysStrings.check() == z3.sat
    # The model should satisfy the constraints
    assert ArraysStrings.model() is not None
    # Now test array behaviors via scoped declarations
    ArraysStrings.push()
    try:
        ArraysStrings.declare_var("A", ("array", ("bv", 8), ("bv", 8)))
        dv = ArraysStrings.dsl_vars()
        ArraysStrings.add(Store(dv["A"], 0, 7)[0] == 7)
        assert ArraysStrings.check() == z3.sat
    finally:
        ArraysStrings.pop()


class Quantified(Solver):
    x: int
    y: int

    # A simple quantified formula: forall i. i >= 0 => x + i >= x
    assert ForAll({"i": int}, (x + locals()["i"]) >= x)


def test_quantifiers_build():
    # Ensure the constraints include a ForAll
    cs = Quantified.constraints()
    assert any(isinstance(c, z3.QuantifierRef) and c.is_forall() for c in cs)
