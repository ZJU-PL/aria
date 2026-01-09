import z3

from pc_dsl.easy_z3 import (
    Solver,
    BV,
    BVVal,
    BVConcat,
    Extract,
    ZeroExt,
    SignExt,
    LShR,
    RotateLeft,
    RotateRight,
    RepeatBV,
    BV2Int,
    Int2BV,
    Contains,
    PrefixOf,
    SuffixOf,
    Substring,
    IndexOf,
    Replace,
    StrToInt,
    IntToStr,
)


class BitVectorOps(Solver):
    x: ("bv", 8)
    y: ("bv", 4)

    # Upper nibble of x is 0xA, lower nibble is unconstrained; y is 0xB
    assert BVConcat(Extract(7, 4, x), y) == BVVal(0xAB, 8)
    assert ZeroExt(4, y) == BVVal(0x0B, 8)
    # Check shifting/rotation/repetition remain consistent
    assert LShR(BVVal(0b10000000, 8), 1) == BVVal(0b01000000, 8)
    assert RotateLeft(BVVal(0b00000001, 8), 1) == BVVal(0b00000010, 8)
    assert RotateRight(BVVal(0b00000001, 8), 1) == BVVal(0b10000000, 8)
    assert RepeatBV(2, BVVal(0b11, 2)) == BVVal(0b1111, 4)
    assert SignExt(4, BVVal(0x8, 4)) == BVVal(0xF8, 8)
    assert Int2BV(8, BV2Int(BVVal(0x2, 4))) == BVVal(0x02, 8)


def test_bitvector_ops():
    assert BitVectorOps.check() == z3.sat


class StringOps(Solver):
    s: str

    assert PrefixOf("hi", s)
    assert SuffixOf("!", s)
    assert Contains(s, "hi!")
    assert Substring("abcdef", 1, 3) == "bcd"
    assert IndexOf("hello world", "world") == 6
    assert Replace("foo bar", "bar", "!") == "foo !"
    assert StrToInt("123") == 123
    assert IntToStr(42) == "42"


def test_string_ops():
    assert StringOps.check() == z3.sat


class UnsatCoreCase(Solver):
    x: int


def test_unsat_core_and_options():
    UnsatCoreCase.reset()
    UnsatCoreCase.set(unsat_core=True, timeout=0)
    dv = UnsatCoreCase.dsl_vars()
    UnsatCoreCase.add(z3.Implies(z3.Bool("a1"), dv["x"] > 0))
    UnsatCoreCase.add(z3.Implies(z3.Bool("a2"), dv["x"] < 0))
    res = UnsatCoreCase.check("a1", "a2")
    assert res == z3.unsat
    core = {str(c) for c in UnsatCoreCase.unsat_core() or ()}
    assert {"a1", "a2"}.issubset(core)
