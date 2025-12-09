import pytest
from z3 import *
from aria.smt.lia_star import dsl, semilinear
from aria.smt.lia_star.lia_star_utils import getModel

# Test dsl.MS and related functions

def test_ms_sort_and_var():
    A = IntSort()
    ms = dsl.MS(A)
    v = Const('v', ms)
    assert dsl.is_ms_sort(ms)
    assert dsl.is_ms_var(v)

# Test card and setof

def test_card_and_setof():
    A = IntSort()
    ms = dsl.MS(A)
    v = Const('v', ms)
    c = dsl.card(v)
    assert isinstance(c, ArithRef)
    s = dsl.setof(v)
    assert isinstance(s, ExprRef)

# Test LS and SLS basic construction

def test_ls_and_sls_basic():
    phi = lambda X: And([x >= 0 for x in X])
    ls = semilinear.LS([0, 0], [[1, 0], [0, 1]], phi)
    assert isinstance(ls, semilinear.LS)
    sls = semilinear.SLS(phi, [Int('x'), Int('y')], 2)
    assert isinstance(sls, semilinear.SLS)
    assert sls.size() >= 1

# Test LS.linearCombination

def test_ls_linear_combination():
    phi = lambda X: And([x >= 0 for x in X])
    ls = semilinear.LS([0, 0], [[1, 0], [0, 1]], phi)
    L, LC = ls.linearCombination('l')
    assert len(L) == 2
    assert len(LC) == 2

# Test SLS.star and SLS.starU

def test_sls_star():
    phi = lambda X: And([x >= 0 for x in X])
    sls = semilinear.SLS(phi, [Int('x'), Int('y')], 2)
    star_formula = sls.star([Int('x'), Int('y')])
    assert isinstance(star_formula, ExprRef)
    vars, fmls = sls.starU([Int('x'), Int('y')])
    assert isinstance(vars, list)
    assert isinstance(fmls, list)

# Test getModel

def test_get_model():
    x, y = Ints('x y')
    s = Solver()
    s.add(x == 1, y == 2)
    model = getModel(s, [x, y])
    assert model == [1, 2]
    s2 = Solver()
    s2.add(x == 1, x == 2)  # 无解约束
    assert getModel(s2, [x]) is None

# Test dsl.LiaStar conversion

def test_liastar_conversion():
    x = Int('x')
    y = Int('y')
    fml = And(x >= 0, y >= 0, x + y == 1)
    lia = dsl.LiaStar()
    fml2, star_defs, star_fmls = lia.convert(fml)
    assert isinstance(fml2, ExprRef)
    assert isinstance(star_defs, list)
    assert isinstance(star_fmls, list)
