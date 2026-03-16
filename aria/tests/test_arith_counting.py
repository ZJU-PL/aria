import pytest
import z3
from typing import cast

from aria.counting.arith.arith_counting_latte import ArithModelCounter, count_lia_models


class TestArithCounting:
    def test_single_var_bounded(self):
        x = z3.Int("x")
        formula = cast(z3.BoolRef, z3.And(x >= 0, x <= 2))
        assert count_lia_models(formula) == 3

    def test_two_vars_bounded_equality(self):
        x, y = z3.Ints("x y")
        formula = cast(z3.BoolRef, z3.And(x + y == 2, x >= 0, y >= 0, x <= 2, y <= 2))
        assert count_lia_models(formula) == 3

    def test_unsat_formula(self):
        x = z3.Int("x")
        formula = cast(z3.BoolRef, z3.And(x > 0, x < 0))
        assert count_lia_models(formula) == 0

    def test_projection_count(self):
        x, y = z3.Ints("x y")
        formula = cast(z3.BoolRef, z3.And(x + y == 2, x >= 0, y >= 0, x <= 2, y <= 2))
        counter = ArithModelCounter()
        result = counter.count_models(formula=formula, variables=[x], method="auto")
        assert result.status == "exact"
        assert result.count == 3

    def test_unbounded_rejected(self):
        x = z3.Int("x")
        formula = cast(z3.BoolRef, z3.And(x >= 0))
        with pytest.raises(ValueError, match="unbounded"):
            count_lia_models(formula)

    def test_real_formula_rejected(self):
        r = z3.Real("r")
        formula = cast(z3.BoolRef, z3.And(r >= 0, r <= 1))
        with pytest.raises(ValueError, match="unsupported"):
            count_lia_models(formula)
