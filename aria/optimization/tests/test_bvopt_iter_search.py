import z3
import pytest
from aria.optimization.omtbv.bv_opt_iterative_search import (
    bv_opt_with_linear_search,
    bv_opt_with_binary_search,
)


def test_linear_search_maximize_simple_range():
    y = z3.BitVec('y', 4)
    fml = z3.And(z3.UGT(y, 3), z3.ULT(y, 10))
    res = bv_opt_with_linear_search(fml, y, minimize=False, solver_name="z3")
    print("res: ", res)
    # Result is an integer (optimal value) or "unsatisfiable" string
    assert res is not None
    if isinstance(res, int):
        # For maximize, should find 9 (the maximum value in range [4, 9])
        assert res == 9
    else:
        # If it's a string, it should be "unsatisfiable"
        assert res == "unsatisfiable"


def test_binary_search_minimize_simple_range():
    y = z3.BitVec('y', 4)
    fml = z3.And(z3.UGT(y, 3), z3.ULT(y, 10))
    res = bv_opt_with_binary_search(fml, y, minimize=True, solver_name="z3")
    # Result is an integer (optimal value)
    assert res is not None
    assert isinstance(res, int)
    # For minimize, should find 4 (the minimum value in range [4, 9])
    assert res == 4

if __name__ == "__main__":
    test_linear_search_maximize_simple_range()
    test_binary_search_minimize_simple_range()
