import pytest
import z3
from arlib.optimization.omtbv.boxed.bv_boxed_seq import solve_boxed_sequential


def test_boxed_sequential_two_objectives():
    """Test sequential boxed optimization: optimize objectives one by one.

    Sequential optimization means:
    1. First optimize x independently to its maximum (5)
    2. Then fix x=5 and optimize y independently to its maximum (5)
    """
    x = z3.BitVec('x', 4)
    y = z3.BitVec('y', 4)

    # Constraint: x <= 5, y <= 5
    formula = z3.And(z3.ULE(x, 5), z3.ULE(y, 5))
    objectives = [x, y]

    # Test with iterative binary search engine
    results = solve_boxed_sequential(formula, objectives, minimize=False,
                                     engine="iter", solver_name="z3-bs")

    assert results is not None
    assert len(results) == 2
    assert all(r is not None for r in results)
    # Verify sequential optimization: first objective maximized independently
    assert results[0] == 5
    # Second objective maximized after fixing first objective
    assert results[1] == 5


def test_boxed_sequential_minimize():
    """Test sequential boxed optimization with minimization.

    Sequential optimization processes objectives one by one:
    1. First minimize x independently
    2. Then fix x at its minimum and minimize y independently
    """
    x = z3.BitVec('x', 4)
    y = z3.BitVec('y', 4)

    formula = z3.And(z3.UGT(x, 3), z3.UGT(y, 3), z3.ULE(x + y, 15))
    objectives = [x, y]

    results = solve_boxed_sequential(formula, objectives, minimize=True,
                                     engine="iter", solver_name="z3-bs")

    assert results is not None
    assert len(results) == 2
    # Sequential minimization: first objective minimized independently
    assert results[0] == 4  # Minimum value satisfying x > 3
    # Second objective minimized after fixing first objective
    assert results[1] is not None and results[1] >= 4


if __name__ == "__main__":
    pytest.main([__file__])
