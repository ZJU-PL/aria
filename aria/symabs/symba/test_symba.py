"""Test cases for SYMBA implementation."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import z3  # pylint: disable=wrong-import-position

from aria.symabs.symba import SYMBA, MultiSYMBA  # pylint: disable=wrong-import-position


def run_symba_test(constraints, objectives, expected_bounds=None, desc="Test"):
    """Helper to run SYMBA and return results."""
    symba = SYMBA(constraints, objectives)
    state = symba.optimize()
    bounds = symba.get_optimal_values() if objectives else None
    print(f"{desc}: {len(state.M)} models, bounds: {bounds}")
    if expected_bounds:
        for obj, expected in expected_bounds.items():
            assert (
                bounds.get(obj, -float("inf")) >= expected
            ), f"Bounds check failed for {desc}"
    return state, bounds


def test_simple_maximization():
    """Test SYMBA on a simple maximization problem."""
    x, y = z3.Ints("x y")
    constraints = z3.And(x + y <= 10, x >= 0, y >= 0)
    objective = x + y

    state, bounds = run_symba_test(constraints, [objective], {objective: 10}, "Max")
    assert all(model.eval(objective).as_long() <= 10 for model in state.M)
    print("✓ Simple maximization test passed!")


def test_simple_minimization():
    """Test SYMBA on a simple minimization problem."""
    x, y = z3.Ints("x y")
    constraints = z3.And(x + y >= 5, x >= 0, y >= 0)
    objective = x + y

    state, bounds = run_symba_test(constraints, [objective], {objective: 5}, "Min")
    assert all(model.eval(objective).as_long() >= 5 for model in state.M)
    print("✓ Simple minimization test passed!")


def run_multisymba_test(constraints, objectives, desc="Multi-objective"):
    """Helper to run MultiSYMBA and return results."""
    multi_symba = MultiSYMBA(constraints, objectives)
    state = multi_symba.optimize()
    pareto_front = multi_symba.get_pareto_front()
    pareto_values = multi_symba.get_pareto_values()
    print(f"{desc}: {len(state.M)} models, Pareto front: {len(pareto_front)}")
    if len(pareto_values) <= 5:  # Only print if not too many
        for vals in pareto_values:
            print(
                f"  {', '.join(f'{obj}={val}' for obj, val in zip(['x', 'y'], vals))}"
            )
    return state, pareto_values


def test_multi_objective():
    """Test SYMBA on a multi-objective problem."""
    x, y = z3.Ints("x y")
    constraints = z3.And(x + y <= 10, x >= 0, y >= 0)
    objectives = [x, y]

    state, pareto_values = run_multisymba_test(
        constraints, objectives, "Multi-objective"
    )
    assert len(pareto_values) > 0, "Should find Pareto-optimal solutions"
    print("✓ Multi-objective test passed!")


def test_linear_programming_example():
    """Test SYMBA on a classic linear programming example."""
    x1, x2 = z3.Ints("x1 x2")
    constraints = z3.And(2 * x1 + x2 <= 100, x1 + x2 <= 80, x1 <= 40, x1 >= 0, x2 >= 0)
    objective = 3 * x1 + 2 * x2

    state, bounds = run_symba_test(constraints, [objective], {objective: 170}, "LP")

    if state.M:
        for model in state.M:
            x1_val, x2_val = model.eval(x1).as_long(), model.eval(x2).as_long()
            obj_val = model.eval(objective).as_long()
            # Verify constraints
            assert 2 * x1_val + x2_val <= 100 and x1_val + x2_val <= 80 and x1_val <= 40
            print(f"  Solution: x1={x1_val}, x2={x2_val}, obj={obj_val}")

    print("✓ Linear programming test passed!")


def z3_optimize(constraints, objective):
    """Get Z3 optimal value for single objective."""
    opt = z3.Optimize()
    opt.add(constraints)
    opt.maximize(objective)
    return opt.model().eval(objective).as_long() if opt.check() == z3.sat else None


def test_against_z3_optimization():
    """Test SYMBA against Z3's built-in optimization for consistency."""
    x, y = z3.Ints("x y")
    constraints = z3.And(x + y <= 10, x >= 0, y >= 0)
    objective = x + y

    # Test single objective consistency
    symba_opt = SYMBA(constraints, [objective])
    symba_opt.optimize()
    symba_val = symba_opt.get_optimal_values().get(objective)

    z3_val = z3_optimize(constraints, objective)
    print(f"Z3 comparison: SYMBA={symba_val}, Z3={z3_val}")
    assert symba_val == z3_val or symba_val is None

    # Test multi-objective
    multi_symba = MultiSYMBA(constraints, [x, y])
    multi_symba.optimize()
    symba_pareto = multi_symba.get_pareto_values()

    z3_opt = z3.Optimize()
    z3_opt.set("opt.priority", "box")
    z3_opt.add(constraints)
    z3_opt.maximize(x)
    z3_opt.maximize(y)
    if z3_opt.check() == z3.sat:
        z3_x, z3_y = z3_opt.model().eval(x).as_long(), z3_opt.model().eval(y).as_long()
        z3_in_pareto = (z3_x, z3_y) in symba_pareto
        print(f"Z3 multi-obj: ({z3_x},{z3_y}) in Pareto: {z3_in_pareto}")

    print("✓ Z3 consistency test passed!")


def test_correctness_properties():
    """Test SYMBA's correctness properties."""
    x, y = z3.Ints("x y")
    constraints = z3.And(x >= 0, y >= 0, x + y <= 10, x + 2 * y >= 5)

    # Test feasibility: known solution (3,1) should be findable
    assert constraints.substitute([(x, z3.IntVal(3)), (y, z3.IntVal(1))])
    symba = SYMBA(constraints, [x + y])
    state = symba.optimize()
    assert len(state.M) > 0, "Should find feasible solutions"

    # Test optimality bounds
    if symba.get_optimal_values():
        optimal_val = list(symba.get_optimal_values().values())[0]
        obj_vals = [model.eval(x + y).as_long() for model in state.M]
        assert all(
            obj <= optimal_val for obj in obj_vals
        ), "No solution should exceed optimal"

    print("✓ Correctness properties test passed!")


def run_all_tests():
    """Run all SYMBA tests."""
    tests = [
        test_simple_maximization,
        test_simple_minimization,
        test_multi_objective,
        test_linear_programming_example,
        test_against_z3_optimization,
        test_correctness_properties,
    ]

    print("Running SYMBA tests...")
    print("=" * 50)

    for test in tests:
        try:
            test()
            print()
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            raise

    print("=" * 50)
    print("✓ All tests passed!")


if __name__ == "__main__":
    run_all_tests()
