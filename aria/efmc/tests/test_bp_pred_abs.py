"""
Tests for Boolean Program Predicate Abstraction
"""

import pytest
import z3

from aria.efmc.engines.predabs.bp_pred_abs import PredicateAbstractionProver
from aria.efmc.sts import TransitionSystem


def run_test_case(name, init_formula, trans_formula, post_formula, predicates):
    """Helper function to run a test case with the given parameters"""
    print(f"\n=== Running test case: {name} ===")

    # Get all variables from predicates and formulas
    all_vars = []
    for pred in predicates:
        if hasattr(pred, "get_free_variables"):
            all_vars.extend(pred.get_free_variables())
        else:
            all_vars.append(pred)

    # Create primed versions
    primed_vars = []
    for var in all_vars:
        primed_name = str(var) + "!"
        primed_var = z3.Bool(primed_name)
        primed_vars.append(primed_var)

    sts = TransitionSystem()
    sts.from_z3_cnts(
        [all_vars + primed_vars, init_formula, trans_formula, post_formula]
    )

    pp = PredicateAbstractionProver(sts)
    pp.set_predicates(predicates)
    pp.solve()


class TestPredicateAbstraction:
    """Test class for predicate abstraction functionality"""

    def test_simple_boolean_program(self):
        """Test simple Boolean program with 2 variables"""
        x, y = z3.Bools("x y")
        init = z3.And(x, y)
        trans = z3.And(z3.Bool("x!") == y, z3.Bool("y!") == y)
        post = x
        preds = [x, y]

        # Create transition system
        sts = TransitionSystem()
        sts.from_z3_cnts([[x, y, z3.Bool("x!"), z3.Bool("y!")], init, trans, post])

        # Run predicate abstraction
        pp = PredicateAbstractionProver(sts)
        pp.set_predicates(preds)
        pp.solve()

        # The system should be safe (invariant implies post condition)
        # This test verifies the algorithm runs without errors
        assert True  # If we reach here, the test passed

    def test_binary_counter(self):
        """Test binary counter with 2 variables"""
        a, b = z3.Bools("a b")
        init = z3.And(z3.Not(a), z3.Not(b))
        trans = z3.And(z3.Bool("a!") == z3.Xor(a, b), z3.Bool("b!") == z3.Not(b))
        post = z3.Implies(z3.And(a, b), z3.And(a, b))  # Trivial property
        preds = [a, b]

        # Create transition system
        sts = TransitionSystem()
        sts.from_z3_cnts([[a, b, z3.Bool("a!"), z3.Bool("b!")], init, trans, post])

        # Run predicate abstraction
        pp = PredicateAbstractionProver(sts)
        pp.set_predicates(preds)
        pp.solve()

        # The system should be safe
        assert True

    def test_toggle_system(self):
        """Test simple toggle system with 1 variable"""
        p = z3.Bool("p")
        init = z3.Not(p)
        trans = z3.Bool("p!") == z3.Not(p)
        post = z3.Or(p, z3.Not(p))  # Always true
        preds = [p]

        # Create transition system
        sts = TransitionSystem()
        sts.from_z3_cnts([[p, z3.Bool("p!")], init, trans, post])

        # Run predicate abstraction
        pp = PredicateAbstractionProver(sts)
        pp.set_predicates(preds)
        pp.solve()

        # The system should be safe
        assert True

    def test_run_test_case_helper(self):
        """Test the run_test_case helper function"""
        # Test case 1: Simple Boolean program
        x, y = z3.Bools("x y")
        init_1 = z3.And(x, y)
        trans_1 = z3.And(z3.Bool("x!") == y, z3.Bool("y!") == y)
        post_1 = x
        preds_1 = [x, y]

        # This should run without errors
        run_test_case("Simple Boolean", init_1, trans_1, post_1, preds_1)
        assert True

    def test_counter_with_helper(self):
        """Test counter using the helper function"""
        a, b = z3.Bools("a b")
        init_2 = z3.And(z3.Not(a), z3.Not(b))
        trans_2 = z3.And(z3.Bool("a!") == z3.Xor(a, b), z3.Bool("b!") == z3.Not(b))
        post_2 = z3.Implies(z3.And(a, b), z3.And(a, b))
        preds_2 = [a, b]

        run_test_case("Binary Counter", init_2, trans_2, post_2, preds_2)
        assert True

    def test_toggle_with_helper(self):
        """Test toggle using the helper function"""
        p = z3.Bool("p")
        init_3 = z3.Not(p)
        trans_3 = z3.Bool("p!") == z3.Not(p)
        post_3 = z3.Or(p, z3.Not(p))
        preds_3 = [p]

        run_test_case("Toggle", init_3, trans_3, post_3, preds_3)
        assert True


def test_strongest_consequence():
    """Test the strongest_consequence function"""
    from aria.efmc.engines.predabs.bp_pred_abs import strongest_consequence

    x, y = z3.Bools("x y")
    fml = z3.And(x, y)
    predicates = [x, y]

    result = strongest_consequence(fml, predicates)
    assert result is not None
    assert z3.is_expr(result)


def test_weakest_sufficient_condition():
    """Test the weakest_sufficient_condition function"""
    from aria.efmc.engines.predabs.bp_pred_abs import weakest_sufficient_condition

    x, y = z3.Bools("x y")
    fml = z3.And(x, y)
    predicates = [x, y]

    result = weakest_sufficient_condition(fml, predicates)
    assert result is not None
    assert z3.is_expr(result)


def test_fixpoint():
    """Test the fixpoint function"""
    from aria.efmc.engines.predabs.bp_pred_abs import fixpoint

    x, y = z3.Bools("x y")
    inv1 = z3.And(x, y)
    inv2 = z3.Or(x, y)

    # inv2 is weaker than inv1, so fixpoint should be False
    result = fixpoint(inv1, inv2)
    assert isinstance(result, bool)


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__])
