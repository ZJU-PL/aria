# coding: utf-8
"""
Test module for counting bitvector models by enumeration.
"""
import itertools
import time

import z3

from aria.utils.z3_expr_utils import get_variables


def check_candidate_model(formula, all_vars, candidate):
    """Check if a candidate assignment satisfies the formula."""
    solver = z3.Solver()
    solver.add(formula)

    # Build assumptions for the candidate assignment
    assumptions = []
    for i, var in enumerate(all_vars):
        assumptions.append(var == z3.BitVecVal(candidate[i], var.sort().size()))

    return solver.check(assumptions) == z3.sat


def count_bv_models(formula):
    """Count all satisfying assignments for a bitvector formula by enumeration.

    Args:
        formula: A Z3 bitvector formula

    Returns:
        Number of satisfying assignments
    """
    all_vars = get_variables(formula)
    solution_count_var = 0

    # Generate all possible assignments for each variable
    ranges = [range(2 ** var.sort().size()) for var in all_vars]

    # Check each assignment
    for assignment in itertools.product(*ranges):
        if check_candidate_model(formula, all_vars, assignment):
            solution_count_var += 1

    return solution_count_var


if __name__ == "__main__":
    # Example: count solutions for (x > 2) AND (y > 1) with 4-bit bitvectors
    x = z3.BitVec("x", 4)
    y = z3.BitVec("y", 4)
    test_formula = z3.And(z3.UGT(x, 2), z3.UGT(y, 1))

    time_start = time.process_time()
    SOLUTION_COUNT = count_bv_models(test_formula)
    elapsed = time.process_time() - time_start

    print(f"Time: {elapsed:.4f}s")
    print(f"Total solutions: {SOLUTION_COUNT}")
