# coding: utf-8

import z3
import itertools
import time
import math

from aria.utils.z3_expr_utils import get_variables


def check_candidate_model(formula, all_vars, candidate):
    """Check if a candidate assignment satisfies the formula."""
    solver = z3.Solver()
    solver.add(formula)

    # Build assumptions for the candidate assignment
    assumptions = []
    for i in range(len(all_vars)):
        assumptions.append(all_vars[i] == z3.BitVecVal(candidate[i], all_vars[i].sort().size()))

    return solver.check(assumptions) == z3.sat


def count_bv_models(formula):
    """Count all satisfying assignments for a bitvector formula by enumeration.

    Args:
        formula: A Z3 bitvector formula

    Returns:
        Number of satisfying assignments
    """
    all_vars = get_variables(formula)
    solutions = 0

    # Generate all possible assignments for each variable
    ranges = [range(2 ** var.sort().size()) for var in all_vars]

    # Check each assignment
    for assignment in itertools.product(*ranges):
        if check_candidate_model(formula, all_vars, assignment):
            solutions += 1

    return solutions


if __name__ == "__main__":
    # Example: count solutions for (x > 2) AND (y > 1) with 4-bit bitvectors
    x = z3.BitVec("x", 4)
    y = z3.BitVec("y", 4)
    formula = z3.And(z3.UGT(x, 2), z3.UGT(y, 1))

    time_start = time.process_time()
    solutions = count_bv_models(formula)
    elapsed = time.process_time() - time_start

    print(f"Time: {elapsed:.4f}s")
    print(f"Total solutions: {solutions}")
