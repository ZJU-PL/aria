# Utility functions for lia_star modules
"""
Utility module for LIA* solver.
"""

from z3 import *
from aria.smt.lia_star import statistics


def getModel(s, X=[]):
    """
    Check satisfiability of solver s and return a model if SAT.

    Args:
        s: Z3 Solver instance.
        X: List of Z3 variables to evaluate.

    Returns:
        If SAT, returns a list of values (as longs) corresponding to X.
        If UNSAT/UNKNOWN, returns None.
    """
    res = s.check()
    statistics.z3_calls += 1
    if res != sat:
        return None
    m = s.model()
    return [m.eval(x).as_long() for x in X]
