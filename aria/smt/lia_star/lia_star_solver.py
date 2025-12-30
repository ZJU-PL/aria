#!/usr/bin/env python3

import argparse
import sys
import time

from z3 import (And, ForAll, Implies, IntSort, Not, Solver, Var, Z3_OP_UNINTERPRETED,
                substitute, unsat)

from aria.smt.lia_star import dsl, interpolant, semilinear
import aria.smt.lia_star.statistics
from aria.smt.lia_star.lia_star_utils import getModel

# Global flags
verbose = False  # pylint: disable=invalid-name
instrument = False  # pylint: disable=invalid-name

# Print if verbose
def print_verbose(msg):  # pylint: disable=invalid-name
    """Print message if verbose mode is enabled."""
    if verbose:
        print(msg)

# Get free arithmetic variables from a formula
def free_arith_vars(fml):  # pylint: disable=invalid-name
    """Extract free arithmetic variables from a formula."""
    seen = set()
    var_set = set()
    int_sort = IntSort()

    def fv(seen_set, var_set_inner, expr):
        if expr in seen_set:
            return
        seen_set.add(expr)
        if expr.sort().eq(int_sort) and expr.decl().kind() == Z3_OP_UNINTERPRETED:
            var_set_inner.add(expr)
        for child in expr.children():
            fv(seen_set, var_set_inner, child)

    fv(seen, var_set, fml)
    return var_set

# Turn A and B into macros and get their shared variables
def to_macro(fmls):  # pylint: disable=invalid-name
    """Convert formulas into a macro function."""
    # Pull free variables from the assertion
    var_set = free_arith_vars(And(fmls))

    def macro_func(x_vars=None):  # pylint: disable=invalid-name
        """Macro function with default arguments."""
        if x_vars is None:
            x_vars = []
        # Default args
        x_vars = x_vars + macro_func.args[len(x_vars):]

        # If args are integers they need to be casted to z3 vars
        x_vars = [Var(x, IntSort()) if isinstance(x, int) else x for x in x_vars]

        # Perform substitution
        subs = list(zip(macro_func.args, x_vars))
        func_list = [substitute(fml, subs) for fml in macro_func.fmls]
        if len(func_list) == 1:
            return func_list[0]
        return And(func_list)
    macro_func.args = var_set
    macro_func.fmls = fmls

    return macro_func


# Print a solution vector and SLS or unsat and exit
def return_solution(result, sls):  # pylint: disable=invalid-name
    """Print solution or unsat and exit."""
    # Print statistics if this is an instrumented run
    if instrument:
        stats = {
            'sat': 1 if result != unsat else 0,
            'problem_size': aria.smt.lia_star.statistics.problem_size,
            'sls_size': sls.size(),
            'z3_calls': aria.smt.lia_star.statistics.z3_calls,
            'interpolants_generated': aria.smt.lia_star.statistics.interpolants_generated,  # noqa: E501
            'merges': aria.smt.lia_star.statistics.merges,
            'shiftdowns': aria.smt.lia_star.statistics.shiftdowns,
            'offsets': aria.smt.lia_star.statistics.offsets,
            'reduction_time': aria.smt.lia_star.statistics.reduction_time,
            'augment_time': aria.smt.lia_star.statistics.augment_time,
            'interpolation_time': aria.smt.lia_star.statistics.interpolation_time,
            'solution_time': aria.smt.lia_star.statistics.solution_time
        }
        print(stats)

    # Print unsat if result is unsat
    if result == unsat:
        print(result)
        sys.exit(0)

    # Print the satisfying assignments, and the SLS if one is provided
    assignments = [
        f"{k} = {v}" for (k, v) in result if k not in sls.set_vars
    ]
    print(f"sat\n{'\n'.join(assignments)}")
    if sls:
        print(f"SLS = {sls.get_sls()}")

    # Quit after the solution is printed
    sys.exit(0)

# Check if I => (not A)
def check_unsat_with_interpolant(inductive_clauses, a_func):  # pylint: disable=invalid-name
    """Check if inductive clauses imply not A."""
    # Assert that I, with non-negativity constraints, implies (not A)
    s = Solver()
    constraints = [x >= 0 for x in a_func.args] + inductive_clauses
    s.add(ForAll(
        a_func.args,
        Implies(And(constraints), Not(a_func()))
    ))

    # Check satisfiability
    return getModel(s) is not None

# Return a non-negative vector which satisfies the formula A and SLS*
# If no such vector exists, return None.
def find_solution(a_func, sls):  # pylint: disable=invalid-name
    """Find a solution vector satisfying A and SLS*."""
    start = time.time()

    # Assert that X satisfies A and is in SLS*
    s = Solver()
    s.add([v >= 0 for v in a_func.args])
    s.add(a_func())
    s.add(sls.star())

    # Check satisfiability
    print_verbose(f"\nLooking for a solution vector with the following constraints:\n\n{s}")
    m = getModel(s, a_func.args)
    end = time.time()
    aria.smt.lia_star.statistics.solution_time += end - start
    return m

# Iteratively construct a semi-linear set, checking with each new vector if there is
# a solution to the given A within that set. On each iteration, also reduce the SLS
# to generalize and complete it without enumerating every SLS vector, and get
# interpolants to see if unsatisfiability can be shown early.
def main():
    """Main entry point for the LIA* solver."""
    global verbose, instrument  # pylint: disable=global-statement

    # Initialize arg parser
    prog_desc = (
        'Translates a set/multiset problem given by a BAPA benchmark '
        'into LIA* and solves it'
    )
    p = argparse.ArgumentParser(description=prog_desc)
    p.add_argument('file', metavar='FILEPATH', type=str,
                   help='smt-lib BAPA file describing a set/multiset problem')
    p.add_argument('-m', '--mapa', action='store_true',
                   help='treat the BAPA benchmark as a MAPA problem '
                         '(interpret the variables as multisets, not sets)')
    p.add_argument('--no-interp', action='store_true',
                   help='turn off interpolation')
    p.add_argument('-v', '--verbose', action='store_true',
                   help='provide descriptive output while solving')
    p.add_argument('-i', '--instrument', action='store_true',
                   help='run with instrumentation to get statistics '
                         'back after solving')
    p.add_argument('--unfold', metavar='N', type=int, default=0,
                   help='number of unfoldings to use when interpolating '
                         '(default: 0)')

    # Read args
    args = p.parse_args()
    bapa_file = args.file
    mapa = args.mapa
    verbose = args.verbose
    instrument = args.instrument
    unfold = args.unfold
    interpolation_on = not args.no_interp

    # Get assertions for A and B from bapa file
    multiset_fmls = dsl.parse_bapa(bapa_file, mapa)
    fmls, star_defs, star_fmls = dsl.to_lia_star(And(multiset_fmls))
    a_assertions = [fmls]  # pylint: disable=invalid-name
    b_assertions = [a == b for (a, b) in star_defs] + star_fmls  # pylint: disable=invalid-name
    set_vars = [a for (a, b) in star_defs]

    # Record statistics
    aria.smt.lia_star.statistics.problem_size = len(a_assertions) + len(b_assertions)
    if instrument:
        print(aria.smt.lia_star.statistics.problem_size, flush=True)

    # Functionalize the given assertions so they can be called with arbitrary args
    a_func = to_macro(a_assertions)  # pylint: disable=invalid-name
    b_func = to_macro(b_assertions)  # pylint: disable=invalid-name
    a_func.args = set_vars + [a for a in a_func.args if a not in set_vars]
    b_func.args = set_vars + [b for b in b_func.args if b not in set_vars]

    # A(0) may be immediately satisfiable
    sls = semilinear.SLS(b_func, set_vars, len(b_func.args))
    x_sol = find_solution(a_func, sls)  # pylint: disable=invalid-name
    if x_sol:
        return_solution(list(zip(a_func.args, x_sol)), sls)

    # SLS construction loop
    interp = interpolant.Interpolant(a_func, b_func)
    incomplete = sls.augment()
    while incomplete:

        # If there's a solution using this SLS, return it
        x_sol = find_solution(a_func, sls)  # pylint: disable=invalid-name
        if x_sol:
            return_solution(list(zip(a_func.args, x_sol)), sls)

        # Compute any new interpolants for this iteration
        start = time.time()
        if interpolation_on:
            interp.update(sls)
            interp.add_forward_interpolant(unfold)
            interp.add_backward_interpolant(unfold)

            # Extract all inductive clauses
            interp.filter_to_inductive()
            inductive_clauses = interp.get_inductive()
            print_verbose(f"\nInductive clauses: {inductive_clauses}\n")

            # Check satisfiability against inductive interpolant
            if check_unsat_with_interpolant(inductive_clauses, a_func):
                end = time.time()
                aria.smt.lia_star.statistics.interpolation_time += end - start
                return_solution(unsat, sls)
        end = time.time()
        aria.smt.lia_star.statistics.interpolation_time += end - start

        # At every iteration, shorten the SLS / its vectors
        sls.reduce()

        # Add another vector to the SLS
        incomplete = sls.augment()
        print_verbose(f"SLS: {sls.getSLS()}")

    # If the SLS is equivalent to B and a solution was not found, the problem is unsat
    return_solution(unsat, sls)

# Entry point
if __name__ == "__main__":
    main()
