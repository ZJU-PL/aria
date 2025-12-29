#!/usr/bin/env python3

import argparse
from aria.smt.lia_star import semilinear
from aria.smt.lia_star import interpolant
import aria.smt.lia_star.statistics
from aria.smt.lia_star import dsl
from aria.smt.lia_star.lia_star_utils import getModel
import time
from z3 import *

# Global flags
verbose = False
instrument = False

# Print if verbose
def printV(s):
    if verbose: print(s)

# Get free arithmetic variables from a formula
def freeArithVars(fml):
    seen = set([])
    vars = set([])
    int_sort = IntSort()

    def fv(seen, vars, f):
        if f in seen:
            return
        seen |= { f }
        if f.sort().eq(int_sort) and f.decl().kind() == Z3_OP_UNINTERPRETED:
            vars |= { f }
        for ch in f.children():
            fv(seen, vars, ch)

    fv(seen, vars, fml)
    return vars

# Turn A and B into macros and get their shared variables
def toMacro(fmls):

    # Pull free variables from the assertion
    vars = freeArithVars(And(fmls))

    def F(X=[]):

        # Default args
        X = X + F.args[len(X):]

        # If args are integers they need to be casted to z3 vars
        X = [Var(x, IntSort()) if isinstance(x, int) else x for x in X]

        # Perform substitution
        subs = list(zip(F.args, X))
        func_list = [substitute(fml, subs) for fml in F.fmls]
        if len(func_list) == 1:
            return func_list[0]
        else:
            return And(func_list)
    F.args = vars
    F.fmls = fmls

    return F


# Print a solution vector and SLS or unsat and exit
def returnSolution(result, sls):

    # Print statistics if this is an instrumented run
    if instrument:
        stats = {
            'sat': 1 if result != unsat else 0,
            'problem_size': aria.smt.lia_star.statistics.problem_size,
            'sls_size': sls.size(),
            'z3_calls': aria.smt.lia_star.statistics.z3_calls,
            'interpolants_generated': aria.smt.lia_star.statistics.interpolants_generated,
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
        exit(0)

    # Print the satisfying assignments, and the SLS if one is provided
    print("sat\n{}".format("\n".join(["{} = {}".format(k, v) for (k, v) in result if k not in sls.set_vars])))
    if sls:
        print("SLS = {}".format(sls.getSLS()))

    # Quit after the solution is printed
    exit(0)

# Check if I => (not A)
def checkUnsatWithInterpolant(inductive_clauses, A):

    # Assert that I, with non-negativity constraints, implies (not A)
    s = Solver()
    s.add(ForAll(A.args, Implies(And([x >= 0 for x in A.args] + inductive_clauses), Not(A()))))

    # Check satisfiability
    return getModel(s) != None

# Return a non-negative vector which satisfies the formula A and SLS*
# If no such vector exists, return None.
def findSolution(A, sls):
    start = time.time()

    # Assert that X satisfies A and is in SLS*
    s = Solver()
    s.add([v >= 0 for v in A.args])
    s.add(A())
    s.add(sls.star())

    # Check satisfiability
    printV("\nLooking for a solution vector with the following constraints:\n\n{}".format(s))
    m = getModel(s, A.args)
    end = time.time()
    aria.smt.lia_star.statistics.solution_time += end - start
    return m

# Iteratively construct a semi-linear set, checking with each new vector if there is
# a solution to the given A within that set. On each iteration, also reduce the SLS
# to generalize and complete it without enumerating every SLS vector, and get
# interpolants to see if unsatisfiability can be shown early.
def main():
    global verbose, instrument

    # Initialize arg parser
    prog_desc = 'Translates a set/multiset problem given by a BAPA benchmark into LIA* and solves it'
    p = argparse.ArgumentParser(description=prog_desc)
    p.add_argument('file', metavar='FILEPATH', type=str,
                   help='smt-lib BAPA file describing a set/multiset problem')
    p.add_argument('-m', '--mapa', action='store_true',
                   help='treat the BAPA benchmark as a MAPA problem (interpret the variables as multisets, not sets)')
    p.add_argument('--no-interp', action='store_true',
                   help='turn off interpolation')
    p.add_argument('-v', '--verbose', action='store_true',
                   help='provide descriptive output while solving')
    p.add_argument('-i', '--instrument', action='store_true',
                   help='run with instrumentation to get statistics back after solving')
    p.add_argument('--unfold', metavar='N', type=int, default=0,
                   help='number of unfoldings to use when interpolating (default: 0)')

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
    A_assertions = [fmls]
    B_assertions = [a == b for (a, b) in star_defs] + star_fmls
    set_vars = [a for (a, b) in star_defs]

    # Record statistics
    aria.smt.lia_star.statistics.problem_size = len(A_assertions) + len(B_assertions)
    if instrument:
        print(aria.smt.lia_star.statistics.problem_size, flush=True)

    # Functionalize the given assertions so they can be called with arbitrary args
    A = toMacro(A_assertions)
    B = toMacro(B_assertions)
    A.args = set_vars + [a for a in A.args if a not in set_vars]
    B.args = set_vars + [b for b in B.args if b not in set_vars]

    # A(0) may be immediately satisfiable
    sls = semilinear.SLS(B, set_vars, len(B.args))
    X = findSolution(A, sls)
    if X: returnSolution(list(zip(A.args, X)), sls)

    # SLS construction loop
    i = interpolant.Interpolant(A, B)
    incomplete = sls.augment()
    while incomplete:

        # If there's a solution using this SLS, return it
        X = findSolution(A, sls)
        if X: returnSolution(list(zip(A.args, X)), sls)

        # Compute any new interpolants for this iteration
        start = time.time()
        if interpolation_on:
            i.update(sls)
            i.add_forward_interpolant(unfold)
            i.add_backward_interpolant(unfold)

            # Extract all inductive clauses
            i.filter_to_inductive()
            inductive_clauses = i.get_inductive()
            printV("\nInductive clauses: {}\n".format(inductive_clauses))

            # Check satisfiability against inductive interpolant
            if checkUnsatWithInterpolant(inductive_clauses, A):
                end = time.time()
                aria.smt.lia_star.statistics.interpolation_time += end - start
                returnSolution(unsat, sls)
        end = time.time()
        aria.smt.lia_star.statistics.interpolation_time += end - start

        # At every iteration, shorten the SLS / its vectors
        sls.reduce()

        # Add another vector to the SLS
        incomplete = sls.augment()
        printV("SLS: {}".format(sls.getSLS()))

    # If the SLS is equivalent to B and a solution was not found, the problem is unsat
    returnSolution(unsat, sls)

# Entry point
if __name__ == "__main__":
    main()
