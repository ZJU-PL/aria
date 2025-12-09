"""
Boolean Formula Model Enumeration and Counting

This module provides functions for enumerating and counting models (satisfying assignments)
of Boolean formulas using different approaches. It includes benchmarking capabilities to
compare the performance of different enumeration strategies.
"""
import itertools
import time
from typing import List, Callable, Tuple, Optional, Set
from z3 import *


def benchmark(name: str, function: Callable[[Solver, List[BoolRef], bool], int],
             solver: Solver, variables: List[BoolRef],
              verbose: bool = True) -> Tuple[int, float]:
    """
    Benchmark a model counting function and return results.

    Args:
        name: Name of the approach being benchmarked
        function: The counting function to benchmark
        solver: Z3 solver containing the formula
        variables: List of Boolean variables in the formula
        verbose: Whether to print results (default: True)

    Returns:
        Tuple of (number of models, execution time in seconds)
    """
    if verbose:
        print(f'--{name} approach--')

    start_time = time.perf_counter()
    model_count = function(solver, variables, verbose)
    end_time = time.perf_counter()
    execution_time = round(end_time - start_time, 2)

    if verbose:
        print(f'Number of models: {model_count}')
        print(f'Time: {execution_time} s')

    return model_count, execution_time


def count_models_with_solver(solver: Solver, variables: List[BoolRef],
                             show_progress: bool = False) -> int:
    """
    Count the number of solutions of a formula using solver-based enumeration.

    This approach uses the solver to find all models by iteratively adding
    constraints that block previously found models.

    Args:
        solver: Z3 solver containing the formula
        variables: List of Boolean variables in the formula
        show_progress: Whether to show progress during enumeration

    Returns:
        Number of satisfying assignments (models)
    """
    solver.push()  # checkpoint current state as we'll add further assertions
    solutions = 0

    while solver.check() == sat:
        solutions += 1
        if show_progress and solutions % 1000 == 0:
            print(f"Found {solutions} models so far...")

        # Get current model
        model = solver.model()

        # Invert at least one variable to get a different solution:
        # For each variable, add constraint to make it different from current assignment
        solver.add(Or([Not(x) if is_true(model[x]) else x for x in variables]))

    solver.pop()  # restore solver to previous state
    return solutions


def count_models_by_enumeration(solver: Solver, variables: List[BoolRef],
                                show_progress: bool = False) -> int:
    """
    Count the number of solutions by enumerating all possible assignments.

    This is the fastest enumeration approach that uses conditional checking
    with direct assignment.

    Args:
        solver: Z3 solver containing the formula
        variables: List of Boolean variables in the formula
        show_progress: Whether to show progress during enumeration

    Returns:
        Number of satisfying assignments (models)
    """
    solutions = 0
    total_assignments = 2 ** len(variables)

    # Generate all possible combinations of variable assignments
    for i, assignment in enumerate(itertools.product(*[(x, Not(x)) for x in variables])):
        if solver.check(assignment) == sat:  # conditional check (does not add assignment permanently)
            solutions += 1

        if show_progress and (i + 1) % 10000 == 0:
            print(f"Checked {i + 1}/{total_assignments} assignments, found {solutions} models so far...")

    return solutions


def count_models_by_enumeration2(solver: Solver, variables: List[BoolRef],
                                 show_progress: bool = False) -> int:
    """
    Count the number of solutions by enumerating all possible assignments.

    This approach creates the assignment as a separate step, which is slightly slower.

    Args:
        solver: Z3 solver containing the formula
        variables: List of Boolean variables in the formula
        show_progress: Whether to show progress during enumeration

    Returns:
        Number of satisfying assignments (models)
    """
    solutions = 0
    total_assignments = 2 ** len(variables)

    # Generate all possible combinations of True/False values
    for i, assignment in enumerate(itertools.product([False, True], repeat=len(variables))):
        # Create Z3 constraints based on the assignment
        constraints = [x if assign_true else Not(x) for x, assign_true in zip(variables, assignment)]

        if solver.check(constraints) == sat:
            solutions += 1

        if show_progress and (i + 1) % 10000 == 0:
            print(f"Checked {i + 1}/{total_assignments} assignments, found {solutions} models so far...")

    return solutions


def count_models_by_enumeration3(solver: Solver, variables: List[BoolRef],
                                 show_progress: bool = False) -> int:
    """
    Count the number of solutions by enumerating all possible assignments.

    This approach uses simplification instead of conditional checking, which is the slowest.

    Args:
        solver: Z3 solver containing the formula
        variables: List of Boolean variables in the formula
        show_progress: Whether to show progress during enumeration

    Returns:
        Number of satisfying assignments (models)
    """
    solutions = 0
    total_assignments = 2 ** len(variables)

    # Generate all possible combinations of True/False values
    for i, assignment in enumerate(itertools.product([BoolVal(False), BoolVal(True)], repeat=len(variables))):
        satisfied = True

        # Check if the assignment satisfies all assertions in the solver
        for assertion in solver.assertions():
            if is_false(simplify(substitute(assertion, list(zip(variables, assignment))))):
                satisfied = False
                break

        if satisfied:
            solutions += 1

        if show_progress and (i + 1) % 10000 == 0:
            print(f"Checked {i + 1}/{total_assignments} assignments, found {solutions} models so far...")

    return solutions


def run_benchmarks(formula_name: str, formula: BoolRef, variables: List[BoolRef],
                   max_vars: int = 20) -> None:
    """
    Run benchmarks for a given formula with different enumeration approaches.

    Args:
        formula_name: Name of the formula for display
        formula: Z3 formula to benchmark
        variables: List of Boolean variables in the formula
        max_vars: Maximum number of variables to use (to prevent excessive runtime)
    """
    if len(variables) > max_vars:
        print(f"Warning: Formula has {len(variables)} variables, which may cause excessive runtime.")
        print(f"Using only the first {max_vars} variables for benchmarking.")
        variables = variables[:max_vars]

    solver = Solver()
    solver.add(formula)

    print(f'\n## {formula_name} formula ##')
    print(f'Variables: {len(variables)}')
    print(f'Formula: {formula}')

    # Run benchmarks for each approach
    benchmark('Solver-based', count_models_with_solver, solver, variables)
    benchmark('Enumeration-based (conditional check, direct assignment)',
              count_models_by_enumeration, solver, variables)
    benchmark('Enumeration-based (conditional check, separate assignment)',
              count_models_by_enumeration2, solver, variables)
    benchmark('Enumeration-based (substitute + simplify)',
              count_models_by_enumeration3, solver, variables)


def test_enu() -> None:
    """Run benchmarks on sample formulas to compare different enumeration approaches."""
    # Create 10 Boolean variables
    x = Bools(' '.join('x' + str(i) for i in range(10)))

    # Test OR formula
    run_benchmarks("OR", Or(x), x)

    # Test AND formula
    run_benchmarks("AND", And(x), x)

    # Test a more complex formula
    complex_formula = And(Or(x[0], x[1], x[2]), Or(Not(x[0]), x[3], x[4]), Or(x[5], Not(x[6])))
    run_benchmarks("Complex", complex_formula, x[:7])


def count_models(formula: BoolRef, variables: Optional[List[BoolRef]] = None,
                 method: str = 'solver', show_progress: bool = False) -> int:
    """
    Count the number of models (satisfying assignments) for a given formula.

    This is the main function intended for external use.

    Args:
        formula: Z3 formula to count models for
        variables: List of Boolean variables in the formula (if None, will be extracted from formula)
        method: Counting method to use ('solver', 'enum1', 'enum2', or 'enum3')
        show_progress: Whether to show progress during enumeration

    Returns:
        Number of satisfying assignments (models)
    """
    # Extract variables from formula if not provided
    if variables is None:
        variables = get_vars(formula)

    # Create solver and add formula
    solver = Solver()
    solver.add(formula)

    # Select counting method
    if method == 'solver':
        return count_models_with_solver(solver, variables, show_progress)
    elif method == 'enum1':
        return count_models_by_enumeration(solver, variables, show_progress)
    elif method == 'enum2':
        return count_models_by_enumeration2(solver, variables, show_progress)
    elif method == 'enum3':
        return count_models_by_enumeration3(solver, variables, show_progress)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'solver', 'enum1', 'enum2', or 'enum3'.")


def get_vars(formula: BoolRef) -> List[BoolRef]:
    """
    Extract all Boolean variables from a formula.

    Args:
        formula: Z3 formula to extract variables from

    Returns:
        List of Boolean variables in the formula
    """
    vars_set: Set[BoolRef] = set()

    def collect_vars(expr: BoolRef) -> None:
        if is_const(expr) and expr.decl().kind() == Z3_OP_UNINTERPRETED and expr.sort() == BoolSort():
            vars_set.add(expr)
        for child in expr.children():
            collect_vars(child)

    collect_vars(formula)
    return list(vars_set)


if __name__ == '__main__':
    test_enu()
