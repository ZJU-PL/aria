"""Counting interfaces for pySMT Boolean formulas."""

from typing import Tuple, List
from pysmt.shortcuts import Solver as PySMTSolver, Not, get_free_variables, Or
from pysmt.rewritings import cnf

from aria.counting.bool.dimacs_counting import (
    count_dimacs_solutions,
    count_dimacs_solutions_parallel,
)


def count_pysmt_models_by_enumeration(formula, max_models: int = None) -> int:
    """
    Count models for a pySMT Boolean formula using model enumeration.

    Args:
        formula: The pySMT formula to count models for
        max_models (int, optional): Maximum number of models to count

    Returns:
        int: Number of models found (-1 if exceeded max_models)
    """
    solver = PySMTSolver()
    solver.add_assertion(formula)
    count = 0
    variables = list(get_free_variables(formula))

    while solver.solve():
        count += 1
        if max_models and count > max_models:
            return -1

        model = solver.get_model()
        # Create blocking clause for all variables
        block = []
        for var in variables:
            val = model.get_value(var)
            if val.is_true():
                block.append(Not(var))
            else:
                block.append(var)

        # Add blocking clause to prevent this model from appearing again
        solver.add_assertion(Or(block))

    return count


def pysmt_to_dimacs(formula) -> Tuple[List[str], List[str]]:
    """
    Convert a pySMT formula to DIMACS format.

    Args:
        formula: PySMT formula to convert

    Returns:
        Tuple[List[str], List[str]]: Header and clauses in DIMACS format
    """
    # Convert to CNF
    cnf_formula = cnf(formula)

    # Get variables and create mapping
    all_vars = cnf_formula.get_free_variables()
    var_map = {var: idx + 1 for idx, var in enumerate(sorted(all_vars, key=str))}

    # Convert clauses
    dimacs_clauses = []
    if cnf_formula.is_and():
        clauses = cnf_formula.args()
    else:
        clauses = [cnf_formula]

    for clause in clauses:
        if clause.is_or():
            lits = clause.args()
        else:
            lits = [clause]

        dimacs_clause = []
        for lit in lits:
            if lit.is_not():
                var = lit.arg(0)
                dimacs_clause.append(f"-{var_map[var]}")
            else:
                dimacs_clause.append(str(var_map[lit]))
        dimacs_clauses.append(" ".join(dimacs_clause))

    header = [f"p cnf {len(var_map)} {len(dimacs_clauses)}"]
    return header, dimacs_clauses


def count_pysmt_solutions(formula, parallel: bool = False) -> int:
    """
    Count solutions for a pySMT formula.

    Args:
        formula: PySMT formula to count solutions for
        parallel (bool): Whether to use parallel counting

    Returns:
        int: Number of solutions
    """
    header, clauses = pysmt_to_dimacs(formula)
    if parallel:
        return count_dimacs_solutions_parallel(header, clauses)
    return count_dimacs_solutions(header, clauses)
