"""Counting interfaces for Z3py Boolean formulas."""

from typing import List, Tuple
import z3
from aria.utils.z3_expr_utils import get_variables

from aria.counting.bool.dimacs_counting import (
    count_dimacs_solutions,
    count_dimacs_solutions_parallel,
)


def count_z3_models_by_enumeration(formula) -> int:
    """
    Count models by enumerating all solutions using Z3's model enumeration

    Args:
        formula: Z3 formula to count models for
    Returns:
        Number of satisfying models
    """
    solver = z3.Solver()
    solver.add(formula)
    count = 0

    # Get all variables in the formula
    variables = get_variables(formula)

    while solver.check() == z3.sat:
        count += 1
        model = solver.model()

        # Create blocking clause from current model
        block = []
        for var in variables:
            val = model.eval(var, model_completion=True)
            block.append(var != val)

        solver.add(z3.Or(block))

    return count


def z3_to_dimacs(formula: z3.BoolRef) -> Tuple[List[str], List[str]]:
    """
    Convert a z3 formula to DIMACS format.

    Args:
        formula (z3.BoolRef): Z3 formula to convert

    Returns:
        Tuple[List[str], List[str]]: Header and clauses in DIMACS format
    """
    # Convert to CNF
    goal = z3.Goal()
    goal.add(formula)
    tactic = z3.Then(
        z3.Tactic("simplify"), z3.Tactic("tseitin-cnf"), z3.Tactic("simplify")
    )
    result = tactic(goal)[0]

    # Handle tautology
    if len(result) == 0:
        # Empty CNF is a tautology - add a dummy variable with (x or not x)
        header = ["p cnf 1 2"]
        return header, ["1", "-1"]

    # Get variables and create mapping
    all_vars = set()
    for f in result:
        for v in get_variables(f):
            all_vars.add(str(v))

        # Ensure at least one variable exists
    if not all_vars:
        header = ["p cnf 1 1"]
        return header, ["1"]  # Add a single true variable

    var_map = {name: idx + 1 for idx, name in enumerate(sorted(all_vars))}

    # Convert clauses
    dimacs_clauses = []
    for clause in result:
        if z3.is_or(clause):
            lits = clause.children()
        else:
            lits = [clause]

        dimacs_clause = []
        for lit in lits:
            if z3.is_not(lit):
                var_name = str(lit.children()[0])
                dimacs_clause.append(f"-{var_map[var_name]}")
            else:
                var_name = str(lit)
                dimacs_clause.append(str(var_map[var_name]))
        dimacs_clauses.append(" ".join(dimacs_clause))

    header = [f"p cnf {len(var_map)} {len(dimacs_clauses)}"]
    return header, dimacs_clauses


def count_z3_solutions(formula: z3.BoolRef, parallel: bool = False) -> int:
    """
    Count solutions for a z3 formula.

    Args:
        formula (z3.BoolRef): Z3 formula to count solutions for
        parallel (bool): Whether to use parallel counting

    Returns:
        int: Number of solutions
    """
    header, clauses = z3_to_dimacs(formula)
    if parallel:
        return count_dimacs_solutions_parallel(header, clauses)
    return count_dimacs_solutions(header, clauses)
