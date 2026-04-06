"""Counting interfaces for Z3py Boolean formulas."""

from timeit import default_timer as counting_timer
from typing import List, Optional, Sequence, Tuple

import z3

from aria.counting.core import CountResult, exact_count_result, unsupported_count_result
from aria.counting.bool.dimacs_counting import call_approxmc
from aria.utils.z3.expr import get_variables

def count_z3_models_by_enumeration(
    formula: z3.BoolRef, variables: Optional[Sequence[z3.ExprRef]] = None
) -> int:
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
    vars_to_count = list(variables) if variables is not None else get_variables(formula)

    if len(vars_to_count) == 0:
        return 1 if solver.check() == z3.sat else 0

    while solver.check() == z3.sat:
        count += 1
        model = solver.model()

        # Create blocking clause from current model
        block = []
        for var in vars_to_count:
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
    simplified = z3.simplify(formula)
    if z3.is_true(simplified):
        return ["p cnf 0 0"], []
    if z3.is_false(simplified):
        return ["p cnf 0 1"], [""]

    # Convert to CNF
    goal = z3.Goal()
    goal.add(formula)
    tactic = z3.Then(
        z3.Tactic("simplify"), z3.Tactic("tseitin-cnf"), z3.Tactic("simplify")
    )
    result = tactic(goal)[0]

    # Get variables and create mapping
    all_vars = set()
    for f in result:
        for v in get_variables(f):
            all_vars.add(str(v))

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


def count_z3_result(
    formula: z3.BoolRef,
    parallel: bool = False,
    variables: Optional[Sequence[z3.ExprRef]] = None,
    method: str = "auto",
) -> CountResult:
    """Count satisfying assignments for a Z3 Boolean formula."""

    time_start = counting_timer()
    simplified = z3.simplify(formula)
    free_variables = get_variables(formula)
    vars_to_count = list(variables) if variables is not None else free_variables
    projection = [str(var) for var in vars_to_count]
    num_variables = len(free_variables)

    free_var_ids = {var.get_id() for var in free_variables}
    for var in vars_to_count:
        if var.get_id() not in free_var_ids:
            return unsupported_count_result(
                backend="z3-enumeration",
                reason=f"projection variable not found in formula: {var}",
                runtime_s=counting_timer() - time_start,
                projection=projection,
                num_variables=num_variables,
            )

    if z3.is_true(simplified):
        return exact_count_result(
            float(2 ** len(vars_to_count)),
            backend="z3-enumeration",
            runtime_s=counting_timer() - time_start,
            projection=projection,
            num_variables=num_variables,
            simplification="tautology",
        )

    if z3.is_false(simplified):
        return exact_count_result(
            0.0,
            backend="z3-enumeration",
            runtime_s=counting_timer() - time_start,
            projection=projection,
            num_variables=num_variables,
            simplification="contradiction",
        )

    if method in ("approx", "approxmc"):
        if variables is not None and len(vars_to_count) != len(free_variables):
            return unsupported_count_result(
                backend="approxmc",
                reason="approximate projected Boolean counting is not supported",
                runtime_s=counting_timer() - time_start,
                projection=projection,
                num_variables=num_variables,
            )
        header, clauses = z3_to_dimacs(formula)
        approx_count = call_approxmc(clauses)
        if approx_count < 0:
            return unsupported_count_result(
                backend="approxmc",
                reason="ApproxMC backend is unavailable or failed",
                runtime_s=counting_timer() - time_start,
                projection=projection,
                num_variables=num_variables,
            )
        return CountResult(
            status="approximate",
            count=float(approx_count),
            backend="approxmc",
            exact=False,
            runtime_s=counting_timer() - time_start,
            projection=projection,
            metadata={
                "num_variables": num_variables,
                "parallel_requested": parallel,
            },
        )

    if method not in ("auto", "exact", "enumeration"):
        return unsupported_count_result(
            backend="z3-none",
            reason=f"unknown Boolean counting method: {method}",
            runtime_s=counting_timer() - time_start,
            projection=projection,
            num_variables=num_variables,
        )

    if method == "auto" and variables is None and num_variables > 16:
        approx_result = count_z3_result(
            formula,
            parallel=parallel,
            variables=variables,
            method="approx",
        )
        if approx_result.status != "unsupported":
            approx_result.metadata["selection"] = "auto"
            return approx_result

    count = count_z3_models_by_enumeration(formula, variables=vars_to_count)
    return exact_count_result(
        float(count),
        backend="z3-enumeration",
        runtime_s=counting_timer() - time_start,
        projection=projection,
        num_variables=num_variables,
        parallel_requested=parallel,
    )


def count_z3_solutions(
    formula: z3.BoolRef,
    parallel: bool = False,
    variables: Optional[Sequence[z3.ExprRef]] = None,
    method: str = "auto",
) -> int:
    """
    Count solutions for a z3 formula.

    Args:
        formula (z3.BoolRef): Z3 formula to count solutions for
        parallel (bool): Whether to use parallel counting

    Returns:
        int: Number of solutions
    """
    _ = parallel
    result = count_z3_result(
        formula, parallel=parallel, variables=variables, method=method
    )
    if result.count is None:
        raise ValueError(f"Z3 model counting failed: {result.status}: {result.reason}")
    return int(result.count)
