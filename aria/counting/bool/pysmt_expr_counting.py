"""Counting interfaces for pySMT Boolean formulas."""

from timeit import default_timer as counting_timer
from typing import List, Optional, Sequence, Tuple

from pysmt.shortcuts import Solver as PySMTSolver, Not, get_free_variables, Or
from pysmt.shortcuts import TRUE, FALSE
from pysmt.rewritings import cnf

from aria.counting.bool.dimacs_counting import call_approxmc
from aria.counting.core import CountResult, exact_count_result, unsupported_count_result

def count_pysmt_models_by_enumeration(
    formula,
    max_models: Optional[int] = None,
    variables: Optional[Sequence] = None,
) -> int:
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
    vars_to_count = list(variables) if variables is not None else list(
        get_free_variables(formula)
    )

    if len(vars_to_count) == 0:
        return 1 if solver.solve() else 0

    while solver.solve():
        count += 1
        if max_models and count > max_models:
            return -1

        model = solver.get_model()
        # Create blocking clause for all variables
        block = []
        for var in vars_to_count:
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
    simplified = formula.simplify()
    if simplified == TRUE():
        return ["p cnf 0 0"], []
    if simplified == FALSE():
        return ["p cnf 0 1"], [""]

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


def count_pysmt_solutions(
    formula,
    parallel: bool = False,
    variables: Optional[Sequence] = None,
    method: str = "auto",
) -> int:
    """
    Count solutions for a pySMT formula.

    Args:
        formula: PySMT formula to count solutions for
        parallel (bool): Whether to use parallel counting

    Returns:
        int: Number of solutions
    """
    _ = parallel
    result = count_pysmt_result(
        formula, parallel=parallel, variables=variables, method=method
    )
    if result.count is None:
        raise ValueError(f"pySMT model counting failed: {result.status}: {result.reason}")
    return int(result.count)


def count_pysmt_result(
    formula,
    parallel: bool = False,
    variables: Optional[Sequence] = None,
    method: str = "auto",
) -> CountResult:
    """Count satisfying assignments for a pySMT Boolean formula."""

    time_start = counting_timer()
    simplified = formula.simplify()
    free_variables = list(get_free_variables(formula))
    vars_to_count = list(variables) if variables is not None else free_variables
    projection = [str(var) for var in vars_to_count]
    num_variables = len(free_variables)

    free_var_ids = {var for var in free_variables}
    for var in vars_to_count:
        if var not in free_var_ids:
            return unsupported_count_result(
                backend="pysmt-enumeration",
                reason=f"projection variable not found in formula: {var}",
                runtime_s=counting_timer() - time_start,
                projection=projection,
                num_variables=num_variables,
            )

    if simplified == TRUE():
        return exact_count_result(
            float(2 ** len(vars_to_count)),
            backend="pysmt-enumeration",
            runtime_s=counting_timer() - time_start,
            projection=projection,
            num_variables=num_variables,
            simplification="tautology",
        )

    if simplified == FALSE():
        return exact_count_result(
            0.0,
            backend="pysmt-enumeration",
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
        _, clauses = pysmt_to_dimacs(formula)
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
            backend="pysmt-none",
            reason=f"unknown Boolean counting method: {method}",
            runtime_s=counting_timer() - time_start,
            projection=projection,
            num_variables=num_variables,
        )

    if method == "auto" and variables is None and num_variables > 16:
        approx_result = count_pysmt_result(
            formula,
            parallel=parallel,
            variables=variables,
            method="approx",
        )
        if approx_result.status != "unsupported":
            approx_result.metadata["selection"] = "auto"
            return approx_result

    count = count_pysmt_models_by_enumeration(formula, variables=vars_to_count)
    return exact_count_result(
        float(count),
        backend="pysmt-enumeration",
        runtime_s=counting_timer() - time_start,
        projection=projection,
        num_variables=num_variables,
        parallel_requested=parallel,
    )
