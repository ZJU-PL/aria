# coding: utf-8
"""SAT-based enumeration of prime implicants and prime implicates."""

from typing import Iterable, List, Optional, Sequence

from pysat.formula import CNF
from pysat.solvers import Solver


def _normalize_clause(literals: Iterable[int]) -> List[int]:
    normalized = sorted(set(literals), key=lambda lit: (abs(lit), lit < 0))
    return normalized


def _normalize_term(literals: Iterable[int]) -> List[int]:
    normalized = sorted(set(literals), key=lambda lit: (abs(lit), lit < 0))
    return normalized


def _formula_vars(formula: CNF) -> List[int]:
    variables = set()
    for clause in formula.clauses:
        for literal in clause:
            variables.add(abs(literal))
    return sorted(variables)


def _project_model(model: Sequence[int], variables: Sequence[int]) -> List[int]:
    assignment = set(model)
    return [var if var in assignment else -var for var in variables]


def _reduce_implicant(dual_solver: Solver, candidate: Sequence[int]) -> List[int]:
    reduced = list(candidate)
    changed = True
    while changed:
        changed = False
        for literal in list(reduced):
            trial = [lit for lit in reduced if lit != literal]
            if not dual_solver.solve(assumptions=trial):
                reduced = trial
                changed = True
                break
    return _normalize_term(reduced)


def _enumerate_projected_prime_implicants(
    formula: CNF, variables: Optional[Sequence[int]] = None, solver_name: str = "cd"
) -> List[List[int]]:
    projection = list(variables) if variables is not None else _formula_vars(formula)

    if not projection:
        with Solver(name=solver_name, bootstrap_with=formula.clauses) as solver:
            return [[]] if solver.solve() else []

    implicants = []
    dual_formula = formula.negate()
    with Solver(name=solver_name, bootstrap_with=formula.clauses) as solver, Solver(
        name=solver_name, bootstrap_with=dual_formula.clauses
    ) as dual_solver:
        while solver.solve():
            model = solver.get_model()
            candidate = _project_model(model, projection)
            implicant = _reduce_implicant(dual_solver, candidate)
            implicants.append(implicant)
            solver.add_clause([-literal for literal in implicant])

    implicants = sorted({tuple(implicant) for implicant in implicants})
    return [list(implicant) for implicant in implicants]


def _reduce_implicate(formula_solver: Solver, candidate: Sequence[int]) -> List[int]:
    reduced = list(candidate)
    changed = True
    while changed:
        changed = False
        for literal in list(reduced):
            trial = [lit for lit in reduced if lit != literal]
            complement_term = [-lit for lit in trial]
            if not formula_solver.solve(assumptions=complement_term):
                reduced = trial
                changed = True
                break
    return _normalize_clause(reduced)


def enumerate_prime_implicants(formula: CNF, solver_name: str = "cd") -> List[List[int]]:
    """Enumerate all prime implicants of a Boolean formula in CNF.

    The result is returned as a list of cubes, where each cube is a list of
    signed integers. A positive integer ``x`` denotes variable ``x`` and a
    negative integer ``-x`` denotes its negation.
    """

    return _enumerate_projected_prime_implicants(formula, solver_name=solver_name)


def enumerate_prime_implicates(formula: CNF, solver_name: str = "cd") -> List[List[int]]:
    """Enumerate all prime implicates of a Boolean formula in CNF."""

    variables = _formula_vars(formula)
    if not variables:
        with Solver(name=solver_name, bootstrap_with=formula.clauses) as solver:
            return [] if solver.solve() else [[]]

    negated = formula.negate()
    implicates = []
    with Solver(name=solver_name, bootstrap_with=negated.clauses) as negated_solver, Solver(
        name=solver_name, bootstrap_with=formula.clauses
    ) as formula_solver:
        while negated_solver.solve():
            model = negated_solver.get_model()
            falsifying_assignment = _project_model(model, variables)
            candidate_clause = [-literal for literal in falsifying_assignment]
            implicate = _reduce_implicate(formula_solver, candidate_clause)
            implicates.append(implicate)
            negated_solver.add_clause(implicate)

    implicates = sorted({tuple(implicate) for implicate in implicates})
    return [list(implicate) for implicate in implicates]


def prime_implicant_cover(formula: CNF, solver_name: str = "cd") -> CNF:
    """Return the DNF-style prime implicant cover as cubes."""

    return CNF(from_clauses=enumerate_prime_implicants(formula, solver_name=solver_name))


def prime_implicate_cover(formula: CNF, solver_name: str = "cd") -> CNF:
    """Return the CNF consisting of all prime implicates."""

    return CNF(from_clauses=enumerate_prime_implicates(formula, solver_name=solver_name))
