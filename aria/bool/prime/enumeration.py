# coding: utf-8
"""SAT-based enumeration of prime implicants and prime implicates."""

from typing import Iterable, List, Sequence

from pysat.formula import CNF
from pysat.solvers import Solver


def _normalize_clause(literals: Iterable[int]) -> List[int]:
    return sorted(set(literals), key=lambda lit: (abs(lit), lit < 0))


def _normalize_term(literals: Iterable[int]) -> List[int]:
    return sorted(set(literals), key=lambda lit: (abs(lit), lit < 0))


def _formula_vars(formula: CNF) -> List[int]:
    variables = set()
    for clause in formula.clauses:
        for literal in clause:
            variables.add(abs(literal))
    return sorted(variables)


def _subsets_of_size(items: Sequence[int], size: int):
    if size == 0:
        yield []
        return

    if size > len(items):
        return

    def rec(start: int, chosen: List[int]):
        if len(chosen) == size:
            yield list(chosen)
            return

        remaining = size - len(chosen)
        limit = len(items) - remaining + 1
        for index in range(start, limit):
            chosen.append(items[index])
            yield from rec(index + 1, chosen)
            chosen.pop()

    yield from rec(0, [])


def _all_terms_for_variables(variables: Sequence[int]):
    assignment: List[int] = []

    def rec(index: int):
        if index == len(variables):
            yield list(assignment)
            return

        variable = variables[index]
        yield from rec(index + 1)

        assignment.append(variable)
        yield from rec(index + 1)
        assignment.pop()

        assignment.append(-variable)
        yield from rec(index + 1)
        assignment.pop()

    yield from rec(0)


def _is_unsat_under(formula: CNF, assumptions: Sequence[int], solver_name: str) -> bool:
    with Solver(name=solver_name, bootstrap_with=formula.clauses) as solver:
        return not solver.solve(assumptions=list(assumptions))


def _is_prime_implicant(formula: CNF, term: Sequence[int], solver_name: str) -> bool:
    negated = formula.negate()
    if not _is_unsat_under(negated, term, solver_name):
        return False

    for index in range(len(term)):
        reduced = list(term[:index]) + list(term[index + 1 :])
        if _is_unsat_under(negated, reduced, solver_name):
            return False

    return True


def _is_prime_implicate(formula: CNF, clause: Sequence[int], solver_name: str) -> bool:
    complement = [-literal for literal in clause]
    if not _is_unsat_under(formula, complement, solver_name):
        return False

    for index in range(len(clause)):
        reduced = list(clause[:index]) + list(clause[index + 1 :])
        reduced_complement = [-literal for literal in reduced]
        if _is_unsat_under(formula, reduced_complement, solver_name):
            return False

    return True


def enumerate_prime_implicants(formula: CNF, solver_name: str = "cd") -> List[List[int]]:
    """Enumerate all prime implicants of a Boolean formula in CNF."""

    variables = _formula_vars(formula)
    if not variables:
        with Solver(name=solver_name, bootstrap_with=formula.clauses) as solver:
            return [[]] if solver.solve() else []

    implicants = []
    for term in _all_terms_for_variables(variables):
        normalized = _normalize_term(term)
        if _is_prime_implicant(formula, normalized, solver_name):
            implicants.append(normalized)

    implicants = sorted({tuple(implicant) for implicant in implicants})
    return [list(implicant) for implicant in implicants]


def enumerate_prime_implicates(formula: CNF, solver_name: str = "cd") -> List[List[int]]:
    """Enumerate all prime implicates of a Boolean formula in CNF."""

    variables = _formula_vars(formula)
    if not variables:
        with Solver(name=solver_name, bootstrap_with=formula.clauses) as solver:
            return [] if solver.solve() else [[]]

    implicates = []
    literals = []
    for variable in variables:
        literals.append(variable)
        literals.append(-variable)

    for size in range(len(variables) + 1):
        for clause in _subsets_of_size(literals, size):
            normalized = _normalize_clause(clause)
            if len(normalized) != len(clause):
                continue
            if any(-literal in normalized for literal in normalized):
                continue
            if _is_prime_implicate(formula, normalized, solver_name):
                implicates.append(normalized)

    implicates = sorted({tuple(implicate) for implicate in implicates})
    return [list(implicate) for implicate in implicates]


def prime_implicant_cover(formula: CNF, solver_name: str = "cd") -> CNF:
    """Return the DNF-style prime implicant cover as cubes."""

    return CNF(from_clauses=enumerate_prime_implicants(formula, solver_name=solver_name))


def prime_implicate_cover(formula: CNF, solver_name: str = "cd") -> CNF:
    """Return the CNF consisting of all prime implicates."""

    return CNF(from_clauses=enumerate_prime_implicates(formula, solver_name=solver_name))
