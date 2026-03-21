"""
Internal helpers for probabilistic reasoning.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import z3
from pysat.formula import CNF

from aria.utils.z3_expr_utils import get_variables, z3_value_to_python


def clone_cnf(cnf: CNF) -> CNF:
    copied = CNF()
    copied.nv = cnf.nv
    copied.clauses = [list(clause) for clause in cnf.clauses]
    return copied


def merge_cnfs(*cnfs: CNF) -> CNF:
    merged = CNF()
    max_nv = 0
    for cnf in cnfs:
        if cnf is None:
            continue
        max_nv = max(max_nv, cnf.nv)
        merged.extend([list(clause) for clause in cnf.clauses])
    merged.nv = max_nv
    return merged


def literals_to_cnf(literals: Optional[Sequence[int]]) -> CNF:
    cnf = CNF()
    if literals is None:
        return cnf
    for lit in literals:
        cnf.append([int(lit)])
        cnf.nv = max(cnf.nv, abs(int(lit)))
    return cnf


def normalize_literal_sequence(literals: Optional[Sequence[int]]) -> List[int]:
    if literals is None:
        return []
    normalized = []
    seen = {}
    for lit in literals:
        literal = int(lit)
        var = abs(literal)
        sign = 1 if literal > 0 else -1
        if var in seen and seen[var] != sign:
            raise ValueError(
                "Contradictory evidence/query assignments for variable {}".format(var)
            )
        seen[var] = sign
        normalized.append(literal)
    return normalized


def z3_value_from_python(var: z3.ExprRef, value: Any) -> z3.ExprRef:
    if var.sort() == z3.BoolSort():
        return z3.BoolVal(bool(value))
    if var.sort() == z3.IntSort():
        return z3.IntVal(int(value))
    if var.sort() == z3.RealSort():
        if isinstance(value, bool):
            raise ValueError("Boolean value is not valid for Real variable {}".format(var))
        if isinstance(value, int):
            return z3.RealVal(str(value))
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("Non-finite Real assignment for {}".format(var))
            return z3.RealVal(repr(value))
        return z3.RealVal(str(value))
    raise ValueError("Unsupported variable sort for {}".format(var))


def simplify_under_assignment(expr: z3.ExprRef, assignment: Dict[str, Any]) -> z3.ExprRef:
    substitutions = []
    for var in get_variables(expr):
        var_name = str(var)
        if var_name in assignment:
            substitutions.append((var, z3_value_from_python(var, assignment[var_name])))
    if substitutions:
        expr = z3.substitute(expr, *substitutions)
    return z3.simplify(expr)


def assignment_satisfies(formula: z3.ExprRef, assignment: Dict[str, Any]) -> bool:
    evaluated = simplify_under_assignment(formula, assignment)
    if z3.is_true(evaluated):
        return True
    if z3.is_false(evaluated):
        return False
    solver = z3.Solver()
    solver.add(evaluated)
    return solver.check() == z3.sat


def evaluate_term(term: z3.ExprRef, assignment: Dict[str, Any]) -> Any:
    evaluated = simplify_under_assignment(term, assignment)
    return z3_value_to_python(evaluated)


def finite_support(bounds: Dict[str, Tuple[float, float]]) -> bool:
    for min_val, max_val in bounds.values():
        if not math.isfinite(min_val) or not math.isfinite(max_val):
            return False
    return True
