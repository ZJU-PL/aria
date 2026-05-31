"""SMT-LIB 2 parser — wraps Z3's native parser and extracts structured formula data."""

import sys
import z3
from dataclasses import dataclass, field
from typing import Optional
import re

sys.setrecursionlimit(100000)


@dataclass
class ParsedFormula:
    assertions: list          # list of Z3 expressions
    variables: dict           # name -> Z3 variable
    var_names: list           # ordered variable names
    logic: str                # QF_LIA, QF_BV, etc.
    source: str               # raw SMT-LIB string


def parse_file(path: str) -> ParsedFormula:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return parse_string(content)


def parse_string(smtlib_str: str) -> ParsedFormula:
    logic = _extract_logic(smtlib_str)
    solver = z3.Solver()
    solver.from_string(smtlib_str)
    assertions = list(solver.assertions())
    variables = _collect_variables(assertions)
    var_names = sorted(variables.keys())
    return ParsedFormula(
        assertions=assertions,
        variables=variables,
        var_names=var_names,
        logic=logic,
        source=smtlib_str,
    )


def _extract_logic(smtlib_str: str) -> str:
    match = re.search(r'\(set-logic\s+(\S+)\)', smtlib_str)
    return match.group(1) if match else "UNKNOWN"


def _collect_variables(assertions: list) -> dict:
    seen = {}
    visited = set()
    stack = list(assertions)
    while stack:
        expr = stack.pop()
        eid = expr.get_id()
        if eid in visited:
            continue
        visited.add(eid)
        if z3.is_const(expr) and expr.decl().kind() == z3.Z3_OP_UNINTERPRETED:
            seen[str(expr)] = expr
        stack.extend(expr.children())
    return seen
