"""SMT parsing helpers for NL abduction.

We keep a small surface area: the LLM is asked to output *terms* (Bool)
for domain/premise/conclusion/psi. We then wrap them with declarations
and parse via Z3.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import z3

from .data_structures import SmtVarDecl


class SmtParseError(ValueError):
    pass


_CODE_BLOCK_RE = re.compile(
    r"```(?:smt|lisp|smt-lib|smt2|smtlib|smtlib2|json)?\s*([\s\S]*?)```",
    re.IGNORECASE,
)


def extract_codeblock_or_raw(text: str) -> str:
    text = (text or "").strip()
    m = _CODE_BLOCK_RE.search(text)
    if m:
        return m.group(1).strip()
    return text


def normalize_smt_term(s: str) -> str:
    """Extract a single SMT term from a response.

    Accepts either a raw term `(and ...)` or a single `(assert <term>)`.
    """

    s = extract_codeblock_or_raw(s)
    s = s.strip()

    if not s:
        raise SmtParseError("Empty SMT term")

    # Common pattern: (assert <term>)
    if s.startswith("(assert"):
        # Very small parser: only strip the outermost (assert ...)
        m = re.match(r"^\(assert\s+([\s\S]+)\)\s*$", s)
        if m:
            s = m.group(1).strip()

    low = s.lower()
    if low in {"true", "false"}:
        return low

    # If multiple top-level s-exprs are present, keep the last fully-balanced one.
    # (Models sometimes include stray text, multiple candidates, etc.)
    if not s.startswith("("):
        raise SmtParseError("SMT term must be 'true'/'false' or start with '('")

    stack: List[str] = []
    start = -1
    last: Optional[str] = None
    for i, ch in enumerate(s):
        if ch == "(" and not stack:
            start = i
            stack.append(ch)
        elif ch == "(":
            stack.append(ch)
        elif ch == ")" and stack:
            stack.pop()
            if not stack and start >= 0:
                last = s[start : i + 1]

    if last is None:
        raise SmtParseError("Unbalanced parentheses in SMT term")
    return last.strip()


def _parse_sort(sort: str) -> Tuple[str, Optional[int]]:
    sort = sort.strip()
    if sort in {"Int", "Real", "Bool", "String"}:
        return sort, None
    m = re.match(r"^\(_\s+BitVec\s+(\d+)\)\s*$", sort)
    if m:
        return "BitVec", int(m.group(1))
    return sort, None


@dataclass
class SmtEnv:
    decls: List[SmtVarDecl]
    z3_vars: Dict[str, z3.ExprRef]
    smt_prelude: str


def build_env(decls: List[SmtVarDecl]) -> SmtEnv:
    z3_vars: Dict[str, z3.ExprRef] = {}
    smt_lines: List[str] = []

    seen_sorts: Dict[str, bool] = {}

    for d in decls:
        if not d.name or not d.sort:
            raise SmtParseError("Variable declarations must have name and sort")

        sort_kind, bv_size = _parse_sort(d.sort)

        if sort_kind == "Int":
            z3_vars[d.name] = z3.Int(d.name)
            smt_lines.append("(declare-const {0} Int)".format(d.name))
        elif sort_kind == "Real":
            z3_vars[d.name] = z3.Real(d.name)
            smt_lines.append("(declare-const {0} Real)".format(d.name))
        elif sort_kind == "Bool":
            z3_vars[d.name] = z3.Bool(d.name)
            smt_lines.append("(declare-const {0} Bool)".format(d.name))
        elif sort_kind == "String":
            z3_vars[d.name] = z3.String(d.name)
            smt_lines.append("(declare-const {0} String)".format(d.name))
        elif sort_kind == "BitVec":
            assert bv_size is not None
            z3_vars[d.name] = z3.BitVec(d.name, bv_size)
            smt_lines.append(
                "(declare-const {0} (_ BitVec {1}))".format(d.name, bv_size)
            )
        else:
            # Uninterpreted sort.
            if sort_kind not in seen_sorts:
                seen_sorts[sort_kind] = True
                smt_lines.append("(declare-sort {0} 0)".format(sort_kind))
            z3_vars[d.name] = z3.Const(d.name, z3.DeclareSort(sort_kind))
            smt_lines.append(
                "(declare-const {0} {1})".format(d.name, sort_kind)
            )

    return SmtEnv(decls=decls, z3_vars=z3_vars, smt_prelude="\n".join(smt_lines))


def parse_bool_term(term: str, env: SmtEnv) -> z3.BoolRef:
    t = normalize_smt_term(term)
    full = "{0}\n(assert {1})".format(env.smt_prelude, t)
    try:
        parsed = z3.parse_smt2_string(full)
    except Exception as e:  # pragma: no cover
        raise SmtParseError("Z3 parse_smt2_string failed: {0}".format(e)) from e

    if not parsed:
        raise SmtParseError("No formula parsed from SMT")
    f = parsed[0]
    if not z3.is_bool(f):
        raise SmtParseError("Expected Bool term, got: {0}".format(f.sort()))
    return f
