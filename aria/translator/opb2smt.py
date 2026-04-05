import re
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence, Set, Tuple


COMPARATOR_RE = re.compile(r"(<=|>=|!=|=|<|>)")
HEADER_ENTRY_RE = re.compile(r"#([A-Za-z_][A-Za-z0-9_-]*)\s*=\s*([+-]?\d+)")
WEIGHT_PREFIX_RE = re.compile(r"^\[\s*([+-]?\d+)\s*\]\s*(.*)$")
LITERAL_RE = re.compile(r"~?[A-Za-z_][A-Za-z0-9_./{}()\[\],-]*")


@dataclass(frozen=True)
class OPBTerm:
    coefficient: int
    factors: Tuple[Tuple[int, str], ...]


@dataclass(frozen=True)
class OPBConstraint:
    weight: Optional[int]
    comparator: str
    rhs: int
    terms: Tuple[OPBTerm, ...]


@dataclass(frozen=True)
class OPBObjective:
    kind: str
    terms: Tuple[OPBTerm, ...]


@dataclass(frozen=True)
class OPBProgram:
    metadata: Tuple[Tuple[str, int], ...]
    soft_top: Optional[int]
    objectives: Tuple[OPBObjective, ...]
    constraints: Tuple[OPBConstraint, ...]
    variables: Tuple[str, ...]
    nonlinear: bool


def _normalize_factor(token: str) -> Tuple[int, str]:
    sign = -1 if token.startswith("~") else 1
    variable = token[1:] if token.startswith("~") else token
    return sign, variable


def _smt_symbol(variable: str) -> str:
    return f"|{variable}|"


def _parse_term_tokens(tokens: Sequence[str], expr: str) -> Tuple[OPBTerm, Set[str]]:
    if not tokens:
        raise ValueError(f"Invalid OPB term in expression: {expr}")

    coeff = 1
    start_index = 0
    if re.fullmatch(r"[+-]?\d+", tokens[0]):
        coeff = int(tokens[0])
        start_index = 1

    if start_index >= len(tokens):
        raise ValueError(f"Missing OPB literal after coefficient in: {expr}")

    factors = []
    variables = set()
    for token in tokens[start_index:]:
        if not LITERAL_RE.fullmatch(token):
            raise ValueError(f"Invalid OPB literal token: {token}")
        literal_sign, variable = _normalize_factor(token)
        factors.append((literal_sign, variable))
        variables.add(variable)

    return OPBTerm(coefficient=coeff, factors=tuple(factors)), variables


def _parse_terms(expr: str) -> Tuple[Tuple[OPBTerm, ...], Set[str], bool]:
    raw_tokens = expr.split()
    if not raw_tokens:
        return tuple(), set(), False

    terms: List[OPBTerm] = []
    variables: Set[str] = set()
    current: List[str] = []
    sign = 1

    for token in raw_tokens:
        if token in {"+", "-"}:
            if current:
                term, term_vars = _parse_term_tokens(current, expr)
                if sign == -1:
                    term = OPBTerm(coefficient=-term.coefficient, factors=term.factors)
                terms.append(term)
                variables.update(term_vars)
                current = []
            sign = 1 if token == "+" else -1
            continue
        current.append(token)

    if current:
        term, term_vars = _parse_term_tokens(current, expr)
        if sign == -1:
            term = OPBTerm(coefficient=-term.coefficient, factors=term.factors)
        terms.append(term)
        variables.update(term_vars)

    nonlinear = any(len(term.factors) > 1 for term in terms)
    return tuple(terms), variables, nonlinear


def _factor_expr(factor: Tuple[int, str]) -> str:
    sign, variable = factor
    symbol = _smt_symbol(variable)
    return symbol if sign > 0 else f"(- 1 {symbol})"


def _term_expr(term: OPBTerm) -> str:
    if not term.factors:
        base_expr = "1"
    else:
        factor_exprs = [_factor_expr(factor) for factor in term.factors]
        if len(factor_exprs) == 1:
            base_expr = factor_exprs[0]
        else:
            base_expr = f"(* {' '.join(factor_exprs)})"

    if term.coefficient == 1:
        return base_expr
    if term.coefficient == -1:
        return f"(- {base_expr})"
    return f"(* {term.coefficient} {base_expr})"


def _sum_expr(terms: Sequence[OPBTerm]) -> str:
    rendered = [_term_expr(term) for term in terms]
    if not rendered:
        return "0"
    if len(rendered) == 1:
        return rendered[0]
    return f"(+ {' '.join(rendered)})"


def _constraint_expr(constraint: OPBConstraint) -> str:
    lhs = _sum_expr(constraint.terms)
    rhs = constraint.rhs
    comparator = constraint.comparator
    if comparator == "<":
        return f"(<= {lhs} {rhs - 1})"
    if comparator == ">":
        return f"(>= {lhs} {rhs + 1})"
    if comparator == "!=":
        return f"(not (= {lhs} {rhs}))"
    return f"({comparator} {lhs} {rhs})"


def _read_statements(input_path: str) -> Tuple[Tuple[Tuple[str, int], ...], List[str]]:
    statements = []
    metadata = []
    current = []

    with open(input_path, "r", encoding="utf-8") as input_file:
        for raw_line in input_file:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("*"):
                metadata.extend((key.lower(), int(value)) for key, value in HEADER_ENTRY_RE.findall(line))
                continue
            current.append(line)
            if ";" in line:
                statement = " ".join(current)
                if statement.count(";") != 1 or not statement.rstrip().endswith(";"):
                    raise ValueError(f"Invalid OPB statement: {statement}")
                statements.append(statement[:-1].strip())
                current = []

    if current:
        raise ValueError(f"Unterminated OPB statement: {' '.join(current)}")

    return tuple(metadata), statements


def parse_opb_program(input_path: str) -> OPBProgram:
    metadata, statements = _read_statements(input_path)
    objectives: List[OPBObjective] = []
    constraints: List[OPBConstraint] = []
    variables: Set[str] = set()
    nonlinear = False
    soft_top: Optional[int] = None

    for statement in statements:
        lowered = statement.lower()
        if lowered.startswith("soft:"):
            body = statement.split(":", 1)[1].strip()
            soft_top = None if body == "" else int(body)
            continue

        if lowered.startswith("min:") or lowered.startswith("max:"):
            kind = "minimize" if lowered.startswith("min:") else "maximize"
            body = statement.split(":", 1)[1].strip()
            terms, term_vars, has_products = _parse_terms(body)
            variables.update(term_vars)
            nonlinear = nonlinear or has_products
            objectives.append(OPBObjective(kind=kind, terms=terms))
            continue

        weight = None
        constraint_body = statement
        weight_match = WEIGHT_PREFIX_RE.match(statement)
        if weight_match is not None:
            weight = int(weight_match.group(1))
            constraint_body = weight_match.group(2).strip()

        comparator_match = COMPARATOR_RE.search(constraint_body)
        if comparator_match is None:
            raise ValueError(f"Invalid OPB constraint: {statement}")

        comparator = comparator_match.group(1)
        lhs = constraint_body[: comparator_match.start()].strip()
        rhs = int(constraint_body[comparator_match.end() :].strip())
        terms, term_vars, has_products = _parse_terms(lhs)
        variables.update(term_vars)
        nonlinear = nonlinear or has_products
        constraints.append(
            OPBConstraint(weight=weight, comparator=comparator, rhs=rhs, terms=terms)
        )

    return OPBProgram(
        metadata=metadata,
        soft_top=soft_top,
        objectives=tuple(objectives),
        constraints=tuple(constraints),
        variables=tuple(sorted(variables)),
        nonlinear=nonlinear,
    )


def convert_opb_to_smt2(input_path, output_path):
    program = parse_opb_program(input_path)

    has_soft_constraints = any(constraint.weight is not None for constraint in program.constraints)
    logic = "QF_NIA" if program.nonlinear else "QF_LIA"
    if has_soft_constraints or program.objectives:
        lines = [f"(set-logic {logic})"]
    else:
        lines = [f"(set-logic {logic})"]

    for variable in program.variables:
        symbol = _smt_symbol(variable)
        lines.append(f"(declare-const {symbol} Int)")
        lines.append(f"(assert (<= 0 {symbol}))")
        lines.append(f"(assert (<= {symbol} 1))")

    top_weight = program.soft_top
    for constraint in program.constraints:
        expr = _constraint_expr(constraint)
        if constraint.weight is None:
            lines.append(f"(assert {expr})")
            continue

        if top_weight is not None and constraint.weight >= top_weight:
            lines.append(f"(assert {expr})")
        else:
            lines.append(f"(assert-soft {expr} :weight {constraint.weight})")

    for objective in program.objectives:
        lines.append(f"({objective.kind} {_sum_expr(objective.terms)})")

    lines.append("(check-sat)")

    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(lines) + "\n")

    return output_path


def main(argv):
    if len(argv) < 3:
        print(f"usage {argv[0]} INPUT_OPB OUTPUT_SMT2")
        return 1
    convert_opb_to_smt2(argv[1], argv[2])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
