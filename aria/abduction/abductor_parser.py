"""Parse abduction problems from SMT-LIB2 into Z3 expressions."""

from typing import Any, Dict, List, Optional, Tuple

import z3
from aria.utils.sexpr import SExprParser

_SMT2_PRELUDE_KEY = "__smt2_prelude__"


def _skip_whitespace(text: str, idx: int) -> int:
    """Advance past whitespace characters."""
    while idx < len(text) and text[idx].isspace():
        idx += 1
    return idx


def extract_balanced_expr(text: str, start_idx: int = 0) -> str:
    """Extract the balanced parenthesized expression starting at ``start_idx``."""
    start_idx = _skip_whitespace(text, start_idx)
    if start_idx >= len(text) or text[start_idx] != "(":
        return ""

    depth = 0
    in_string = False
    escaped = False

    for idx in range(start_idx, len(text)):
        char = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[start_idx : idx + 1]

    return text[start_idx:]


def extract_all_commands(smt2_str: str) -> List[str]:
    """Extract all top-level SMT-LIB commands in source order."""
    commands: List[str] = []
    idx = 0
    length = len(smt2_str)

    while idx < length:
        idx = _skip_whitespace(smt2_str, idx)
        if idx >= length:
            break
        if smt2_str[idx] == ";":
            next_newline = smt2_str.find("\n", idx)
            idx = length if next_newline == -1 else next_newline + 1
            continue
        if smt2_str[idx] != "(":
            idx += 1
            continue

        command = extract_balanced_expr(smt2_str, idx)
        if not command:
            break
        commands.append(command)
        idx += len(command)

    return commands


def _parse_command(expr: str) -> Any:
    """Parse one SMT-LIB command into a Python S-expression."""
    sexpr = SExprParser.parse(expr)
    if sexpr is None:
        raise ValueError(f"Failed to parse SMT-LIB command: {expr}")
    return sexpr


def _collect_declarations(commands: List[str]) -> Tuple[Dict[str, Any], Dict[str, z3.SortRef]]:
    """Collect constants, functions, and sorts declared in the input."""
    declarations: Dict[str, Any] = {}
    datatypes: Dict[str, z3.SortRef] = {}

    for command in commands:
        sexpr = _parse_command(command)
        if not isinstance(sexpr, list) or not sexpr:
            continue

        head = sexpr[0]
        if head == "declare-sort" and len(sexpr) >= 2:
            sort_name = str(sexpr[1])
            datatypes[sort_name] = z3.DeclareSort(sort_name)
            continue
        if head == "define-sort" and len(sexpr) >= 4:
            sort_name = str(sexpr[1])
            parameters = sexpr[2] if isinstance(sexpr[2], list) else []
            if parameters:
                raise ValueError(f"Parameterized sort aliases are not supported: {sort_name}")
            datatypes[sort_name] = _sexpr_to_sort(sexpr[3], datatypes)
            continue

        if head not in {"declare-fun", "declare-const"} or len(sexpr) < 3:
            continue

        name = str(sexpr[1])
        if head == "declare-const":
            domain_spec: List[Any] = []
            range_spec = sexpr[2]
        else:
            domain_spec = list(sexpr[2]) if isinstance(sexpr[2], list) else []
            range_spec = sexpr[3]

        domain_sorts = [_sexpr_to_sort(item, datatypes) for item in domain_spec]
        range_sort = _sexpr_to_sort(range_spec, datatypes)

        if domain_sorts:
            declarations[name] = z3.Function(name, *domain_sorts, range_sort)
        else:
            declarations[name] = z3.Const(name, range_sort)

    return declarations, datatypes


def _sexpr_to_sort(sort_expr: Any, datatypes: Dict[str, z3.SortRef]) -> z3.SortRef:
    """Convert a parsed SMT-LIB sort expression into a Z3 sort."""
    if isinstance(sort_expr, str):
        if sort_expr == "Int":
            return z3.IntSort()
        if sort_expr == "Real":
            return z3.RealSort()
        if sort_expr == "Bool":
            return z3.BoolSort()
        if sort_expr == "String":
            return z3.StringSort()
        if sort_expr in datatypes:
            return datatypes[sort_expr]
        raise ValueError(f"Unsupported sort: {sort_expr}")

    if isinstance(sort_expr, list) and sort_expr:
        head = sort_expr[0]
        if head == "_" and len(sort_expr) == 3 and sort_expr[1] == "BitVec":
            return z3.BitVecSort(int(sort_expr[2]))
        if head == "Array" and len(sort_expr) == 3:
            return z3.ArraySort(
                _sexpr_to_sort(sort_expr[1], datatypes),
                _sexpr_to_sort(sort_expr[2], datatypes),
            )

    raise ValueError(f"Unsupported sort expression: {sort_expr}")


def parse_smt2_expr(expr_str: str, variables: Dict[str, Any]) -> z3.ExprRef:
    """Parse one SMT-LIB2 expression using the provided symbol environment."""
    return parse_expr(expr_str, variables)


def extract_assertion(smt2_str: str, start: int) -> Tuple[Optional[str], int]:
    """Extract the body of an ``assert`` command starting at ``start``."""
    command = extract_balanced_expr(smt2_str, start)
    if not command:
        return None, start

    sexpr = SExprParser.parse(command)
    if not isinstance(sexpr, list) or len(sexpr) != 2 or sexpr[0] != "assert":
        return None, start + len(command)

    body = SExprParser.sexpr_to_string(sexpr[1])
    return body, start + len(command)


def extract_abduction_goal(smt2_str: str) -> Optional[str]:
    """Extract the goal expression from the first ``get-abduct`` command."""
    for command in extract_all_commands(smt2_str):
        sexpr = SExprParser.parse(command)
        if not isinstance(sexpr, list) or not sexpr or sexpr[0] != "get-abduct":
            continue
        if len(sexpr) < 3:
            raise ValueError("Malformed get-abduct command")
        return SExprParser.sexpr_to_string(sexpr[2])
    return None


def _extract_assertions(commands: List[str]) -> List[str]:
    """Return all assertion bodies in source order."""
    assertions: List[str] = []
    for command in commands:
        sexpr = SExprParser.parse(command)
        if isinstance(sexpr, list) and len(sexpr) == 2 and sexpr[0] == "assert":
            assertions.append(SExprParser.sexpr_to_string(sexpr[1]))
    return assertions


def _build_prelude(commands: List[str]) -> str:
    """Build the SMT-LIB prelude used when parsing isolated expressions."""
    prelude_commands: List[str] = []
    for command in commands:
        sexpr = SExprParser.parse(command)
        if not isinstance(sexpr, list) or not sexpr:
            continue
        if sexpr[0] in {"declare-sort", "define-sort", "define-fun"}:
            prelude_commands.append(command)
    return "\n".join(prelude_commands)


def _with_internal_prelude(variables: Dict[str, Any], prelude: str) -> Dict[str, Any]:
    """Return a parsing environment augmented with internal parser metadata."""
    parser_env = dict(variables)
    parser_env[_SMT2_PRELUDE_KEY] = prelude
    return parser_env


def parse_abduction_problem(
    smt2_str: str,
) -> Tuple[z3.BoolRef, z3.BoolRef, Dict[str, Any]]:
    """Parse an SMT-LIB abduction problem into precondition, goal, and symbols."""
    commands = extract_all_commands(smt2_str)
    variables, _ = _collect_declarations(commands)
    parser_env = _with_internal_prelude(variables, _build_prelude(commands))

    assertions = [parse_expr(expr, parser_env) for expr in _extract_assertions(commands)]

    goal_expr = extract_abduction_goal(smt2_str)
    if not goal_expr:
        raise ValueError("No abduction goal found in the input")
    goal = parse_expr(goal_expr, parser_env)

    if assertions:
        precond = z3.And(*assertions) if len(assertions) > 1 else assertions[0]
    else:
        precond = z3.BoolVal(True)
    return precond, goal, variables


def parse_expr(expr_str: str, variables: Dict[str, Any]) -> z3.ExprRef:
    """Parse an SMT-LIB expression using Z3's parser with custom declarations."""
    prelude = variables.get(_SMT2_PRELUDE_KEY, "")
    decls = {
        name: value
        for name, value in variables.items()
        if name != _SMT2_PRELUDE_KEY
    }
    try:
        wrapped = "\n".join(part for part in [prelude, f"(assert {expr_str})"] if part)
        exprs = z3.parse_smt2_string(wrapped, decls=decls)
    except z3.Z3Exception as exc:
        raise ValueError(f"Failed to parse expression: {expr_str}. Error: {exc}") from exc

    if len(exprs) != 1:
        raise ValueError(f"Expected a single expression, got {len(exprs)} from: {expr_str}")
    return exprs[0]
