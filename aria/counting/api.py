"""Shared frontend dispatch for model counting from formulas and files."""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, cast

import z3

from aria.counting.arith.arith_counting_latte import ArithModelCounter
from aria.counting.bool.dimacs_counting import count_dimacs_solutions_parallel
from aria.counting.bool.z3py_expr_counting import count_z3_result
from aria.counting.core import CountResult, exact_count_result, unsupported_count_result
from aria.counting.bv import BVModelCounter
from aria.sampling.general_sampler import count_solutions
from aria.utils.z3.expr import get_variables


def load_formula_and_content(filename: str) -> Tuple[str, str, Optional[z3.BoolRef]]:
    """Load raw file content, detect format, and parse SMT-LIB if possible."""

    with open(filename, encoding="utf-8") as f:
        content = f.read()

    file_ext = Path(filename).suffix.lower()
    if file_ext in (".cnf", ".dimacs"):
        return content, "dimacs", None
    if file_ext == ".smt2":
        try:
            formula = cast(z3.BoolRef, z3.And(*z3.parse_smt2_file(filename)))
        except z3.Z3Exception:
            formula = None
        return content, "smtlib2", formula
    return content, "smtlib2", None


def detect_theory(
    theory: str,
    format_type: str,
    content: str,
    formula: Optional[z3.BoolRef],
) -> str:
    """Infer the theory when auto detection is requested."""

    if theory != "auto":
        return theory

    if format_type == "dimacs":
        return "bool"

    if formula is not None:
        variables = get_variables(formula)
        has_bv = any(z3.is_bv(v) for v in variables)
        has_real = any(z3.is_real(v) for v in variables)
        has_int = any(z3.is_int(v) for v in variables)

        if has_bv:
            return "bv"
        if has_real:
            return "generic"
        if has_int:
            analysis = ArithModelCounter().analyze(formula)
            if analysis.status == "exact":
                return "arith"
            return "generic"
        return "bool"

    if "BitVec" in content or "(_ bv" in content:
        return "bv"
    return "bool"


def resolve_projection_variables(
    formula: z3.BoolRef,
    names: Optional[Sequence[str]],
) -> Optional[List[z3.ExprRef]]:
    """Resolve user-provided variable names against a parsed formula."""

    if not names:
        return None

    variable_map: Dict[str, z3.ExprRef] = {str(var): var for var in get_variables(formula)}
    resolved: List[z3.ExprRef] = []
    missing: List[str] = []
    for name in names:
        if name in variable_map:
            resolved.append(variable_map[name])
        else:
            missing.append(name)

    if missing:
        raise ValueError(
            "Projection variable(s) not found in formula: {}".format(
                ", ".join(missing)
            )
        )

    return resolved


def count_result(
    formula: z3.BoolRef,
    theory: str = "auto",
    method: str = "auto",
    variables: Optional[Sequence[z3.ExprRef]] = None,
    timeout: Optional[int] = None,
) -> CountResult:
    """Count models for a parsed Z3 formula and return a structured result."""

    detected_theory = detect_theory(theory, "smtlib2", "", formula)

    if detected_theory == "bool":
        bool_method = method if method != "solver" else "auto"
        result = count_z3_result(
            formula,
            variables=variables,
            method=bool_method,
        )
        result.metadata.setdefault("format", "smtlib2")
        result.metadata.setdefault("theory", detected_theory)
        result.metadata.setdefault("method", bool_method)
        return result

    if detected_theory == "bv":
        counter = BVModelCounter()
        counter.init_from_fml(formula)
        bv_method = method if method != "solver" else "auto"
        result = counter.count_models(method=bv_method, variables=variables)
        result.metadata.setdefault("format", "smtlib2")
        result.metadata.setdefault("theory", detected_theory)
        result.metadata.setdefault("method", bv_method)
        return result

    if detected_theory == "arith":
        arith_method = method
        if method in ("auto", "solver"):
            arith_method = "auto"
        elif method == "enumeration":
            arith_method = "enumeration"
        elif method == "latte":
            arith_method = "latte"
        else:
            raise ValueError(f"Unsupported method for arithmetic theory: {method}")

        counter = ArithModelCounter()
        result = counter.count_models(
            formula=formula,
            variables=cast(Optional[List[z3.ExprRef]], variables),
            method=arith_method,
        )
        result.metadata.setdefault("format", "smtlib2")
        result.metadata.setdefault("theory", detected_theory)
        result.metadata.setdefault("method", arith_method)
        return result

    if variables:
        return unsupported_count_result(
            backend="generic",
            reason="projection is not supported for generic SMT counting",
            projection=[str(var) for var in variables],
            format="smtlib2",
            theory=detected_theory,
            method=method,
        )

    _ = timeout
    return unsupported_count_result(
        backend="generic",
        reason="formula-level generic SMT counting is not supported",
        format="smtlib2",
        theory=detected_theory,
        method=method,
    )


def count(
    formula: z3.BoolRef,
    theory: str = "auto",
    method: str = "auto",
    variables: Optional[Sequence[z3.ExprRef]] = None,
    timeout: Optional[int] = None,
) -> int:
    """Count models for a parsed Z3 formula and return an integer count."""

    result = count_result(
        formula,
        theory=theory,
        method=method,
        variables=variables,
        timeout=timeout,
    )
    if result.count is None:
        raise ValueError(f"Model counting failed: {result.status}: {result.reason}")
    return int(result.count)


def count_result_from_file(
    filename: str,
    theory: str = "auto",
    method: str = "auto",
    timeout: Optional[int] = None,
    project: Optional[Sequence[str]] = None,
) -> CountResult:
    """Count models from a file and return a structured result."""

    content, format_type, parsed_formula = load_formula_and_content(filename)
    detected_theory = detect_theory(theory, format_type, content, parsed_formula)

    if detected_theory == "bool":
        if format_type == "dimacs":
            if project:
                return unsupported_count_result(
                    backend="dimacs-sharpsat",
                    reason="projection is not supported for DIMACS CLI inputs",
                    projection=list(project),
                    format=format_type,
                    theory=detected_theory,
                )
            lines = content.strip().split("\n")
            header = []
            clauses = []
            for line in lines:
                line = line.strip()
                if not line or line.startswith("c"):
                    continue
                if line.startswith("p"):
                    header.append(line)
                else:
                    clauses.append(line.rstrip(" 0").strip())

            count = count_dimacs_solutions_parallel(header, clauses)
            return exact_count_result(
                float(count),
                backend="dimacs-sharpsat",
                format=format_type,
                theory=detected_theory,
                method=method,
            )

        if parsed_formula is None:
            raise ValueError("Failed to parse SMT-LIB2 Boolean formula")

        projection_vars = resolve_projection_variables(parsed_formula, project)
        result = count_result(
            parsed_formula,
            theory=detected_theory,
            method=method,
            variables=projection_vars,
            timeout=timeout,
        )
        result.metadata["format"] = format_type
        return result

    if detected_theory == "bv":
        if parsed_formula is None:
            raise ValueError("Failed to parse SMT-LIB2 bit-vector formula")

        projection_vars = resolve_projection_variables(parsed_formula, project)
        result = count_result(
            parsed_formula,
            theory=detected_theory,
            method=method,
            variables=projection_vars,
            timeout=timeout,
        )
        result.metadata["format"] = format_type
        return result

    if detected_theory == "arith":
        if parsed_formula is None:
            raise ValueError("Failed to parse SMT-LIB2 arithmetic formula")

        projection_vars = resolve_projection_variables(parsed_formula, project)
        result = count_result(
            parsed_formula,
            theory=detected_theory,
            method=method,
            variables=projection_vars,
            timeout=timeout,
        )
        result.metadata["format"] = format_type
        return result

    if project:
        return unsupported_count_result(
            backend="generic",
            reason="projection is not supported for generic SMT counting",
            projection=list(project),
            format=format_type,
            theory=detected_theory,
        )

    count = count_solutions(content, fmt=format_type, timeout=timeout)
    return exact_count_result(
        float(count),
        backend="generic",
        format=format_type,
        theory=detected_theory,
        method=method,
    )


def count_from_file(
    filename: str,
    theory: str = "auto",
    method: str = "auto",
    timeout: Optional[int] = None,
    project: Optional[Sequence[str]] = None,
) -> int:
    """Count models from a file and return an integer count."""

    result = count_result_from_file(
        filename,
        theory=theory,
        method=method,
        timeout=timeout,
        project=project,
    )
    if result.count is None:
        raise ValueError(f"Model counting failed: {result.status}: {result.reason}")
    return int(result.count)


__all__ = [
    "count",
    "count_from_file",
    "count_result",
    "count_result_from_file",
    "detect_theory",
    "load_formula_and_content",
    "resolve_projection_variables",
]
