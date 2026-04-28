"""Property-based testing demo for the QF_DTLIA sampler.

The demo treats the sampler as a constraint-directed test generator.  It
builds a small expression datatype, constrains the shape and integer literals
with a QF_DTLIA formula, samples candidate test inputs, and runs a toy property
check over the generated Python values.

Run from the repository root:

    python -m aria.sampling.dtlia.examples.property_based_testing_demo
"""

from typing import Any, Dict, List, Tuple

import z3

from aria.sampling import (
    Logic,
    SamplingMethod,
    SamplingOptions,
    SamplingResult,
    sample_models_from_formula,
)


def build_expression_problem() -> Tuple[z3.ExprRef, z3.ExprRef]:
    """Build a finite, structured QF_DTLIA input space for test generation."""
    expr = z3.Datatype("PBTExpr")
    expr.declare("lit", ("value", z3.IntSort()))
    expr.declare("neg", ("arg", expr))
    expr.declare("add", ("left", expr), ("right", expr))
    expr_sort = expr.create()

    root = z3.Const("root", expr_sort)

    def literal_bound(value: z3.ArithRef) -> z3.BoolRef:
        return z3.And(value >= -2, value <= 2)

    formula = z3.Or(
        z3.And(
            expr_sort.is_lit(root),
            literal_bound(expr_sort.value(root)),
        ),
        z3.And(
            expr_sort.is_neg(root),
            expr_sort.is_lit(expr_sort.arg(root)),
            literal_bound(expr_sort.value(expr_sort.arg(root))),
        ),
        z3.And(
            expr_sort.is_add(root),
            expr_sort.is_lit(expr_sort.left(root)),
            expr_sort.is_lit(expr_sort.right(root)),
            literal_bound(expr_sort.value(expr_sort.left(root))),
            literal_bound(expr_sort.value(expr_sort.right(root))),
            expr_sort.value(expr_sort.left(root))
            + expr_sort.value(expr_sort.right(root))
            >= -1,
            expr_sort.value(expr_sort.left(root))
            + expr_sort.value(expr_sort.right(root))
            <= 3,
        ),
    )
    return formula, root


def sample_expressions(diversity_mode: str, num_samples: int = 8) -> SamplingResult:
    """Sample valid expression values using one DTLIA selection policy."""
    formula, root = build_expression_problem()
    return sample_models_from_formula(
        formula,
        Logic.QF_DTLIA,
        SamplingOptions(
            method=SamplingMethod.SEARCH_TREE,
            num_samples=num_samples,
            random_seed=7,
            max_shapes=8,
            candidates_per_shape=5,
            tracked_terms=[root],
            return_full_model=True,
            diversity_mode=diversity_mode,
        ),
    )


def eval_reference(expr_value: Any) -> int:
    """Reference evaluator for sampled expression values."""
    if not isinstance(expr_value, dict):
        raise ValueError(f"Expected constructor value, got {expr_value!r}")

    constructor = expr_value["constructor"]
    fields = expr_value["fields"]
    if constructor == "lit":
        return int(fields[0])
    if constructor == "neg":
        return -eval_reference(fields[0])
    if constructor == "add":
        return eval_reference(fields[0]) + eval_reference(fields[1])
    raise ValueError(f"Unknown constructor: {constructor}")


def eval_buggy(expr_value: Any) -> int:
    """Evaluator under test, with an intentional bug in negation."""
    if not isinstance(expr_value, dict):
        raise ValueError(f"Expected constructor value, got {expr_value!r}")

    constructor = expr_value["constructor"]
    fields = expr_value["fields"]
    if constructor == "lit":
        return int(fields[0])
    if constructor == "neg":
        return eval_buggy(fields[0])
    if constructor == "add":
        return eval_buggy(fields[0]) + eval_buggy(fields[1])
    raise ValueError(f"Unknown constructor: {constructor}")


def format_expr(expr_value: Any) -> str:
    """Format sampled expression values as compact source-like strings."""
    if not isinstance(expr_value, dict):
        return repr(expr_value)

    constructor = expr_value["constructor"]
    fields = expr_value["fields"]
    if constructor == "lit":
        return str(fields[0])
    if constructor == "neg":
        return f"(-{format_expr(fields[0])})"
    if constructor == "add":
        return f"({format_expr(fields[0])} + {format_expr(fields[1])})"
    return repr(expr_value)


def check_samples(result: SamplingResult) -> List[Tuple[str, int, int]]:
    """Run the property check and return failing counterexamples."""
    failures: List[Tuple[str, int, int]] = []
    for sample in result:
        expr_value = sample["root"]
        reference_value = eval_reference(expr_value)
        actual_value = eval_buggy(expr_value)
        if reference_value != actual_value:
            failures.append((format_expr(expr_value), reference_value, actual_value))
    return failures


def print_result(mode: str, result: SamplingResult) -> None:
    """Print samples, compact sampler stats, and property failures."""
    print(f"\n=== {mode} ===")
    print(
        (
            "samples={samples} shapes={shapes} "
            "candidates={candidates} coverage={coverage}"
        ).format(
            samples=len(result),
            shapes=result.stats.get("shape_count"),
            candidates=result.stats.get("candidate_count"),
            coverage=result.stats.get("coverage_ratio", "n/a"),
        )
    )

    for index, sample in enumerate(result, 1):
        expr_text = format_expr(sample["root"])
        reference_value = eval_reference(sample["root"])
        actual_value = eval_buggy(sample["root"])
        status = "PASS" if reference_value == actual_value else "FAIL"
        print(
            f"{index:2d}. {expr_text:14s} "
            f"reference={reference_value:3d} actual={actual_value:3d} {status}"
        )

    failures = check_samples(result)
    if not failures:
        print("No counterexamples found.")
        return

    print("Counterexamples:")
    for expr_text, reference_value, actual_value in failures:
        print(
            f"  {expr_text}: reference evaluator returns {reference_value}, "
            f"buggy evaluator returns {actual_value}"
        )


def main() -> None:
    """Run the property-based testing demo with two selection policies."""
    print("QF_DTLIA property-based testing demo")
    print("Input space: lit(n), neg(lit(n)), add(lit(a), lit(b)) with LIA bounds.")

    for mode in ("coverage_guided", "max_distance"):
        print_result(mode, sample_expressions(mode))


if __name__ == "__main__":
    main()
