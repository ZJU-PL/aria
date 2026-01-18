#!/usr/bin/env python3
"""CLI for E-matching trigger selection and SMT2 annotation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import z3

from aria.ml.ematching import LLMTriggerGenerator, TriggerSelector


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Annotate SMT-LIB2 formulas with E-matching triggers.",
    )
    parser.add_argument("input", help="Input SMT2 file.")
    parser.add_argument(
        "--output",
        "-o",
        help="Output SMT2 file (defaults to stdout).",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Use LLM-guided trigger selection (requires LLM deps/config).",
    )
    parser.add_argument(
        "--llm-direct",
        action="store_true",
        help="Let the LLM synthesize SMT-LIB trigger terms directly.",
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-4o-mini",
        help="LLM model name (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=0.1,
        help="LLM sampling temperature (default: 0.1).",
    )
    parser.add_argument(
        "--max-groups",
        type=int,
        default=3,
        help="Max trigger groups per quantifier (default: 3).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser


def _load_formula(path: Path) -> z3.ExprRef:
    assertions = z3.parse_smt2_file(str(path))
    if not assertions:
        return z3.BoolVal(True)
    return z3.And(assertions)


def _emit_smt2(expr: z3.ExprRef) -> str:
    solver = z3.Solver()
    solver.add(expr)
    return solver.to_smt2()


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 2

    formula = _load_formula(input_path)

    llm_generator = None
    if args.llm or args.llm_direct:
        llm_generator = LLMTriggerGenerator(
            model=args.llm_model,
            temperature=args.llm_temperature,
            verbose=args.verbose,
            max_groups=args.max_groups,
            direct_terms=args.llm_direct,
        )
        if llm_generator.llm is None:
            print(
                "LLM backend is unavailable. Install dependencies and set API keys.",
                file=sys.stderr,
            )
            return 2

    selector = TriggerSelector(
        formula,
        llm_generator=llm_generator,
        max_groups=args.max_groups,
        verbose=args.verbose,
    )
    annotated = selector.annotate_with_triggers()
    output_text = _emit_smt2(annotated)

    if args.output:
        Path(args.output).write_text(output_text, encoding="utf-8")
    else:
        print(output_text)

    return 0


if __name__ == "__main__":
    sys.exit(main())
