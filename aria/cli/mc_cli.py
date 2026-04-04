"""CLI tool for model counting (Boolean, QF_BV, Arithmetic, etc.)."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, cast

import z3

from aria.counting.qfbv_counting import BVModelCounter
from aria.utils.z3.expr import get_variables
from aria.allsmt.bool_enumeration import count_models as count_bool_models
from aria.sampling.general_sampler import count_solutions


def count_from_file(
    filename: str,
    theory: str = "auto",
    method: str = "auto",  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    timeout: Optional[int] = None,
):
    """Count models from a file.

    Args:
        filename: Path to the formula file
        theory: Theory type (bool, bv, arith, auto)
        method: Counting method (varies by theory)
        timeout: Timeout in seconds
    """
    with open(filename, encoding="utf-8") as f:
        content = f.read()

    # Auto-detect format
    file_ext = Path(filename).suffix.lower()
    if file_ext in (".cnf", ".dimacs"):
        format_type = "dimacs"
        theory = "bool" if theory == "auto" else theory
    elif file_ext == ".smt2":
        format_type = "smtlib2"
    else:
        format_type = "smtlib2"  # default

    auto_formula: Optional[z3.BoolRef] = None
    if theory == "auto" and format_type == "smtlib2":
        try:
            auto_formula = cast(z3.BoolRef, z3.And(*z3.parse_smt2_file(filename)))
            variables = get_variables(auto_formula)
            has_bv = any(z3.is_bv(v) for v in variables)
            has_real = any(z3.is_real(v) for v in variables)
            has_int = any(z3.is_int(v) for v in variables)

            if has_bv:
                theory = "bv"
            elif has_real:
                theory = "generic"
            elif has_int:
                from aria.counting.arith.arith_counting_latte import (
                    ArithModelCounter,
                )  # pylint: disable=import-outside-toplevel

                analysis = ArithModelCounter().analyze(auto_formula)
                if analysis.status == "exact":
                    theory = "arith"
                else:
                    theory = "generic"
            else:
                theory = "bool"
        except Exception:
            theory = "bool"

    if theory == "auto":
        if "BitVec" in content or "(_ bv" in content:
            theory = "bv"
        else:
            theory = "bool"

    logging.info("Detected theory: %s, format: %s", theory, format_type)

    if theory == "bool":
        if format_type == "dimacs":
            from aria.counting.bool.dimacs_counting import (
                count_dimacs_solutions_parallel,
            )  # pylint: disable=import-outside-toplevel

            # Parse DIMACS format
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
            return count
        # Parse SMT-LIB2 and count Boolean models
        try:
            smt_body = "\n".join(
                line
                for line in content.splitlines()
                if not line.lstrip().startswith("(set-logic")
                and not line.lstrip().startswith("(check-sat")
            )
            solver = z3.Solver()
            solver.from_string(smt_body)
            formula = cast(z3.BoolRef, z3.And(*solver.assertions()))
            count = count_bool_models(
                formula, method=method if method != "auto" else "solver"
            )
            return count
        except Exception as e:
            logging.error("Error parsing SMT-LIB2: %s", e)
            raise

    elif theory == "bv":
        # QF_BV model counting
        counter = BVModelCounter()
        counter.init_from_file(filename)
        if method in ("enumeration", "auto"):
            count = counter.count_model_by_bv_enumeration()
        else:
            # Use general sampler
            count = count_solutions(content, fmt="smtlib2", timeout=timeout)
        return count

    elif theory == "arith":
        from aria.counting.arith.arith_counting_latte import (
            ArithModelCounter,
        )  # pylint: disable=import-outside-toplevel

        formula = auto_formula
        if formula is None:
            formula = cast(z3.BoolRef, z3.And(*z3.parse_smt2_file(filename)))

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
            method=arith_method,
        )
        if result.status != "exact" or result.count is None:
            raise ValueError(
                "Arithmetic model counting failed: "
                f"{result.status}: {result.reason}"
            )
        return result.count

    else:
        # Generic SMT-LIB2 counting
        count = count_solutions(content, fmt=format_type, timeout=timeout)
        return count


def main() -> int:
    """Main entry point for model counting CLI."""
    parser = argparse.ArgumentParser(
        description="Count models of formulas (Boolean, QF_BV, Arithmetic, etc.)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("file", type=str, help="Formula file (.smt2, .cnf, .dimacs)")

    parser.add_argument(
        "--theory",
        type=str,
        choices=["bool", "bv", "arith", "auto"],
        default="auto",
        help=(
            "Theory type: bool (Boolean/SAT), bv (bitvector), "
            "arith (arithmetic), auto (detect, default)"
        ),
    )

    parser.add_argument(
        "--method",
        type=str,
        help="Counting method (theory-specific, auto-selected if not specified)",
    )

    parser.add_argument("--timeout", type=int, help="Timeout in seconds")

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    if not Path(args.file).exists():
        print(f"Error: File not found: {args.file}", file=sys.stderr)
        return 1

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        method = args.method if args.method else "auto"
        count = count_from_file(args.file, args.theory, method, args.timeout)
        print(f"Number of models: {count}")
        return 0
    except (ValueError, IOError, OSError, z3.Z3Exception) as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.log_level == "DEBUG":
            import traceback  # pylint: disable=import-outside-toplevel

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
