import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import z3

from aria.counting.api import count_from_file, count_result_from_file
from aria.counting.core import CountResult


def _result_to_json(result: CountResult) -> str:
    return json.dumps(asdict(result), sort_keys=True)


def _format_text_result(result: CountResult) -> List[str]:
    lines = []
    if result.count is None:
        lines.append(f"Model counting failed: {result.status}: {result.reason}")
    else:
        count_value = result.count
        if float(count_value).is_integer():
            count_text = str(int(count_value))
        else:
            count_text = str(count_value)
        qualifier = "approximate" if not result.exact else "exact"
        lines.append(f"Number of models: {count_text} ({qualifier})")

    details = [f"backend={result.backend}", f"status={result.status}"]
    if result.runtime_s is not None:
        details.append(f"runtime_s={result.runtime_s:.6f}")
    if result.projection:
        details.append("projection={}".format(",".join(result.projection)))
    if result.error_bound is not None:
        details.append(f"error_bound={result.error_bound}")
    if result.confidence is not None:
        details.append(f"confidence={result.confidence}")
    if result.metadata.get("selection"):
        details.append(f"selection={result.metadata['selection']}")
    lines.append("Details: " + ", ".join(details))

    theory = result.metadata.get("theory")
    method = result.metadata.get("method")
    format_type = result.metadata.get("format")
    support = []
    if theory is not None:
        support.append(f"theory={theory}")
    if method is not None:
        support.append(f"method={method}")
    if format_type is not None:
        support.append(f"format={format_type}")
    if support:
        lines.append("Input: " + ", ".join(support))
    if result.reason and result.count is not None:
        lines.append(f"Reason: {result.reason}")
    return lines


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
    parser.add_argument(
        "--project",
        nargs="+",
        help="Variable names to count over instead of all free variables",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the structured CountResult as JSON",
    )
    parser.add_argument(
        "--fail-on-status",
        nargs="+",
        choices=[
            "unsupported",
            "unbounded",
            "timeout",
            "error",
            "approximate",
        ],
        help="Return a nonzero exit code when the result status matches any listed value",
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

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        method = args.method if args.method else "auto"
        result = count_result_from_file(
            args.file,
            theory=args.theory,
            method=method,
            timeout=args.timeout,
            project=args.project,
        )

        if args.json:
            print(_result_to_json(result))
        else:
            print("\n".join(_format_text_result(result)))
        if args.fail_on_status and result.status in set(args.fail_on_status):
            return 2
        return 0
    except (ValueError, IOError, OSError, z3.Z3Exception) as e:
        if args.json:
            error_result = CountResult(
                status="error",
                count=None,
                backend="cli",
                exact=False,
                reason=str(e),
            )
            print(_result_to_json(error_result), file=sys.stderr)
        else:
            print(f"Error: {e}", file=sys.stderr)
        if args.log_level == "DEBUG":
            import traceback  # pylint: disable=import-outside-toplevel

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
