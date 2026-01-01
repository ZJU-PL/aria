"""
CLI tool for counting models of Boolean/SMT formulas
"""

import argparse
import sys
from typing import Optional, Sequence

from aria.sampling.general_sampler import count_solutions


def create_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser.

    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(description="Count models of Boolean/SMT formulas")
    parser.add_argument("input_file", help="Input formula file")
    parser.add_argument(
        "--format", choices=["smtlib2", "dimacs"], required=True, help="Input format"
    )
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Main entry point for the model counting CLI.

    Args:
        argv: Optional command line arguments (defaults to sys.argv)

    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    try:
        with open(args.input_file, encoding="utf-8") as f:
            formula = f.read()

        if args.format == "smtlib2":
            count = count_solutions(formula, timeout=args.timeout)
        else:
            # Handle DIMACS format
            count = count_solutions(formula, format="dimacs", timeout=args.timeout)

        print(f"Number of models: {count}")
        return 0

    except (IOError, OSError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
