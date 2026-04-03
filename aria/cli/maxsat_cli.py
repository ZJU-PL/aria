"""CLI tool for MaxSAT (weighted partial MaxSAT)."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from pysat.formula import WCNF

from aria.bool.maxsat.maxsat_solver import MaxSATSolver, MaxSATSolverResult


def solve_maxsat_from_file(
    filename: str,
    solver: str = "rc2",
    timeout: Optional[int] = None,
) -> MaxSATSolverResult:
    """Solve MaxSAT from a WCNF file.

    Args:
        filename: Path to WCNF file.
        solver: Engine name: rc2, fm, lsu.
        timeout: Timeout in seconds (used only by some engines).

    Returns:
        MaxSATSolverResult with cost, solution, status.
    """
    wcnf = WCNF(from_file=filename)
    engine = solver.upper()

    if engine == "LSU":
        from aria.bool.maxsat.lsu import LSU  # pylint: disable=import-outside-toplevel

        lsu = LSU(wcnf, verbose=0)
        ok = lsu.solve()
        return MaxSATSolverResult(
            cost=getattr(lsu, "cost", float("inf")),
            solution=getattr(lsu, "model", None),
            status="optimal" if ok and getattr(lsu, "cost", None) is not None else "unknown",
        )

    msolver = MaxSATSolver(wcnf)
    if engine == "RC2":
        msolver.set_maxsat_engine("RC2")
    elif engine == "FM":
        msolver.set_maxsat_engine("FM")
    else:
        logging.warning("Unknown solver %s, using RC2", solver)
        msolver.set_maxsat_engine("RC2")
    return msolver.solve()


def main() -> int:
    """Main entry point for MaxSAT CLI."""
    parser = argparse.ArgumentParser(
        description="Solve MaxSAT problems (WCNF format)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "file",
        type=str,
        help="WCNF formula file (.wcnf, .cnf)",
    )
    parser.add_argument(
        "--solver",
        type=str,
        choices=["rc2", "fm", "lsu"],
        default="rc2",
        help="MaxSAT engine (default: rc2)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout in seconds (for engines that support it)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: WARNING)",
    )
    parser.add_argument(
        "--print-model",
        action="store_true",
        help="Print satisfying assignment (literal list)",
    )
    args = parser.parse_args()

    if not Path(args.file).exists():
        print(f"Error: File not found: {args.file}", file=sys.stderr)
        return 1

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(message)s",
    )

    try:
        result = solve_maxsat_from_file(args.file, args.solver, args.timeout)
        print(f"cost: {result.cost}")
        if result.status:
            print(f"status: {result.status}")
        if args.print_model and result.solution is not None:
            print("model:", " ".join(str(lit) for lit in result.solution))
        return 0
    except (ValueError, OSError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
