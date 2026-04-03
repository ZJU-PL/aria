"""CLI tool for UNSAT core, MUS, and MSS computation."""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import z3

from aria.unsat_core.unsat_core import (
    UnsatCoreResult,
    get_unsat_core,
    enumerate_all_mus,
)


def _constraints_from_smt2(filename: str) -> List[z3.ExprRef]:
    """Load assertions from an SMT-LIB2 file as a list of Z3 expressions."""
    solver = z3.Solver()
    solver.from_file(filename)
    return list(solver.assertions())


def _format_cores(result: UnsatCoreResult, constraints: List[z3.ExprRef]) -> str:
    """Format core indices and optional constraint snippets for output."""
    lines: List[str] = []
    for i, core in enumerate(result.cores):
        indices = sorted(core)
        lines.append(f"core {i + 1}: indices {indices}")
        for idx in indices:
            if 0 <= idx < len(constraints):
                lines.append(f"  [{idx}] {constraints[idx].sexpr()}")
    return "\n".join(lines)


def run_unsat_core(
    constraints: List[z3.ExprRef],
    algorithm: str = "marco",
    all_mus: bool = False,
    timeout: Optional[int] = None,
) -> UnsatCoreResult:
    """Compute one UNSAT core or enumerate all MUSes.

    Args:
        constraints: List of Z3 assertions.
        algorithm: Algorithm: marco, musx, optux.
        all_mus: If True, enumerate all MUSes (only with marco).
        timeout: Timeout in seconds.

    Returns:
        UnsatCoreResult with cores (each core is a set of assertion indices).
    """
    if not constraints:
        raise ValueError("No assertions provided")

    def solver_factory():
        return z3.Solver()

    if all_mus:
        try:
            return enumerate_all_mus(constraints, solver_factory, timeout=timeout)
        except AttributeError as e:
            raise ValueError(
                "Enumerate all MUS not supported by this algorithm backend"
            ) from e
    return get_unsat_core(
        constraints,
        solver_factory,
        algorithm=algorithm,
        timeout=timeout,
    )


def main() -> int:
    """Main entry point for UNSAT core CLI."""
    parser = argparse.ArgumentParser(
        description="Compute UNSAT core(s) or enumerate MUSes from SMT-LIB2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "file",
        type=str,
        help="SMT-LIB2 formula file (.smt2)",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["marco", "musx", "optux"],
        default="marco",
        help="Algorithm for core extraction (default: marco)",
    )
    parser.add_argument(
        "--all-mus",
        action="store_true",
        help="Enumerate all minimal unsatisfiable subsets (MARCO only)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout in seconds",
    )
    parser.add_argument(
        "--no-formulas",
        action="store_true",
        help="Print only core indices, not constraint formulas",
    )
    args = parser.parse_args()

    if not Path(args.file).exists():
        print(f"Error: File not found: {args.file}", file=sys.stderr)
        return 1

    try:
        constraints = _constraints_from_smt2(args.file)
    except z3.Z3Exception as e:
        print(f"Error parsing file: {e}", file=sys.stderr)
        return 1

    if not constraints:
        print("Error: No assertions in file.", file=sys.stderr)
        return 1

    solver = z3.Solver()
    for c in constraints:
        solver.add(c)
    if solver.check() == z3.sat:
        print("Formula is satisfiable; no UNSAT core.")
        return 0

    try:
        result = run_unsat_core(
            constraints,
            algorithm=args.algorithm,
            all_mus=args.all_mus,
            timeout=args.timeout,
        )
    except (ValueError, NotImplementedError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    if args.no_formulas:
        for i, core in enumerate(result.cores):
            print(f"core {i + 1}: {' '.join(str(x) for x in sorted(core))}")
    else:
        print(_format_cores(result, constraints))
    return 0


if __name__ == "__main__":
    sys.exit(main())
