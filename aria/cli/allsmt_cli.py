"""CLI tool for AllSMT: enumerate all satisfying models of an SMT formula."""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import z3

from aria.allsmt import create_allsmt_solver
from aria.utils.z3_expr_utils import get_variables


def _formula_and_vars_from_smt2(
    filename: str,
) -> Tuple[z3.ExprRef, List[z3.ExprRef]]:
    """Load formula and free variables from an SMT-LIB2 file."""
    solver = z3.Solver()
    solver.from_file(filename)
    assertions = list(solver.assertions())
    if not assertions:
        raise ValueError("No assertions in file")
    formula = z3.And(assertions)
    variables = get_variables(formula)
    return formula, variables


def enumerate_models(
    filename: str,
    solver_name: str = "z3",
    model_limit: int = 100,
    project_vars: Optional[List[str]] = None,
) -> int:
    """Enumerate satisfying models for the formula in the given file.

    Args:
        filename: Path to SMT-LIB2 file.
        solver_name: AllSMT backend: z3, pysmt, mathsat.
        model_limit: Maximum number of models to enumerate.
        project_vars: If set, only include these variable names in models.

    Returns:
        Number of models found.
    """
    formula, variables = _formula_and_vars_from_smt2(filename)
    if project_vars is not None:
        name_set = set(project_vars)
        variables = [v for v in variables if v.decl().name() in name_set]
    if not variables:
        raise ValueError(
            "No variables to enumerate; use (declare-const ...) in the formula."
        )

    backend = create_allsmt_solver(solver_name)
    backend.solve(formula, variables, model_limit=model_limit)
    return backend.get_model_count()


def main() -> int:
    """Main entry point for AllSMT CLI."""
    parser = argparse.ArgumentParser(
        description="Enumerate all satisfying models of an SMT-LIB2 formula",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "file",
        type=str,
        help="SMT-LIB2 formula file (.smt2)",
    )
    parser.add_argument(
        "--solver",
        type=str,
        choices=["z3", "pysmt", "mathsat"],
        default="z3",
        help="AllSMT backend (default: z3)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        metavar="N",
        help="Maximum number of models to enumerate (default: 100)",
    )
    parser.add_argument(
        "--project",
        type=str,
        metavar="VAR1,VAR2,...",
        default=None,
        help="Comma-separated variable names to include in models (default: all)",
    )
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Print only the model count",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each model in detail",
    )
    args = parser.parse_args()

    if not Path(args.file).exists():
        print(f"Error: File not found: {args.file}", file=sys.stderr)
        return 1

    try:
        formula, variables = _formula_and_vars_from_smt2(args.file)
    except (ValueError, z3.Z3Exception) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    project_vars = None
    if args.project:
        project_vars = [s.strip() for s in args.project.split(",")]

    try:
        if project_vars is not None:
            name_set = set(project_vars)
            variables = [v for v in variables if v.decl().name() in name_set]
        if not variables:
            print("Error: No variables to enumerate.", file=sys.stderr)
            return 1

        backend = create_allsmt_solver(args.solver)
        backend.solve(formula, variables, model_limit=args.limit)
        count = backend.get_model_count()
    except (ValueError, ImportError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if args.count_only:
        print(count)
    else:
        backend.print_models(verbose=args.verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
