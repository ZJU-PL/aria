"""CLI tool for optimization problems (OMT, MaxSMT, etc.)."""

import argparse
import logging
import sys
from pathlib import Path

import z3

from aria.optimization.omt_solver import solve_opt_file
from aria.optimization.omt_parser import OMTParser


def solve_omt_problem(filename: str, engine: str, solver_name: str, theory: str = None):
    """Solve OMT problem (supports both BV and arithmetic)."""
    parser = OMTParser()
    parser.parse_with_z3(filename, is_file=True)

    fml = z3.And(parser.assertions)
    obj = parser.objective

    # Auto-detect theory if not specified
    if theory is None:
        # Check if objective is bitvector or arithmetic
        obj_sort = obj.sort()
        sort_kind = obj_sort.kind()
        if sort_kind == z3.Z3_BV_SORT:
            theory = "bv"
        elif sort_kind in (z3.Z3_INT_SORT, z3.Z3_REAL_SORT):
            theory = "arith"
        else:
            # Try to infer from formula - check for BV operations in string representation
            obj_str = str(obj)
            if 'BitVec' in obj_str or 'BV' in obj_str:
                theory = "bv"
            else:
                theory = "bv"  # default to BV for OMT solver compatibility

    if theory == "arith":
        # Use arithmetic optimization
        from aria.optimization.omtarith.arith_opt_qsmt import arith_opt_with_qsmt
        from aria.optimization.omtarith.arith_opt_ls import arith_opt_with_ls

        if engine == "qsmt":
            result = arith_opt_with_qsmt(fml, obj, minimize=False, solver_name=solver_name)
            logging.info("Arithmetic QSMT result: %s", result)
        elif engine == "iter":
            result = arith_opt_with_ls(fml, obj, minimize=False, solver_name=solver_name)
            logging.info("Arithmetic iterative search result: %s", result)
        else:
            # Fall back to general OMT solver
            solve_opt_file(filename, engine, solver_name)
    else:
        # Use general OMT solver (handles BV by default)
        solve_opt_file(filename, engine, solver_name)


def main():
    """Main entry point for optimization CLI."""
    parser = argparse.ArgumentParser(
        description="Solve optimization problems (OMT, MaxSMT, etc.)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("file", type=str, help="Optimization problem file (.smt2)")

    parser.add_argument(
        "--type",
        type=str,
        choices=["omt", "maxsmt"],
        default="omt",
        help="Problem type: omt (Optimization Modulo Theory) or maxsmt (default: omt)"
    )

    parser.add_argument(
        "--theory",
        type=str,
        choices=["bv", "arith", "auto"],
        default="auto",
        help="Theory type for OMT: bv (bitvector), arith (arithmetic), auto (detect, default)"
    )

    parser.add_argument(
        "--engine",
        type=str,
        default="qsmt",
        choices=["qsmt", "maxsat", "iter", "z3py"],
        help="Optimization engine (default: qsmt)"
    )

    parser.add_argument(
        "--solver",
        type=str,
        help="Solver name (engine-specific, auto-selected if not specified)"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )

    args = parser.parse_args()

    if not Path(args.file).exists():
        print(f"Error: File not found: {args.file}", file=sys.stderr)
        return 1

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Set default solver based on engine
    solver = args.solver
    if not solver:
        solver_map = {
            "qsmt": "z3",
            "maxsat": "FM",
            "iter": "z3-ls",
            "z3py": "z3py"
        }
        solver = solver_map.get(args.engine, "z3")

    try:
        if args.type == "maxsmt":
            # MaxSMT requires different handling - would need to parse hard/soft constraints
            print("Error: MaxSMT support not yet implemented in CLI", file=sys.stderr)
            print("Note: MaxSMT problems need hard/soft constraint specification", file=sys.stderr)
            return 1
        # OMT problem
        theory = None if args.theory == "auto" else args.theory
        solve_omt_problem(args.file, args.engine, solver, theory)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.log_level == "DEBUG":
            import traceback  # pylint: disable=import-outside-toplevel
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
