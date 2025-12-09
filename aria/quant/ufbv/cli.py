"""
Command-line interface for the UFBV parallel solver.
"""

from __future__ import annotations

import argparse
import os

from .orchestrator import (
    solve_qbv_file_parallel,
    solve_qbv_str_parallel,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Parallel QBV Solver (UFBV)")
    parser.add_argument("--file", type=str, help="SMT2 file to solve")
    parser.add_argument("--formula", type=str, help="SMT2 formula string to solve")
    parser.add_argument("--demo", action="store_true", help="Run demo examples")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    try:
        if args.file:
            print(f"Solving file: {args.file}")
            result = solve_qbv_file_parallel(args.file)
            print(f"Result: {result}")
            return 0
        if args.formula:
            print(f"Solving formula: {args.formula}")
            result = solve_qbv_str_parallel(args.formula)
            print(f"Result: {result}")
            return 0
        if args.demo:
            # Keep a tiny demo to avoid duplication; users can refer to orchestrator.
            demo_fml = """(assert (exists ((x (_ BitVec 4))) (forall ((y (_ BitVec 4))) (= (bvadd x y) (bvadd y x))))) (check-sat)"""
            print("Running demo...")
            print(demo_fml)
            result = solve_qbv_str_parallel(demo_fml)
            print(f"Result: {result}")
            return 0
        parser.print_help()
        return 2
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
