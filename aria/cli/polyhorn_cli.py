"""Command-line interface for PolyHorn.
This module provides the main CLI interface for running PolyHorn on transition
systems specified in various formats.
"""

import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path

from aria.efmc.engines.polyhorn.main import execute, add_default_config

logger = logging.getLogger(__name__)


class PolyHornRunner:  # pylint: disable=too-few-public-methods
    """Main runner class for PolyHorn solving tasks"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def solve_file(self, file_path: str, config: dict, output_file: str = None):
        """Solve a PolyHorn problem from a file"""
        try:
            start_time = time.time()

            # Execute PolyHorn
            result, model = execute(file_path, config)

            # Print results
            print(f"Result: {result}")
            if model:
                print(f"Model: {model}")
            print(f"Time: {time.time() - start_time:.2f}s")

            # Save output if requested
            if output_file:
                output_data = {
                    "result": result,
                    "model": model,
                    "time": time.time() - start_time,
                }
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(output_data, f, indent=2)
                self.logger.info("Results saved to %s", output_file)

            return None

        except (ValueError, FileNotFoundError, OSError) as e:
            self.logger.error("Error solving PolyHorn: %s", str(e))
            return str(e)
        except Exception as e:
            self.logger.error("Unexpected error solving PolyHorn: %s", str(e))
            return str(e)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="PolyHorn - Polynomial Constraint Solver",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Input options
    parser.add_argument("file", type=str, help="Input SMT2 file to solve")
    parser.add_argument("--config", type=str, help="Configuration file (JSON format)")

    # Theorem selection
    parser.add_argument(
        "--theorem",
        type=str,
        default="auto",
        choices=["auto", "farkas", "handelman", "putinar"],
        help="Theorem to use (default: auto)",
    )

    # Solver options
    parser.add_argument(
        "--solver",
        type=str,
        default="z3",
        choices=["z3", "cvc4", "cvc5", "yices2", "mathsat"],
        help="SMT solver to use (default: z3)",
    )

    # Degree control
    parser.add_argument(
        "--degree-sat",
        type=int,
        default=0,
        help="Maximum degree for satisfiability constraints (default: 0)",
    )
    parser.add_argument(
        "--degree-unsat",
        type=int,
        default=0,
        help="Maximum degree for unsatisfiability constraints (default: 0)",
    )
    parser.add_argument(
        "--degree-strict",
        type=int,
        default=0,
        help="Maximum degree for strict unsatisfiability constraints (default: 0)",
    )

    # Advanced options
    parser.add_argument(
        "--integer",
        action="store_true",
        default=False,
        help="Use integer arithmetic instead of real arithmetic",
    )
    parser.add_argument(
        "--sat-heuristic",
        action="store_true",
        default=False,
        help="Enable SAT heuristics",
    )
    parser.add_argument(
        "--unsat-core",
        action="store_true",
        default=False,
        help="Enable unsat core heuristics",
    )

    # Output options
    parser.add_argument(
        "--output", type=str, help="Output file for results (JSON format)"
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False, help="Enable verbose logging"
    )

    return parser.parse_args()


def create_config_from_args(args):
    """Create configuration dictionary from command line arguments"""
    config = {
        "theorem_name": args.theorem,
        "solver_name": args.solver,
        "degree_of_sat": args.degree_sat,
        "degree_of_nonstrict_unsat": args.degree_unsat,
        "degree_of_strict_unsat": args.degree_strict,
        "max_d_of_strict": args.degree_strict,
        "integer_arithmetic": args.integer,
        "SAT_heuristic": args.sat_heuristic,
        "unsat_core_heuristic": args.unsat_core,
    }

    return add_default_config(config)


def signal_handler(_sig, _frame):
    """Handle Ctrl+C gracefully"""
    print("\nTerminating...")
    sys.exit(0)


def main():
    """Main entry point"""
    signal.signal(signal.SIGINT, signal_handler)

    args = parse_arguments()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Validate input file
    if not Path(args.file).exists():
        print(f"Error: Input file '{args.file}' does not exist")
        sys.exit(1)

    # Load or create configuration
    if args.config:
        if not Path(args.config).exists():
            print(f"Error: Config file '{args.config}' does not exist")
            sys.exit(1)

        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
        config = add_default_config(config)
    else:
        config = create_config_from_args(args)

    # Run PolyHorn
    runner = PolyHornRunner()
    runner.solve_file(file_path=args.file, config=config, output_file=args.output)


if __name__ == "__main__":
    sys.exit(main())
