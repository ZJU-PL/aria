"""CLI tool for model counting (Boolean, QF_BV, Arithmetic, etc.)."""

import argparse
import logging
import sys
from pathlib import Path

import z3

from arlib.counting.qfbv_counting import BVModelCounter
from arlib.allsmt.bool_enumeration import count_models as count_bool_models
from arlib.sampling.general_sampler import count_solutions


def count_from_file(filename: str, theory: str = "auto", method: str = "auto", timeout: int = None):
    """Count models from a file.

    Args:
        filename: Path to the formula file
        theory: Theory type (bool, bv, arith, auto)
        method: Counting method (varies by theory)
        timeout: Timeout in seconds
    """
    with open(filename) as f:
        content = f.read()

    # Auto-detect format
    file_ext = Path(filename).suffix.lower()
    if file_ext == '.cnf' or file_ext == '.dimacs':
        format_type = 'dimacs'
        theory = 'bool' if theory == 'auto' else theory
    elif file_ext == '.smt2':
        format_type = 'smtlib2'
    else:
        format_type = 'smtlib2'  # default

    if theory == 'auto':
        # Try to infer from content
        if 'BitVec' in content or '(_ bv' in content:
            theory = 'bv'
        elif 'Int' in content or 'Real' in content:
            theory = 'arith'
        else:
            theory = 'bool'  # default

    logging.info(f"Detected theory: {theory}, format: {format_type}")

    if theory == 'bool':
        if format_type == 'dimacs':
            from arlib.counting.bool.dimacs_counting import count_dimacs_solutions_parallel
            # Parse DIMACS format
            lines = content.strip().split('\n')
            header = []
            clauses = []
            for line in lines:
                line = line.strip()
                if not line or line.startswith('c'):
                    continue
                if line.startswith('p'):
                    header.append(line)
                else:
                    clauses.append(line.rstrip(' 0').strip())
            count = count_dimacs_solutions_parallel(header, clauses)
            return count
        else:
            # Parse SMT-LIB2 and count Boolean models
            try:
                formula = z3.And(z3.parse_smt2_string(content))
                count = count_bool_models(formula, method=method if method != 'auto' else 'solver')
                return count
            except Exception as e:
                logging.error(f"Error parsing SMT-LIB2: {e}")
                raise

    elif theory == 'bv':
        # QF_BV model counting
        counter = BVModelCounter()
        result = counter.init_from_file(filename)
        if result is None:
            raise ValueError("Failed to initialize BV model counter from file")
        if method == 'enumeration' or method == 'auto':
            count = counter.count_model_by_bv_enumeration()
        else:
            # Use general sampler
            count = count_solutions(content, format='smtlib2', timeout=timeout)
        return count

    elif theory == 'arith':
        # Arithmetic model counting
        try:
            from arlib.counting.arith.arith_counting_latte import count_lia_models
            formula = z3.And(z3.parse_smt2_file(filename))
            count = count_lia_models(formula)
            return count
        except NotImplementedError:
            logging.warning("LattE-based arithmetic counting not implemented")
            logging.info("Falling back to enumeration-based counting")
            # Fallback to enumeration using AllSMT
            try:
                formula = z3.And(z3.parse_smt2_file(filename))
                from arlib.allsmt.allsat import allsat
                # Extract variables from formula
                from arlib.utils.z3_expr_utils import get_variables
                vars = get_variables(formula)
                models = list(allsat(formula, vars[:20] if len(vars) > 20 else vars))  # Limit vars for performance
                return len(models)
            except Exception as e:
                logging.error(f"Enumeration-based counting failed: {e}")
                raise ValueError("Arithmetic model counting is not fully supported yet")
        except Exception as e:
            logging.warning(f"LattE-based counting failed: {e}")
            raise

    else:
        # Generic SMT-LIB2 counting
        count = count_solutions(content, format=format_type, timeout=timeout)
        return count


def main():
    """Main entry point for model counting CLI."""
    parser = argparse.ArgumentParser(
        description="Count models of formulas (Boolean, QF_BV, Arithmetic, etc.)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("file", type=str, help="Formula file (.smt2, .cnf, .dimacs)")

    parser.add_argument(
        "--theory",
        type=str,
        choices=["bool", "bv", "arith", "auto"],
        default="auto",
        help="Theory type: bool (Boolean/SAT), bv (bitvector), arith (arithmetic), auto (detect, default)"
    )

    parser.add_argument(
        "--method",
        type=str,
        help="Counting method (theory-specific, auto-selected if not specified)"
    )

    parser.add_argument(
        "--timeout",
        type=int,
        help="Timeout in seconds"
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

    try:
        method = args.method if args.method else "auto"
        count = count_from_file(args.file, args.theory, method, args.timeout)
        print(f"Number of models: {count}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.log_level == "DEBUG":
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
