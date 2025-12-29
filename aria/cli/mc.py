"""CLI tool for model counting (Boolean, QF_BV, Arithmetic, etc.)."""

import argparse
import logging
import sys
from pathlib import Path

import z3

from aria.counting.qfbv_counting import BVModelCounter
from aria.allsmt.bool_enumeration import count_models as count_bool_models
from aria.sampling.general_sampler import count_solutions


def count_from_file(filename: str, theory: str = "auto", method: str = "auto",  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
                    timeout: int = None):
    """Count models from a file.

    Args:
        filename: Path to the formula file
        theory: Theory type (bool, bv, arith, auto)
        method: Counting method (varies by theory)
        timeout: Timeout in seconds
    """
    with open(filename, encoding='utf-8') as f:
        content = f.read()

    # Auto-detect format
    file_ext = Path(filename).suffix.lower()
    if file_ext in ('.cnf', '.dimacs'):
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

    logging.info("Detected theory: %s, format: %s", theory, format_type)

    if theory == 'bool':
        if format_type == 'dimacs':
            from aria.counting.bool.dimacs_counting import count_dimacs_solutions_parallel  # pylint: disable=import-outside-toplevel
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
                logging.error("Error parsing SMT-LIB2: %s", e)
                raise

    elif theory == 'bv':
        # QF_BV model counting
        counter = BVModelCounter()
        counter.init_from_file(filename)
        if method in ('enumeration', 'auto'):
            count = counter.count_model_by_bv_enumeration()
        else:
            # Use general sampler
            count = count_solutions(content, format='smtlib2', timeout=timeout)
        return count

    elif theory == 'arith':
        # Arithmetic model counting
        try:
            from aria.counting.arith.arith_counting_latte import count_lia_models  # pylint: disable=import-outside-toplevel
            formula = z3.And(z3.parse_smt2_file(filename))
            count = count_lia_models(formula)
            return count
        except NotImplementedError:
            logging.warning("LattE-based arithmetic counting not implemented")
            logging.info("Falling back to enumeration-based counting")
            # Fallback to enumeration using AllSMT
            try:
                formula = z3.And(z3.parse_smt2_file(filename))
                from aria.allsmt import create_allsmt_solver  # pylint: disable=import-outside-toplevel
                from aria.utils.z3_expr_utils import get_variables  # pylint: disable=import-outside-toplevel
                # Extract variables from formula
                variables = get_variables(formula)
                # Limit vars for performance
                vars_to_use = variables[:20] if len(variables) > 20 else variables
                # Use AllSMT solver to enumerate models
                solver = create_allsmt_solver()
                models = solver.solve(formula, vars_to_use, model_limit=1000)  # pylint: disable=no-member
                return len(models)
            except Exception as e:
                logging.error("Enumeration-based counting failed: %s", e)
                raise ValueError("Arithmetic model counting is not fully supported yet") from e
        except Exception as e:
            logging.warning("LattE-based counting failed: %s", e)
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
            import traceback  # pylint: disable=import-outside-toplevel
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
