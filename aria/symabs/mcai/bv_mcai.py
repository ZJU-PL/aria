"""
Model Counting Meets Abstract Interpretation

This module provides functionality to:
1. Count models using SharpSAT
2. Compute and analyze different abstract domains (Interval, Zone, Octagon)
3. Calculate false positive rates for each abstraction

Key Components:
- ModelCounter: Handles model counting operations
- AbstractionAnalyzer: Performs analysis across different abstract domains
- AbstractionResults: Stores false positive rates for each domain

Dependencies:
- z3: SMT solver for constraint solving
- aria.smt.bv: Bit-vector model counting
- aria.symabs: Symbolic abstraction implementations
"""

import argparse
import logging
import multiprocessing as mp
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Union

import z3
from z3 import parse_smt2_file, ExprRef, BitVecRef

from aria.counting.qfbv_counting import BVModelCounter
from aria.symabs.omt_symabs.bv_symbolic_abstraction import BVSymbolicAbstraction
from aria.tests.formula_generator import FormulaGenerator
from aria.utils.z3_expr_utils import get_variables

# Initialize module-level logger
logger = logging.getLogger(__name__)

# from ..utils.plot_util import ScatterPlot  # See aria/scripts


@dataclass
class AbstractionResults:
    """Store results from different abstraction domains"""

    interval_fp_rate: float = 0.0
    interval_time: float = 0.0
    interval_abs_count: int = 0
    interval_fp_count: int = 0
    zone_fp_rate: float = 0.0
    zone_time: float = 0.0
    zone_abs_count: int = 0
    zone_fp_count: int = 0
    octagon_fp_rate: float = 0.0
    octagon_time: float = 0.0
    octagon_abs_count: int = 0
    octagon_fp_count: int = 0
    bitwise_fp_rate: float = 0.0
    bitwise_time: float = 0.0
    bitwise_abs_count: int = 0
    bitwise_fp_count: int = 0
    i_z_count: int = 0
    i_o_count: int = 0
    i_b_count: int = 0

    def __add__(self, other: Optional["AbstractionResults"]) -> "AbstractionResults":
        if other is None:
            return self
        return AbstractionResults(
            self.interval_fp_rate + other.interval_fp_rate,
            self.interval_time + other.interval_time,
            self.zone_fp_rate + other.zone_fp_rate,
            self.zone_time + other.zone_time,
            self.octagon_fp_rate + other.octagon_fp_rate,
            self.octagon_time + other.octagon_time,
            self.bitwise_fp_rate + other.bitwise_fp_rate,
            self.bitwise_time + other.bitwise_time,
        )

    def __truediv__(self, other: Union[int, float]) -> "AbstractionResults":
        return AbstractionResults(
            self.interval_fp_rate / other,
            self.interval_time / other,
            self.zone_fp_rate / other,
            self.zone_time / other,
            self.octagon_fp_rate / other,
            self.octagon_time / other,
            self.bitwise_fp_rate / other,
            self.bitwise_time / other,
        )

    def __str__(self) -> str:
        return (
            f"Interval FP rate: {self.interval_fp_rate:.4f}, "
            f"Zone FP rate: {self.zone_fp_rate:.4f}, "
            f"Octagon FP rate: {self.octagon_fp_rate:.4f}, "
            f"Bitwise FP rate: {self.bitwise_fp_rate:.4f}, "
            f"Interval time: {self.interval_time:.4f}, "
            f"Zone time: {self.zone_time:.4f}, "
            f"Octagon time: {self.octagon_time:.4f}, "
            f"Bitwise time: {self.bitwise_time:.4f}"
        )


class ModelCounter:
    """Handles model counting operations"""

    def __init__(self, timeout_ms: int = 6000) -> None:
        self.timeout_ms: int = timeout_ms

    def is_sat(self, expression: ExprRef) -> bool:
        """Check if the expression is satisfiable"""
        solver = z3.Solver()
        solver.set("timeout", self.timeout_ms)
        solver.add(expression)
        return solver.check() == z3.sat

    def count_models(self, formula: ExprRef) -> Tuple[int, float]:
        """Count models using sharpSAT"""
        counter = BVModelCounter()
        counter.init_from_fml(formula)
        return counter.count_models_by_sharp_sat()


class AbstractionAnalyzer:
    """Analyzes different abstraction domains"""

    def __init__(self, formula: ExprRef, variables: List[BitVecRef]) -> None:
        self.formula: ExprRef = z3.And(formula)
        self.variables: List[BitVecRef] = variables
        self.sa: BVSymbolicAbstraction = BVSymbolicAbstraction()
        self.sa.init_from_fml(formula)
        self.sa.do_simplification()
        self.formula = self.sa.formula

        counter = ModelCounter()
        if not counter.is_sat(self.formula):
            logger.info("Formula is unsatisfiable")
            sys.exit(1)
        # Count models
        model_count = counter.count_models(self.formula)
        logger.info("SharpSAT model count: %s", model_count)
        if model_count[0] == -1:
            logger.info("model count failed")
            sys.exit(1)

    def compute_false_positives(
        self, abs_formula: ExprRef
    ) -> Tuple[bool, float, float]:
        """Compute false positive rate for an abstraction"""
        solver = z3.Solver()
        solver.add(abs_formula)
        if solver.check() == z3.unsat:
            return True, -1.0, -1.0, 0, 0
        solver = z3.Solver()
        solver.add(z3.And(abs_formula, z3.Not(self.formula)))

        has_false_positives = solver.check() != z3.unsat
        if not has_false_positives:
            return False, 0.0, 0.0, 0, 0

        # Count models for abstraction and false positives
        mc = BVModelCounter()
        mc.init_from_fml(abs_formula)
        abs_count, abs_time = mc.count_models_by_sharp_sat()

        mc_fp = BVModelCounter()
        mc_fp.init_from_fml(z3.And(abs_formula, z3.Not(self.formula)))
        fp_count, fp_time = mc_fp.count_models_by_sharp_sat()
        if abs_count < 0 or fp_count < 0:
            return True, -1.0, -1.0, 0, 0
        logger.info("fp_count=%s, abs_count=%s", fp_count, abs_count)
        return True, fp_count / abs_count, abs_time + fp_time, abs_count, fp_count

    def analyze_abstractions(self) -> Optional[AbstractionResults]:
        """Analyze all abstraction domains"""
        # Perform abstractions
        try:
            self.sa.interval_abs()
        except Exception as exc:
            exc_info = sys.exc_info()
            line_no = exc_info[-1].tb_lineno if exc_info[-1] else "unknown"
            logger.error("Error analyzing abstractions: %s, line %s", str(exc), line_no)
            self.sa.interval_abs_as_fml = z3.BoolVal(False)

        try:
            self.sa.zone_abs()
        except Exception as exc:
            exc_info = sys.exc_info()
            line_no = exc_info[-1].tb_lineno if exc_info[-1] else "unknown"
            logger.error("Error analyzing abstractions: %s, line %s", str(exc), line_no)
            self.sa.zone_abs_as_fml = z3.BoolVal(False)

        try:
            self.sa.octagon_abs()
        except Exception as exc:
            exc_info = sys.exc_info()
            line_no = exc_info[-1].tb_lineno if exc_info[-1] else "unknown"
            logger.error("Error analyzing abstractions: %s, line %s", str(exc), line_no)
            self.sa.octagon_abs_as_fml = z3.BoolVal(False)

        try:
            self.sa.bitwise_abs()
        except Exception as exc:
            exc_info = sys.exc_info()
            line_no = exc_info[-1].tb_lineno if exc_info[-1] else "unknown"
            logger.error("Error analyzing abstractions: %s, line %s", str(exc), line_no)
            self.sa.bitwise_abs_as_fml = z3.BoolVal(False)

        try:
            results = AbstractionResults()

            # Analyze each domain
            for domain, formula in [
                ("Interval", self.sa.interval_abs_as_fml),
                ("Zone", self.sa.zone_abs_as_fml),
                ("Octagon", self.sa.octagon_abs_as_fml),
                ("Bitwise", self.sa.bitwise_abs_as_fml),
            ]:
                logger.info("%s:\n%s", domain, formula)
                if formula == z3.BoolVal(False):
                    logger.warning("Skipping %s domain", domain)
                    continue
                has_fp, fp_rate, time_fp, abs_count, fp_count = (
                    self.compute_false_positives(formula)
                )
                if fp_rate >= 0:
                    msg = (
                        f"{domain} domain: has FP rate {fp_rate:.4f}"
                        if has_fp
                        else f"{domain} domain: no false positives"
                    )
                    logger.info(msg)

                if domain == "Interval":
                    results.interval_fp_rate = fp_rate
                    results.interval_time = time_fp
                    results.interval_abs_count = abs_count
                    results.interval_fp_count = fp_count
                elif domain == "Zone":
                    results.zone_fp_rate = fp_rate
                    results.zone_time = time_fp
                    results.zone_abs_count = abs_count
                    results.zone_fp_count = fp_count
                elif domain == "Octagon":
                    results.octagon_fp_rate = fp_rate
                    results.octagon_time = time_fp
                    results.octagon_abs_count = abs_count
                    results.octagon_fp_count = fp_count
                elif domain == "Bitwise":
                    results.bitwise_fp_rate = fp_rate
                    results.bitwise_time = time_fp
                    results.bitwise_abs_count = abs_count
                    results.bitwise_fp_count = fp_count

            for d1, f1, d2, f2 in [
                (
                    "Interval",
                    self.sa.interval_abs_as_fml,
                    "Zone",
                    self.sa.zone_abs_as_fml,
                ),
                (
                    "Interval",
                    self.sa.interval_abs_as_fml,
                    "Octagon",
                    self.sa.octagon_abs_as_fml,
                ),
                (
                    "Interval",
                    self.sa.interval_abs_as_fml,
                    "Bitwise",
                    self.sa.bitwise_abs_as_fml,
                ),
            ]:
                mc = BVModelCounter()
                mc.init_from_fml(z3.And(f1, f2))
                count, _ = mc.count_models_by_sharp_sat()
                if d2 == "Zone":
                    results.i_z_count = count
                elif d2 == "Octagon":
                    results.i_o_count = count
                elif d2 == "Bitwise":
                    results.i_b_count = count
                else:
                    logger.info("%s & %s : %s", d1, d2, count)

            return results

        except Exception as exc:
            exc_info = sys.exc_info()
            line_no = exc_info[-1].tb_lineno if exc_info[-1] else "unknown"
            logger.error("Error analyzing abstractions: %s, line %s", str(exc), line_no)
            return None


def setup_logging(log_file: Optional[str] = None):
    """Configure logging to both file and console"""
    log_format = "%(asctime)s - %(levelname)s - %(message)s"

    if log_file:
        if not os.path.exists(os.path.dirname(log_file)):
            os.makedirs(os.path.dirname(log_file))
        logging.basicConfig(
            level=logging.DEBUG,
            format=log_format,
            handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
        )
    else:
        logging.basicConfig(level=logging.INFO, format=log_format)
    return logging.getLogger(__name__)


def process_smt_file(file_path: str, args) -> Optional[AbstractionResults]:
    """Process a single SMT-LIB2 file"""
    try:
        # Parse SMT-LIB2 file
        formula = z3.And(parse_smt2_file(file_path))

        # Extract variables from formula
        variables = get_variables(formula)

        if not variables:
            logger.warning("No bit-vector variables found in %s", file_path)
            return None
        counter = ModelCounter()
        if not counter.is_sat(formula):
            logger.info("%s: Formula is unsatisfiable", file_path)
            return None

        if not counter.is_sat(z3.Not(formula)):
            logger.info("%s: Formula is always satisfiable", file_path)
            return None

        # Count models
        model_count = counter.count_models(formula)
        logger.info("%s: SharpSAT model count: %s", file_path, model_count)

        # Analyze abstractions
        analyzer = AbstractionAnalyzer(formula, variables)
        results = analyzer.analyze_abstractions()

        if results:
            logger.info("%s: Analysis completed successfully", file_path)
            if args.file:
                if not os.path.exists(args.csv):
                    header = (
                        "filename,interval_fp_rate,zone_fp_rate,octagon_fp_rate,"
                        "bitwise_fp_rate,interval_time,zone_time,octagon_time,"
                        "bitwise_time,interval_abs_time,zone_abs_time,"
                        "octagon_abs_time,bitwise_abs_time,model_count,"
                        "i_z_count,i_o_count,i_b_count\n"
                    )
                    with open(args.csv, "w", encoding="utf-8") as f:
                        f.write(header)
                with open(args.csv, "a", encoding="utf-8") as csv:
                    csv.write(
                        f"{file_path},{results.interval_fp_rate},"
                        f"{results.zone_fp_rate},{results.octagon_fp_rate},"
                        f"{results.bitwise_fp_rate},{results.interval_time},"
                        f"{results.zone_time},{results.octagon_time},"
                        f"{results.bitwise_time},{analyzer.sa.interval_abs_time},"
                        f"{analyzer.sa.zone_abs_time},"
                        f"{analyzer.sa.octagon_abs_time},"
                        f"{analyzer.sa.bitwise_abs_time},{model_count[0]},"
                        f"{results.i_z_count},{results.i_o_count},"
                        f"{results.i_b_count}\n"
                    )
            return results

        logger.debug("Analysis failed: %s", file_path)
        return None

    except Exception as exc:
        logger.error("Error processing %s: %s", file_path, str(exc))
        return None


def process_directory(dir_path: str, args) -> None:
    """Process all SMT-LIB2 files in directory using parallel processing"""
    smt_files = [str(f) for f in Path(dir_path).glob("**/*.smt2")]

    if not smt_files:
        logger.warning("No SMT-LIB2 files found in %s", dir_path)
        return

    logger.info("Found %d SMT-LIB2 files to process", len(smt_files))

    # NOTE: Parallel processing disabled due to:
    # AssertionError: daemonic processes are not allowed to have children
    # with mp.Pool(processes=num_processes) as pool:
    #     results = pool.map(process_smt_file, smt_files)

    results = []
    for file in smt_files:
        results.append((file, process_smt_file(file, args)))

    successful = sum(1 for f, r in results if r is not None)
    non_none_results = [r for f, r in results if r is not None]
    final_results = (
        sum(non_none_results, start=AbstractionResults()) / successful
        if successful > 0
        else AbstractionResults()
    )
    logger.info("Successfully processed %d/%d files", successful, len(smt_files))
    logger.info("Final results: %s", final_results)
    parent_dir = os.path.dirname(dir_path)
    if not os.path.exists(f"{parent_dir}/results.csv"):
        header = (
            "filename,interval_fp_rate,zone_fp_rate,octagon_fp_rate,"
            "bitwise_fp_rate,interval_time,zone_time,octagon_time,"
            "bitwise_time,interval_abs_time,zone_abs_time,"
            "octagon_abs_time,bitwise_abs_time,model_count,"
            "i_z_count,i_o_count,i_b_count\n"
        )
        with open(args.csv, "w", encoding="utf-8") as f:
            f.write(header)
    with open(f"{parent_dir}/results.csv", "a", encoding="utf-8") as csv:
        for f, r in results:
            if r is not None:
                csv.write(
                    f"{f},{r.interval_fp_rate},{r.zone_fp_rate},"
                    f"{r.octagon_fp_rate},{r.bitwise_fp_rate},"
                    f"{r.interval_time},{r.zone_time},{r.octagon_time},"
                    f"{r.bitwise_time}\n"
                )


def main():
    """Main entry point with command line argument handling"""
    parser = argparse.ArgumentParser(
        description="Model counting and abstract interpretation analysis"
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-f", "--file", help="Path to SMT-LIB2 file to analyze")
    input_group.add_argument(
        "-d", "--directory", help="Path to directory containing SMT-LIB2 files"
    )
    input_group.add_argument(
        "-g",
        "--generate",
        help="Generate random formulas for demo",
        action="store_true",
    )

    parser.add_argument(
        "-l",
        "--log",
        help="Path to log file (optional)",
        default=f"log/analysis_{datetime.now():%Y%m%d_%H%M%S}.log",
    )

    parser.add_argument(
        "-p",
        "--processes",
        help="Number of parallel processes for directory processing",
        type=int,
        default=mp.cpu_count(),
    )

    parser.add_argument(
        "-c", "--csv", help="Path to csv file (optional)", default="results.csv"
    )

    args = parser.parse_args()

    # Setup logging
    global logger  # pylint: disable=global-statement
    logger = setup_logging(args.log)

    if args.generate:
        demo()
    else:
        try:
            if args.file:
                logger.info("Processing single file: %s", args.file)
                success = process_smt_file(args.file, args)
                sys.exit(0 if success else 1)

            elif args.directory:
                logger.info("Processing directory: %s", args.directory)
                process_directory(args.directory, args)

        except Exception as exc:
            logger.error("Fatal error: %s", str(exc))
            sys.exit(1)


def demo():
    """Generate and analyze a demo formula."""
    try:
        # Create test variables and formula
        x, y, z = z3.BitVecs("x y z", 8)
        variables = [x, y, z]

        formula = FormulaGenerator(variables).generate_formula()
        sol = z3.Solver()
        sol.add(formula)
        logger.debug("Generated formula: %s", sol.sexpr())
        counter = ModelCounter()

        while not counter.is_sat(formula):
            logger.info("Formula is unsatisfiable. Regenerating...")
            formula = FormulaGenerator(variables).generate_formula()
            sol = z3.Solver()
            sol.add(formula)
            logger.debug("Generated formula: %s", sol.sexpr())

        # Count models
        model_count = counter.count_models(formula)
        logger.info("SharpSAT model count: %s", model_count)

        # Analyze abstractions
        analyzer = AbstractionAnalyzer(formula, variables)
        results = analyzer.analyze_abstractions()

        if results:
            logger.info("Analysis completed successfully")
            return True

        return False

    except Exception as exc:
        logger.error("Error in main: %s", str(exc))
        return False


if __name__ == "__main__":
    main()
