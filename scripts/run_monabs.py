"""Regression tests for monadic predicate abstraction"""

import argparse
import logging
import multiprocessing as mp

# import csv
import random
import time
from typing import List, Tuple

import z3

from aria.monabs.dis_check import disjunctive_check, disjunctive_check_incremental
from aria.monabs.unary_check import unary_check, unary_check_cached
from aria.monabs.unsat_check import unsat_check
from aria.tests.formula_generator import FormulaGenerator


def check_identical(*lists):
    mismatched_pairs = []

    for i in range(len(lists)):
        for j in range(i + 1, len(lists)):
            if lists[i] != lists[j]:
                mismatched_pairs.append((i, j))

    return mismatched_pairs


def run_single_test(logic_type: str, timeout: int) -> Tuple[str, float, dict]:
    """Run a single test case with specified logic type and timeout"""
    # FIXME: here, we use the random generator for generating a test case.
    # Later, may need to parse inputs dumpted from real-world clients.
    if logic_type == "int":
        x, y, z = z3.Ints("x y z")
        init_vars = [x, y, z]
    elif logic_type == "real":
        x, y, z = z3.Reals("x y z")
        init_vars = [x, y, z]
    else:  # bitvector
        x, y, z = z3.BitVecs("x y z", 32)
        init_vars = [x, y, z]

    # Generate formulas
    generator = FormulaGenerator(init_vars)
    precond = generator.generate_formula()
    constraints = generator.get_preds(random.randint(5, 15))

    s = z3.Solver()
    s.set("timeout", 3000)
    s.add(precond)
    if s.check() != z3.sat:
        return "invalid", -1, {}

    # Store timing stats for each approach
    stats = {}
    total_start_time = time.time()
    try:
        # Unary check
        start = time.time()
        res_unary = unary_check(precond, constraints)
        stats["unary"] = time.time() - start

        # Unary check with cache
        start = time.time()
        res_unary_cached = unary_check_cached(precond, constraints)
        stats["unary_cached"] = time.time() - start

        # Disjunctive check
        start = time.time()
        res_dis = disjunctive_check(precond, constraints)
        stats["disjunctive"] = time.time() - start

        # Disjunctive check incremental
        start = time.time()
        res_dis_inc = disjunctive_check_incremental(precond, constraints)
        stats["disjunctive_inc"] = time.time() - start

        # New: unsat check
        start = time.time()
        res_unsat = unsat_check(precond, constraints)
        stats["unsat"] = time.time() - start

        # Check for inconsistency
        check_res = check_identical(
            res_unary, res_unary_cached, res_dis, res_dis_inc, res_unsat
        )

        status = "failure" if len(check_res) > 0 else "success"

        # Add result stats
        stats["num_constraints"] = len(constraints)
        stats["consistent"] = len(check_res) == 0

    except z3.Z3Exception as e:
        status = f"error: {str(e)}"
        stats["error"] = str(e)

    total_time = time.time() - total_start_time
    return status, total_time, stats


def worker(args: Tuple[int, str, int, str]) -> Tuple[int, str, float, dict]:
    """Worker function for parallel execution"""
    # FIXME: In case of timeout, shoud we start a new process to call the APIs? (instead of calling the API via Python directly.)
    test_id, logic_type, timeout, _ = args
    status, execution_time, stats = run_single_test(logic_type, timeout)
    return test_id, status, execution_time, stats


def main():
    # mp.cpu_count()
    parser = argparse.ArgumentParser(description="Z3 Formula Testing Script")
    parser.add_argument(
        "--num_tests", type=int, default=100, help="Number of test cases"
    )
    parser.add_argument(
        "--logic_type", choices=["int", "real", "bv"], default="int", help="Logic type"
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of parallel workers"
    )
    parser.add_argument(
        "--timeout", type=int, default=1000, help="Timeout in milliseconds"
    )
    parser.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )
    parser.add_argument(
        "--output", type=str, default="results.csv", help="Output CSV file"
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=getattr(logging, args.log_level))
    logger = logging.getLogger(__name__)

    # Prepare test parameters
    test_params = [
        (i, args.logic_type, args.timeout, str(time.time()))
        for i in range(args.num_tests)
    ]

    # Run tests in parallel
    logger.info(f"Starting {args.num_tests} tests with {args.workers} workers")
    with mp.Pool(args.workers) as pool:
        results = pool.map(worker, test_params)

    # Collect statistics
    total_stats = {
        "unary": [],
        "unary_cached": [],
        "disjunctive": [],
        "disjunctive_inc": [],
        "unsat": [],
        "num_constraints": [],
    }

    for _, status, _, stats in results:
        if status == "success":
            for key in total_stats:
                if key in stats:
                    total_stats[key].append(stats[key])

    # Print detailed statistics
    logger.info("\nPerformance Statistics:")
    for method in ["unary", "unary_cached", "disjunctive", "disjunctive_inc", "unsat"]:
        times = total_stats[method]
        if times:
            avg_time = sum(times) / len(times)
            max_time = max(times)
            min_time = min(times)
            logger.info(
                f"{method:15s}: avg={avg_time:.3f}s, min={min_time:.3f}s, max={max_time:.3f}s"
            )

    # Print summary
    success_count = sum(1 for r in results if r[1] == "success")
    invalid_count = sum(1 for r in results if r[1] == "invalid")
    logger.info(f"\nTests completed: {len(results)}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Invalid: {invalid_count}")
    logger.info(f"Failed: {len(results) - success_count - invalid_count}")


if __name__ == "__main__":
    main()
