#!/usr/bin/env python3
"""Run MBP algorithms on SMT2 files and record results as JSON."""

from __future__ import annotations

import argparse
import json
import os
import time
import multiprocessing as mp
from dataclasses import dataclass
from typing import Dict, List

from cores.unary_check_pysmt import (
    unary_check,
    unary_check_cached,
    unary_check_incremental,
    unary_check_incremental_cached,
)
from cores.dis_check_pysmt import (
    disjunctive_check_cached,
    disjunctive_check_incremental_cached,
)
from cores.new_check_pysmt import (
    core_lit_filter,
)

from utils.logger import setup_logger
from utils.parse_monabs_pysmt import parse_monabs_pysmt
from utils.utils import collect_smt2_files
import utils.config as cf


@dataclass
class RunResult:
    """Container for algorithm outputs and timings."""

    outputs: Dict[str, List[int]]
    solver_calls: Dict[str, int]
    times: Dict[str, float]
    total_time: float
    length: int = 0


def _run_algorithms(precond, constraints, timeout_ms) -> RunResult:
    outputs: Dict[str, List[int]] = {}
    solver_calls: Dict[str, int] = {}
    times: Dict[str, float] = {}

    start_total = time.perf_counter()

    t0 = time.perf_counter()
    outputs["LS-Naive"], solver_calls["LS-Naive"] = unary_check(precond, constraints, timeout_ms)
    times["LS-Naive"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    outputs["LS-Inc"], solver_calls["LS-Inc"] = unary_check_incremental(precond, constraints, timeout_ms)
    times["LS-Inc"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    outputs["LS-Reuse"], solver_calls["LS-Reuse"] = unary_check_cached(precond, constraints, timeout_ms)
    times["LS-Reuse"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    outputs["LS-IncReuse"], solver_calls["LS-IncReuse"] = unary_check_incremental_cached(precond, constraints, timeout_ms)
    times["LS-IncReuse"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    outputs["OA"], solver_calls["OA"] = disjunctive_check_cached(precond, constraints, timeout_ms)
    times["OA"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    outputs["OA-Inc"], solver_calls["OA-Inc"] = disjunctive_check_incremental_cached(precond, constraints, timeout_ms)
    times["OA-Inc"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    outputs["CORE-LIT-FILTER"], solver_calls["CORE-LIT-FILTER"] = core_lit_filter(precond, constraints, timeout_ms)
    times["CORE-LIT-FILTER"] = time.perf_counter() - t0

    total_time = time.perf_counter() - start_total

    return RunResult(outputs=outputs, solver_calls=solver_calls, times=times, total_time=total_time, length=len(constraints))


def _all_equal(values: List[List[int]]) -> bool:
    if not values:
        return True
    first = values[0]
    return all(v == first for v in values[1:])


def _process_single_file(filepath: str, timeout_ms: float) -> Dict:
    precond, constraints = parse_monabs_pysmt(filepath)

    if len(constraints) < cf.MIN_LENGTH:
        return {
            "id": filepath,
            "status": "invalid",
            "length": 0,
            "sat_ratio": 0.0,
            "total_execution_time": 0.0,
            "algo_execution_time": {},
            "results": {},
            "solver_calls": {},
        }

    run = _run_algorithms(precond, constraints, timeout_ms)
    
    if any(2 in result for result in run.outputs.values()): # 2 stands for unknown, aka timeout
        status = "timeout"
        sat_ratio = -1
    else:
        status = "valid" if _all_equal(list(run.outputs.values())) else "error"
        if run.length > 0:
            sat_ratio = sum(1 for v in run.outputs["LS-Naive"] if v == 1) / run.length
        else:
            sat_ratio = 0.0

    return {
        "id": filepath,
        "status": status,
        "length": run.length,
        "sat_ratio": sat_ratio,
        "total_execution_time": run.total_time,
        "algo_execution_time": run.times,
        "results": run.outputs,
        "solver_calls": run.solver_calls,
    }


def process_file(args) -> Dict:
    filepath, timeout_ms = args
    print("[Debug] Processing file:", filepath)
    return _process_single_file(filepath, timeout_ms)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MBP algorithms on SMT2 files and record JSON output."
    )
    parser.add_argument(
        "-i",
        "--input_directory",
        required=True,
        help="Path to the input directory containing SMT2 files.",
    )
    parser.add_argument(
        "-o",
        "--output_jsonl",
        required=True,
        help="Path to the output JSONL file.",
    )
    parser.add_argument(
        "-l",
        "--log_file",
        default='logs/test_pysmt_monabs.log',
        help="Optional log file path.",
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=float,
        default=30,
        help="Timeout in seconds for each algorithm run.",
    )
    parser.add_argument(
        "-w",
        "--max_workers",
        type=int,
        default=10,
        help="Number of worker processes.",
    )
    args = parser.parse_args()

    logger = setup_logger(log_file=args.log_file)

    timeout_ms = int(args.timeout * 1000)
    logger.info("Set solver timeout to %d milliseconds", timeout_ms)

    smt2_files = collect_smt2_files(args.input_directory)
    logger.info("Found %d SMT2 files under %s", len(smt2_files), args.input_directory)

    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)
    with open(args.output_jsonl, "a", encoding="utf-8") as f:
        with mp.Pool(processes=args.max_workers) as pool:
            for result in pool.imap_unordered(
                process_file, ((p, timeout_ms) for p in smt2_files)
            ):
                logger.info("Processed %s [%s]", result.get("id"), result.get("status"))
                f.write(json.dumps(result) + "\n")
                f.flush()


if __name__ == "__main__":
    main()
