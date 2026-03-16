"""
Running different SMT solvers for Exists-Forall Problems (not used for now)
"""

import logging
import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from typing import List, Dict
from efmc.utils.eval_utils import kill_process_group


def setup_logging(log_level=logging.INFO):
    logger = logging.getLogger("efsmt_solver")
    logger.setLevel(log_level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


class SolverConfig:
    def __init__(self, name: str, command: str, additional_opts: List[str] = None):
        self.name = name
        self.command = command
        self.additional_opts = additional_opts or []

    def get_command_args(self, input_file: str) -> List[str]:
        cmd = [self.command]
        cmd.extend(self.additional_opts)
        cmd.append(input_file)
        return cmd


def load_config(config_file: str) -> Dict[str, SolverConfig]:
    logger = logging.getLogger("efsmt_solver")
    logger.info(f"Loading configuration from {config_file}")

    with open(config_file) as f:
        data = json.load(f)

    configs = {}
    for name, cfg in data.items():
        configs[name] = SolverConfig(
            name=name,
            command=cfg.get("command"),
            additional_opts=cfg.get("additional_opts", []),
        )

    logger.info(f"Loaded {len(configs)} solvers")
    return configs


def run_solver(solver: SolverConfig, input_file: str, timeout: int) -> tuple:
    logger = logging.getLogger("efsmt_solver")
    start_time = time.time()
    cmd = solver.get_command_args(input_file)

    logger.debug(f"Running command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        elapsed = time.time() - start_time

        # Initialize retcode
        retcode = result.returncode

        if retcode != 0:
            logger.warning(
                f"Solver {solver.name} failed for {os.path.basename(input_file)}"
            )
        else:
            logger.debug(
                f"Finished {os.path.basename(input_file)} with {solver.name} in {elapsed:.2f}s"
            )

        return (input_file, solver.name, retcode, result.stdout, result.stderr, elapsed)
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        logger.warning(
            f"Timeout for {os.path.basename(input_file)} with {solver.name} after {elapsed:.2f}s"
        )
        return input_file, solver.name, -1, "", "Timeout", elapsed


def summarize_results(results: List[tuple]):
    logger = logging.getLogger("efsmt_solver")

    solver_results = {}
    for result in results:
        file, solver_name, retcode, stdout, stderr, elapsed = result
        if solver_name not in solver_results:
            solver_results[solver_name] = {
                "total": 0,
                "success": 0,
                "timeout": 0,
                "failed": 0,
                "total_time": 0.0,
                "max_time": 0.0,
                "files_success": [],
                "files_timeout": [],
                "files_failed": [],
            }

        stats = solver_results[solver_name]
        stats["total"] += 1
        stats["total_time"] += elapsed
        stats["max_time"] = max(stats["max_time"], elapsed)

        if retcode == 0:
            stats["success"] += 1
            stats["files_success"].append(os.path.basename(file))
        elif retcode == -1:
            stats["timeout"] += 1
            stats["files_timeout"].append(os.path.basename(file))
        else:
            stats["failed"] += 1
            stats["files_failed"].append(os.path.basename(file))

    print("\n=== Summary of Results ===")
    for solver_name, stats in solver_results.items():
        print(f"\nSolver: {solver_name}")
        print(f"Total files processed: {stats['total']}")
        print(
            f"Successful runs: {stats['success']} ({stats['success']*100/stats['total']:.1f}%)"
        )
        print(
            f"Timeouts: {stats['timeout']} ({stats['timeout']*100/stats['total']:.1f}%)"
        )
        print(
            f"Failed runs: {stats['failed']} ({stats['failed']*100/stats['total']:.1f}%)"
        )
        print(f"Total time: {stats['total_time']:.2f}s")
        print(f"Average time: {stats['total_time']/stats['total']:.2f}s")
        print(f"Max time: {stats['max_time']:.2f}s")

        if stats["files_timeout"]:
            print("\nTimeout files:")
            for f in sorted(stats["files_timeout"]):
                print(f"  - {f}")

        if stats["files_failed"]:
            print("\nFailed files:")
            for f in sorted(stats["files_failed"]):
                print(f"  - {f}")


def main():
    parser = argparse.ArgumentParser(description="Run SMT solvers on benchmarks")
    parser.add_argument(
        "--bench-dir", required=True, help="Directory containing benchmark files"
    )
    parser.add_argument("--timeout", type=int, required=True, help="Timeout in seconds")
    parser.add_argument(
        "--config", required=True, help="Configuration file in JSON format"
    )
    parser.add_argument(
        "--parallel", action="store_true", help="Run solvers in parallel"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )
    args = parser.parse_args()

    log_level = getattr(logging, args.log_level)
    logger = setup_logging(log_level)

    logger.info(
        f"Starting SMT solvers with bench-dir={args.bench_dir}, timeout={args.timeout}s"
    )

    configs = load_config(args.config)

    if not os.path.isdir(args.bench_dir):
        logger.error(f"Benchmark directory does not exist: {args.bench_dir}")
        sys.exit(1)

    benchmark_files = [
        os.path.join(args.bench_dir, f)
        for f in os.listdir(args.bench_dir)
        if f.endswith(".smt2")
    ]

    logger.info(f"Found {len(benchmark_files)} benchmark files")

    if not benchmark_files:
        logger.warning(f"No .smt2 files found in {args.bench_dir}")
        sys.exit(1)

    results = []

    if args.parallel:
        logger.info("Running in parallel mode")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for solver in configs.values():
                for benchmark in benchmark_files:
                    futures.append(
                        executor.submit(run_solver, solver, benchmark, args.timeout)
                    )

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                file, solver_name, retcode, stdout, stderr, elapsed = result
                print(f"\nFile: {os.path.basename(file)}")
                print(f"Solver: {solver_name}")
                print(f"Return code: {retcode}")
                print(f"Elapsed time: {elapsed:.2f}s")
                if stdout:
                    print(f"Output:\n{stdout}")
                if stderr:
                    print(f"Errors:\n{stderr}")
                sys.stdout.flush()
    else:
        logger.info("Running in sequential mode")
        for solver in configs.values():
            for benchmark in benchmark_files:
                result = run_solver(solver, benchmark, args.timeout)
                results.append(result)
                file, solver_name, retcode, stdout, stderr, elapsed = result
                print(f"\nFile: {os.path.basename(file)}")
                print(f"Solver: {solver_name}")
                print(f"Return code: {retcode}")
                print(f"Elapsed time: {elapsed:.2f}s")
                if stdout:
                    print(f"Output:\n{stdout}")
                if stderr:
                    print(f"Errors:\n{stderr}")
                sys.stdout.flush()

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    logger.info(f"Completed {len(results)} solver runs")
    summarize_results(results)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
