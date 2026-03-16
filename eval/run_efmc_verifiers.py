"""
Running different configurations of efmc
"""

import logging
import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
import csv
from datetime import datetime
from typing import List, Dict
from efmc.utils.eval_utils import (
    kill_process_group,
    classify_result,
    detect_inconsistencies,
)


def setup_logging(log_level=logging.INFO):
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger("efmc_verifier")


class VerifierConfig:
    def __init__(
        self,
        engine: str,
        template: str = None,
        aux_inv: bool = False,
        lang: str = "sygus",
        additional_opts: List[str] = None,
        external_solver: str = None,
    ):
        self.engine = engine
        self.template = template
        self.aux_inv = aux_inv
        self.lang = lang
        self.additional_opts = additional_opts or []
        self.external_solver = external_solver

    def get_command_args(self) -> List[str]:
        if self.external_solver:
            # For external solvers, return the solver path and its arguments
            return [self.external_solver]
        else:
            # For EFMC internal engines
            cmd = ["--engine", self.engine]
            if self.template:
                cmd.extend(["--template", self.template])
            if self.aux_inv:
                cmd.append("--kind-aux-inv")
            cmd.extend(["--lang", self.lang, *self.additional_opts])
            return cmd

    @property
    def config_id(self) -> str:
        if self.external_solver:
            solver_name = os.path.basename(self.external_solver)
            return f"external_{solver_name}"

        config_id = self.engine
        if self.template:
            config_id += f"_{self.template}"
        if self.aux_inv:
            config_id += "_aux"
        if "--prop-strengthen" in self.additional_opts:
            config_id += "_strengthen"
        return config_id


def load_config(config_file: str) -> Dict[str, List[VerifierConfig]]:
    with open(config_file) as f:
        data = json.load(f)
    return {name: [VerifierConfig(**c) for c in cfg] for name, cfg in data.items()}


def run_verifier(
    verifier_path: str, config: VerifierConfig, input_file: str, timeout: int
) -> tuple:
    start_time = time.time()

    if config.external_solver:
        # For external solvers, run them directly on the input file
        cmd = config.get_command_args() + [input_file]
    else:
        # For EFMC internal engines
        cmd = [verifier_path] + config.get_command_args() + ["--file", input_file]

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid,
        )
        pgid = os.getpgid(process.pid)

        try:
            stdout, stderr = process.communicate(timeout=timeout)
            retcode = process.returncode
        except subprocess.TimeoutExpired:
            kill_process_group(process, pgid, logging.getLogger("efmc_verifier"))
            stdout, stderr = "", "Timeout"
            retcode = -1

        elapsed = time.time() - start_time
        result = classify_result(config, retcode, stdout)
        return (input_file, config.config_id, result, stdout, stderr, elapsed)

    except Exception as e:
        elapsed = time.time() - start_time
        logging.getLogger("efmc_verifier").error(f"Error running verifier: {e}")
        return input_file, config.config_id, "unknown", "", str(e), elapsed


def print_result(result):
    file, engine, result_class, stdout, stderr, elapsed = result
    print(
        f"\nFile: {os.path.basename(file)}, Engine: {engine}, Result: {result_class}, Time: {elapsed:.2f}s"
    )
    if stdout:
        print(f"Output: {stdout}")
    if stderr:
        print(f"Errors: {stderr}")
    sys.stdout.flush()


def dump_results(results: List[tuple], output_dir: str = "eval_results"):
    """Dump evaluation results to JSON and CSV files for later processing"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Prepare data for JSON dump
    json_data = {"timestamp": timestamp, "total_runs": len(results), "results": []}

    for file, config_id, result_class, stdout, stderr, elapsed in results:
        result_entry = {
            "file": file,
            "filename": os.path.basename(file),
            "config_id": config_id,
            "result": result_class,
            "stdout": stdout,
            "stderr": stderr,
            "elapsed_time": elapsed,
        }
        json_data["results"].append(result_entry)

    # Write JSON file
    json_file = os.path.join(output_dir, f"results_{timestamp}.json")
    with open(json_file, "w") as f:
        json.dump(json_data, f, indent=2)

    # Write CSV file
    csv_file = os.path.join(output_dir, f"results_{timestamp}.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "file",
                "filename",
                "config_id",
                "result",
                "elapsed_time",
                "stdout",
                "stderr",
            ]
        )
        for file, config_id, result_class, stdout, stderr, elapsed in results:
            writer.writerow(
                [
                    file,
                    os.path.basename(file),
                    config_id,
                    result_class,
                    elapsed,
                    stdout,
                    stderr,
                ]
            )

    print(f"\nResults dumped to:")
    print(f"  JSON: {json_file}")
    print(f"  CSV:  {csv_file}")

    return json_file, csv_file


def summarize_results(results: List[tuple]):
    # Collect statistics
    config_results = {}
    for file, config_id, result_class, stdout, stderr, elapsed in results:
        if config_id not in config_results:
            config_results[config_id] = {
                "total": 0,
                "safe": 0,
                "unsafe": 0,
                "timeout": 0,
                "unknown": 0,
                "total_time": 0.0,
                "max_time": 0.0,
            }

        stats = config_results[config_id]
        stats["total"] += 1
        stats["total_time"] += elapsed
        stats["max_time"] = max(stats["max_time"], elapsed)
        stats[result_class] += 1

    # Print comparison table
    print("\n=== Configuration Comparison Table ===")
    if config_results:
        print(
            f"{'Configuration':<25} {'Total':<6} {'Safe':<6} {'Unsafe':<7} {'Solved':<7} {'Timeout':<8} {'Unknown':<8} {'Success%':<10} {'Avg Time':<10} {'Max Time':<10}"
        )
        print("-" * 110)

        # Sort by solved rate
        sorted_configs = sorted(
            config_results.items(),
            key=lambda x: (x[1]["safe"] + x[1]["unsafe"]) / x[1]["total"],
            reverse=True,
        )

        for config_id, stats in sorted_configs:
            total_solved = stats["safe"] + stats["unsafe"]
            success_rate = total_solved * 100 / stats["total"]
            avg_time = stats["total_time"] / stats["total"]

            print(
                f"{config_id:<25} {stats['total']:<6} {stats['safe']:<6} {stats['unsafe']:<7} {total_solved:<7} "
                f"{stats['timeout']:<8} {stats['unknown']:<8} {success_rate:<9.1f}% "
                f"{avg_time:<9.2f}s {stats['max_time']:<9.2f}s"
            )

    # Print inconsistency report
    inconsistencies = detect_inconsistencies(results)
    if inconsistencies:
        print(f"\n=== INCONSISTENT RESULTS DETECTED ({len(inconsistencies)} files) ===")
        print("WARNING: Conflicting verification results detected!\n")

        for filename, file_data in sorted(inconsistencies.items()):
            print(f"File: {filename}")

            # Group by result type
            by_result = {}
            for data in file_data:
                by_result.setdefault(data["result"], []).append(data)

            # Show conflicting definitive results
            for result_type, configs in by_result.items():
                if result_type in ["safe", "unsafe"]:
                    print(
                        f"  {result_type.upper()}: {', '.join(c['config'] for c in configs)}"
                    )

            print("  All results:")
            for data in sorted(file_data, key=lambda x: x["config"]):
                print(
                    f"    {data['config']}: {data['result']} ({data['elapsed']:.2f}s)"
                )
            print()

        print(f"Total inconsistent files: {len(inconsistencies)}\n")
    else:
        print("\n=== NO INCONSISTENCIES DETECTED ===")


def run_all_configs(configs, benchmark_files, args, parallel=False):
    results = []

    if parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(
                    run_verifier, args.verifier, config, benchmark, args.timeout
                )
                for config_list in configs.values()
                for config in config_list
                for benchmark in benchmark_files
            ]

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                print_result(result)
    else:
        for config_list in configs.values():
            for config in config_list:
                for benchmark in benchmark_files:
                    result = run_verifier(
                        args.verifier, config, benchmark, args.timeout
                    )
                    results.append(result)
                    print_result(result)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run verifier with multiple configurations"
    )
    parser.add_argument(
        "--bench-dir",
        nargs="+",
        required=True,
        help="Directory(ies) containing benchmark files",
    )
    parser.add_argument("--timeout", type=int, required=True, help="Timeout in seconds")
    parser.add_argument("--config", help="Configuration file in JSON format")
    parser.add_argument(
        "--parallel", action="store_true", help="Run configurations in parallel"
    )
    parser.add_argument(
        "--verifier", default="efmc", help="Path to the verifier executable"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )
    parser.add_argument(
        "--dump-results", action="store_true", help="Dump results to JSON and CSV files"
    )
    parser.add_argument(
        "--output-dir",
        default="eval_results",
        help="Directory to save result files (default: eval_results)",
    )
    args = parser.parse_args()

    logger = setup_logging(getattr(logging, args.log_level))
    logger.info(
        f"Starting EFMC verifier with bench-dir={', '.join(args.bench_dir)}, timeout={args.timeout}s"
    )

    # Load configurations
    if args.config:
        configs = load_config(args.config)
        logger.info(f"Loaded custom configuration from {args.config}")
    else:
        # Get the path to Eldarica solver
        eldarica_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "bin_solvers",
            "bin",
            "eldarica",
        )

        configs = {
            "default": [
                # EFMC internal engines
                VerifierConfig("ef", template="bv_interval", aux_inv=False, lang="chc"),
                VerifierConfig("kind", aux_inv=False, lang="chc"),
                # FIXME: kind  and kind+aux_inv seems to be buggy
                # See data/BV/LoopInvGen/chc/2016.SyGuS-Comp/64bits_unsigned/fig1_vars.sl_64bits_unsigned.smt2
                #  - eldarica returns "unsafe", but kind returns "safe"??
                # data/BV/LoopInvGen/chc/2016.SyGuS-Comp/64bits_unsigned/
                # VerifierConfig("kind", aux_inv=True, lang="chc"),
                VerifierConfig("pdr", lang="chc"),  # Added PDR engine
                VerifierConfig("ef", template="bv_octagon", lang="chc"),
                VerifierConfig("ef", template="knownbits", lang="chc"),
                VerifierConfig(
                    "ef",
                    template="knownbits",
                    lang="chc",
                    additional_opts=["--prop-strengthen"],
                ),
                VerifierConfig(
                    "ef",
                    template="bv_interval",
                    lang="chc",
                    additional_opts=["--prop-strengthen"],
                ),
                # Bit-level template
                VerifierConfig("ef", template="bv_pattern", lang="chc"),
                VerifierConfig("ef", template="bv_rotation", lang="chc"),
                VerifierConfig(
                    "ef",
                    template="bv_xor_parity",
                    lang="chc",
                    additional_opts=[
                        "--prop-strengthen",
                        "--strengthen-templates",
                        "bv_interval",
                    ],
                ),
                # External solvers
                (
                    VerifierConfig("external", external_solver=eldarica_path)
                    if os.path.exists(eldarica_path)
                    else None
                ),
            ]
        }
        # Remove None entries (when external solver doesn't exist)
        configs["default"] = [c for c in configs["default"] if c is not None]
        logger.info("Using default configurations")

    # Get benchmark files from all directories
    benchmark_files = []
    for bench_dir in args.bench_dir:
        if not os.path.isdir(bench_dir):
            logger.error(f"Benchmark directory does not exist: {bench_dir}")
            sys.exit(1)

        dir_files = [
            os.path.join(bench_dir, f)
            for f in os.listdir(bench_dir)
            if f.endswith((".sl", ".smt2"))
        ]
        benchmark_files.extend(dir_files)
        logger.info(f"Found {len(dir_files)} benchmark files in {bench_dir}")

    if not benchmark_files:
        logger.warning(
            f"No .sl or .smt2 files found in any of the benchmark directories: {', '.join(args.bench_dir)}"
        )
        sys.exit(1)

    logger.info(f"Found {len(benchmark_files)} benchmark files")

    # Run all configurations and summarize
    results = run_all_configs(configs, benchmark_files, args, args.parallel)

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    logger.info(f"Completed {len(results)} verification tasks")

    # Dump results if requested
    if args.dump_results:
        json_file, csv_file = dump_results(results, args.output_dir)
        logger.info(f"Results saved to {json_file} and {csv_file}")

    inconsistencies = detect_inconsistencies(results)
    if inconsistencies:
        logger.warning(
            f"DETECTED {len(inconsistencies)} FILES WITH INCONSISTENT RESULTS!"
        )

    summarize_results(results)


if __name__ == "__main__":
    main()
