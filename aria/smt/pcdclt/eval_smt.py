#!/usr/bin/env python3
"""Evaluation script for PCDCLT solver - compare performance and correctness with other SMT solvers.

Usage:
    python eval_smt.py --bench-dir benchmarks/smtlib2 --timeout 60 --parallel
    python eval_smt.py --bench-dir benchmarks/smtlib2 --solvers pcdclt z3 cvc5 --output results.json
"""

import argparse
import concurrent.futures
import json
import logging
import os
import signal
import subprocess
import sys
import time
import multiprocessing as mp
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from aria.smt.pcdclt import solve as pcdclt_solve
from aria.utils import SolverResult
from aria.global_params import SMT_SOLVERS_PATH
from aria.smt.pcdclt import config as pcdclt_config

logger = logging.getLogger('pcdclt_eval')


def kill_process_group(process, pgid, logger_inst):
    """Kill process group with graceful termination followed by force kill"""
    try:
        os.killpg(pgid, signal.SIGTERM)
        for _ in range(5):
            if process.poll() is not None:
                break
            time.sleep(0.1)
        if process.poll() is None:
            os.killpg(pgid, signal.SIGKILL)
        process.wait(timeout=1)
    except (subprocess.TimeoutExpired, ProcessLookupError, PermissionError) as e:
        logger_inst.warning(f"Error killing process group {pgid}: {e}")
        try:
            process.kill()
            process.wait(timeout=1)
        except Exception as e2:
            logger_inst.error(f"Failed to kill process: {e2}")


@dataclass
class SolverConfig:
    """Configuration for an SMT solver"""
    name: str
    solver_type: str  # 'pcdclt' or 'external'
    command: Optional[str] = None
    args: str = ""
    params: Optional[Dict] = None

    def __post_init__(self):
        if self.params is None:
            self.params = {}


def parse_result(output: str) -> SolverResult:
    """Parse solver output to extract result"""
    output_lower = output.strip().lower()
    if 'unsat' in output_lower:
        return SolverResult.UNSAT
    if 'sat' in output_lower:
        return SolverResult.SAT
    if 'unknown' in output_lower:
        return SolverResult.UNKNOWN
    return SolverResult.ERROR


def extract_logic(filepath: str) -> str:
    """Extract logic declaration from SMT-LIB2 file"""
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip().startswith('(set-logic'):
                    return line.split()[1].rstrip(')')
    except Exception:
        pass
    return "ALL"


def apply_pcdclt_params(params: Dict):
    """Apply PCDCLT configuration parameters"""
    for param, value in params.items():
        if hasattr(pcdclt_config, param.upper()):
            setattr(pcdclt_config, param.upper(), value)


def run_pcdclt(input_file: str, timeout: int, params: Optional[Dict] = None) -> Tuple[str, str, int, str, str, float]:
    """Run PCDCLT solver with process-based timeout using process groups"""
    start_time = time.time()

    # Create a wrapper script to run pcdclt
    import sys
    python_executable = sys.executable

    # Build command to run pcdclt via Python
    script = f"""
import sys
sys.path.insert(0, '{os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))}')
from aria.smt.pcdclt import solve as pcdclt_solve
from aria.smt.pcdclt import config as pcdclt_config

# Apply params if needed
{('for k, v in ' + str(params) + '.items():\\n    if hasattr(pcdclt_config, k.upper()):\\n        setattr(pcdclt_config, k.upper(), v)') if params else ''}

with open('{input_file}', 'r') as f:
    smt2_string = f.read()

# Extract logic
logic = 'ALL'
for line in smt2_string.split('\\n'):
    if line.strip().startswith('(set-logic'):
        logic = line.split()[1].rstrip(')')
        break

result = pcdclt_solve(smt2_string, logic=logic)
print(result.name.lower())
"""

    try:
        process = subprocess.Popen(
            [python_executable, '-c', script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid
        )
        pgid = os.getpgid(process.pid)

        try:
            stdout, stderr = process.communicate(timeout=timeout)
            retcode = process.returncode
        except subprocess.TimeoutExpired:
            kill_process_group(process, pgid, logger)
            stdout, stderr = "", "Timeout"
            retcode = -1

        elapsed = time.time() - start_time
        return (input_file, "pcdclt", retcode, stdout.strip(), stderr, elapsed)

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Error running PCDCLT: {e}")
        return (input_file, "pcdclt", 1, "", str(e), elapsed)


def run_external(solver: SolverConfig, input_file: str, timeout: int) -> Tuple[str, str, int, str, str, float]:
    """Run external SMT solver using process groups"""
    start_time = time.time()
    args_list = solver.args.split() if solver.args else []

    # Check if we should use stdin (when -in or -i flags are present)
    use_stdin = any(arg in ['-in', '-i'] for arg in args_list)

    cmd = [solver.command] + args_list
    if not use_stdin:
        cmd.append(input_file)

    try:
        if use_stdin:
            with open(input_file, 'r') as f:
                input_data = f.read()
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=os.setsid
            )
        else:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=os.setsid
            )

        pgid = os.getpgid(process.pid)

        try:
            if use_stdin:
                stdout, stderr = process.communicate(input=input_data, timeout=timeout)
            else:
                stdout, stderr = process.communicate(timeout=timeout)
            retcode = process.returncode
        except subprocess.TimeoutExpired:
            kill_process_group(process, pgid, logger)
            stdout, stderr = "", "Timeout"
            retcode = -1

        elapsed = time.time() - start_time
        return (input_file, solver.name, retcode, stdout, stderr, elapsed)

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Error running {solver.name}: {e}")
        return (input_file, solver.name, 1, "", str(e), elapsed)


def run_solver(solver: SolverConfig, input_file: str, timeout: int) -> Tuple[str, str, int, str, str, float]:
    """Run a solver (dispatches to PCDCLT or external)"""
    if solver.solver_type == 'pcdclt':
        return run_pcdclt(input_file, timeout, solver.params)
    return run_external(solver, input_file, timeout)


def load_config_file(config_file: str) -> Dict[str, SolverConfig]:
    """Load solver configurations from JSON file"""
    with open(config_file, 'r') as f:
        data = json.load(f)

    return {
        name: SolverConfig(
            name=name,
            solver_type=cfg.get('type', 'external'),
            command=cfg.get('command'),
            args=cfg.get('args', ''),
            params=cfg.get('params', {})
        )
        for name, cfg in data.items()
    }


def get_available_solvers(config_file: Optional[str] = None) -> Dict[str, SolverConfig]:
    """Get all available solvers (from config file or auto-detect)"""
    if config_file:
        return load_config_file(config_file)

    solvers = {'pcdclt': SolverConfig('pcdclt', 'pcdclt')}

    for name, config in SMT_SOLVERS_PATH.items():
        if config.get('available', False):
            solvers[name] = SolverConfig(name, 'external', config['path'], config.get('args', ''))

    return solvers


def compare_results(results: List[Tuple]) -> Dict:
    """Compare results across solvers for correctness checking"""
    file_results = defaultdict(dict)

    for file, solver, retcode, stdout, stderr, elapsed in results:
        # Always parse stdout first; fall back to timeout/error based on retcode
        result = parse_result(stdout) if stdout else SolverResult.ERROR
        if result == SolverResult.ERROR and retcode == -1:
            result = SolverResult.UNKNOWN  # Timeout
        file_results[file][solver] = {'result': result, 'time': elapsed}

    disagreements, agreements = [], 0

    for file, solver_data in file_results.items():
        valid = [(s, d['result']) for s, d in solver_data.items()
                if d['result'] in [SolverResult.SAT, SolverResult.UNSAT]]

        if len(valid) > 1:
            if len(set(r for _, r in valid)) == 1:
                agreements += 1
            else:
                disagreements.append({
                    'file': os.path.basename(file),
                    'results': {s: r.name for s, r in valid}
                })

    return {
        'total_files': len(file_results),
        'agreements': agreements,
        'disagreements': len(disagreements),
        'disagreement_details': disagreements
    }


def _init_solver_stats():
    return {
        'total': 0, 'success': 0, 'timeout': 0, 'failed': 0,
        'total_time': 0.0, 'max_time': 0.0,
        'results': {'sat': 0, 'unsat': 0, 'unknown': 0, 'error': 0},
        'files_timeout': [], 'files_failed': []
    }

def _update_stats(stats, retcode, elapsed, result_name, filename):
    stats['total'] += 1
    stats['total_time'] += elapsed
    stats['max_time'] = max(stats['max_time'], elapsed)

    if retcode == -1:
        stats['timeout'] += 1
        stats['files_timeout'].append(filename)
    elif result_name in ['sat', 'unsat']:
        # Count as success if we got SAT or UNSAT, regardless of return code
        stats['success'] += 1
        stats['results'][result_name] += 1
    else:
        stats['failed'] += 1
        stats['files_failed'].append(filename)
        if result_name:
            stats['results'][result_name] += 1

def _print_solver_summary(solver, stats, show_details):
    total = stats['total']
    print(f"\n{'Solver: ' + solver:-^80}")
    print(f"Files: {stats['success']}/{total} success, {stats['timeout']}/{total} timeout, {stats['failed']}/{total} failed")
    print(f"Time: total={stats['total_time']:.2f}s, avg={stats['total_time']/total:.2f}s, max={stats['max_time']:.2f}s")
    print(f"Results: SAT={stats['results']['sat']}, UNSAT={stats['results']['unsat']}, "
          f"UNKNOWN={stats['results']['unknown']}, ERROR={stats['results']['error']}")

    if show_details and (stats['files_timeout'] or stats['files_failed']):
        if stats['files_timeout']:
            timeout_list = ', '.join(stats['files_timeout'][:5])
            if len(stats['files_timeout']) > 5:
                timeout_list += f" ... +{len(stats['files_timeout'])-5} more"
            print(f"Timeouts: {timeout_list}")
        if stats['files_failed']:
            failed_list = ', '.join(stats['files_failed'][:5])
            if len(stats['files_failed']) > 5:
                failed_list += f" ... +{len(stats['files_failed'])-5} more"
            print(f"Failures: {failed_list}")

def summarize_results(results: List[Tuple], show_details: bool = False):
    """Summarize results by solver"""
    solver_stats = defaultdict(_init_solver_stats)

    for file, solver, retcode, stdout, stderr, elapsed in results:
        filename = os.path.basename(file)
        # Always parse the output to get the result, regardless of return code
        result_name = parse_result(stdout).name.lower() if stdout else None
        _update_stats(solver_stats[solver], retcode, elapsed, result_name, filename)

    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)

    for solver, stats in sorted(solver_stats.items()):
        _print_solver_summary(solver, stats, show_details)


def print_comparison(comparison: Dict):
    """Print correctness comparison summary"""
    print("\n" + "="*80)
    print("CORRECTNESS COMPARISON")
    print("="*80)
    print(f"Compared: {comparison['total_files']} files")
    print(f"Agreements: {comparison['agreements']}, Disagreements: {comparison['disagreements']}")

    if comparison['disagreements'] > 0:
        print(f"\n⚠️  WARNING: Found {comparison['disagreements']} disagreements!")
        for detail in comparison['disagreement_details'][:10]:
            print(f"  {detail['file']}: {detail['results']}")
        if len(comparison['disagreement_details']) > 10:
            print(f"  ... +{len(comparison['disagreement_details'])-10} more")
    else:
        print("\n✓ All solvers agree!")
    print("="*80)


def save_json(results: List[Tuple], comparison: Dict, output_file: str):
    """Save detailed results to JSON"""
    organized = defaultdict(dict)

    for file, solver, retcode, stdout, stderr, elapsed in results:
        # Always parse stdout first; fall back to timeout/error based on retcode
        parsed = parse_result(stdout) if stdout else SolverResult.ERROR
        if parsed == SolverResult.ERROR and retcode == -1:
            result = "TIMEOUT"
        else:
            result = parsed.name
        organized[file][solver] = {
            'result': result, 'time': elapsed, 'retcode': retcode,
            'stdout': stdout[:200] if stdout else None, 'stderr': stderr or None
        }

    with open(output_file, 'w') as f:
        json.dump({'results': dict(organized), 'comparison': comparison}, f, indent=2)
    logger.info(f"Results saved to {output_file}")


def _setup_args():
    parser = argparse.ArgumentParser(description="Evaluate PCDCLT solver performance and correctness")
    parser.add_argument("--bench-dir", required=True, help="Directory with SMT-LIB2 files")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout per benchmark (default: 60s)")
    parser.add_argument("--config", "-c", help="JSON config file for solver configurations")
    parser.add_argument("--solvers", nargs='+', help="Specific solvers to test (default: all available)")
    parser.add_argument("--parallel", action="store_true", help="Run benchmarks in parallel")
    parser.add_argument("--max-workers", type=int, help="Max parallel workers (default: CPU count)")
    parser.add_argument("--output", "-o", help="Output JSON file for detailed results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--pattern", default="*.smt2", help="File pattern (default: *.smt2)")
    return parser.parse_args()

def _find_benchmarks(bench_dir, pattern):
    bench_path = Path(bench_dir)
    if not bench_path.is_dir():
        logger.error(f"Directory not found: {bench_dir}")
        sys.exit(1)

    benchmarks = [str(f) for f in bench_path.glob(pattern)]
    if not benchmarks:
        logger.error(f"No {pattern} files found in {bench_dir}")
        sys.exit(1)

    logger.info(f"Found {len(benchmarks)} benchmark files")
    return benchmarks

def _log_progress(i, total, result, parallel=True):
    file, solver, retcode, stdout, stderr, elapsed = result
    status = "✓" if retcode == 0 else ("T" if retcode == -1 else "✗")
    logger.info(f"[{i}/{total}] {status} {solver:8s} {os.path.basename(file):40s} {elapsed:6.2f}s")

def _run_benchmarks_parallel(solvers, benchmarks, timeout, max_workers):
    results = []
    total = len(solvers) * len(benchmarks)

    max_workers = max_workers or min(os.cpu_count() or 4, total)
    logger.info(f"Running in parallel with {max_workers} workers (processes)")

    # Use ProcessPoolExecutor for true parallelism and proper process isolation
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(run_solver, solver, bench, timeout)
            for solver in solvers.values() for bench in benchmarks
        ]

        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                result = future.result()
                results.append(result)
                _log_progress(i, total, result, parallel=True)
            except Exception as e:
                logger.error(f"Future {i} failed: {e}")
                results.append(("unknown", "unknown", 1, "", str(e), 0.0))

    return results

def _run_benchmarks_sequential(solvers, benchmarks, timeout):
    results = []
    total = len(solvers) * len(benchmarks)

    logger.info("Running sequentially")
    for i, (solver, bench) in enumerate([(s, b) for s in solvers.values() for b in benchmarks], 1):
        result = run_solver(solver, bench, timeout)
        results.append(result)
        _log_progress(i, total, result, parallel=False)

    return results

def main():
    args = _setup_args()

    # Setup logging
    logging.basicConfig(level=getattr(logging, args.log_level),
                       format='%(asctime)s - %(levelname)s - %(message)s')

    # Get solvers
    all_solvers = get_available_solvers(args.config)
    if args.solvers:
        solvers = {n: all_solvers[n] for n in args.solvers if n in all_solvers}
        missing = [n for n in args.solvers if n not in all_solvers]
        if missing:
            logger.warning(f"Unavailable solvers: {missing}")
    else:
        solvers = all_solvers

    logger.info(f"Solvers: {list(solvers.keys())}, Timeout: {args.timeout}s")

    # Find benchmarks
    benchmarks = _find_benchmarks(args.bench_dir, args.pattern)

    # Run benchmarks
    results = (_run_benchmarks_parallel(solvers, benchmarks, args.timeout, args.max_workers)
              if args.parallel else _run_benchmarks_sequential(solvers, benchmarks, args.timeout))

    # Summarize and compare
    summarize_results(results, args.verbose)

    comparison = (compare_results(results) if len(solvers) > 1
                 else {'total_files': len(benchmarks), 'agreements': 0, 'disagreements': 0})

    if len(solvers) > 1:
        print_comparison(comparison)

    # Save JSON
    if args.output:
        save_json(results, comparison, args.output)

    # Exit with error if disagreements found
    if comparison['disagreements'] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
