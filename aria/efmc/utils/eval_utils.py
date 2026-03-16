"""Utility functions for evaluation and result classification."""

import os
import signal
import subprocess
import time
from typing import List, Dict


def kill_process_group(process, pgid, logger):
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
        logger.warning(f"Error killing process group {pgid}: {e}")
        try:
            process.kill()
            process.wait(timeout=1)
        except (subprocess.TimeoutExpired, ProcessLookupError, PermissionError) as e2:
            logger.error(f"Failed to kill process: {e2}")


def classify_result(_config, retcode, stdout):
    """Classify verification result into safe/unsafe/unknown/timeout"""
    if retcode == -1:
        return "timeout"

    if not stdout:
        return "unknown"

    stdout_lower = stdout.lower()

    # Check for explicit safe/unsafe in output
    # Priority: unsafe > safe (because unsafe is more definitive)
    if "unsafe" in stdout_lower:
        return "unsafe"
    if "safe" in stdout_lower:
        return "safe"
    if "timeout" in stdout_lower:
        return "timeout"
    # Handle Eldarica sat/unsat output (sat means safe, unsat means unsafe)
    if "unsat" in stdout_lower:
        return "unsafe"
    if "sat" in stdout_lower and "unsat" not in stdout_lower:
        return "safe"
    # Default: what if the return code is 0, but no output?
    return "unknown"


def detect_inconsistencies(results: List[tuple]) -> Dict[str, List[Dict]]:
    """Detect inconsistent results between different verifiers for the same file"""
    # Group results by file
    file_results = {}
    for file, config_id, result_class, stdout, stderr, elapsed in results:
        basename = os.path.basename(file)
        if basename not in file_results:
            file_results[basename] = []

        file_results[basename].append(
            {
                "config": config_id,
                "result": result_class,
                "stdout": stdout,
                "stderr": stderr,
                "elapsed": elapsed,
            }
        )

    # Find inconsistencies
    inconsistencies = {}

    for filename, file_data in file_results.items():
        # Get all definitive results (safe/unsafe, excluding timeout/unknown/solved)
        definitive_results = [r for r in file_data if r["result"] in ["safe", "unsafe"]]

        if len(definitive_results) < 2:
            continue  # Need at least 2 definitive results to detect inconsistency

        # Check if there are conflicting definitive results
        result_types = set(r["result"] for r in definitive_results)
        if len(result_types) > 1:  # Inconsistency detected
            inconsistencies[filename] = file_data

    return inconsistencies
