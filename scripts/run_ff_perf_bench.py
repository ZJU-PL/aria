#!/usr/bin/env python3
"""Deterministic benchmark runner for FF backend comparisons.

Example:
    python3 scripts/run_ff_perf_bench.py \
      --bench-dir benchmarks/smtlib2/ff \
      --timeouts 10,30,60 \
      --repetitions 3 \
      --out results/ff_perf_bench.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import subprocess
import sys
import time
from typing import Dict, List, Tuple


def _solver_cmd(backend: str, file_path: str) -> List[str]:
    """Build a one-shot subprocess command for a backend/file pair."""
    if backend == "bv":
        init = "from aria.smt.ff import parse_ff_file, FFBVSolver;"
        solver_name = "FFBVSolver"
    elif backend == "bv2":
        init = "from aria.smt.ff import parse_ff_file, FFBVBridgeSolver;"
        solver_name = "FFBVBridgeSolver"
    elif backend == "int":
        init = "from aria.smt.ff import parse_ff_file, FFIntSolver;"
        solver_name = "FFIntSolver"
    elif backend == "auto":
        init = "from aria.smt.ff import parse_ff_file, FFAutoSolver;"
        solver_name = "FFAutoSolver"
    elif backend == "perf":
        init = "from aria.smt.ff import parse_ff_file, FFPerfSolver;"
        solver_name = "FFPerfSolver"
    else:
        raise ValueError("Unknown backend %s" % backend)
    run = (
        "import json;"
        "f=parse_ff_file(r'%s');"
        "s=%s();"
        "verdict=str(s.check(f));"
        "payload={'verdict': verdict};"
        "payload['stats']=s.stats() if hasattr(s,'stats') else {};"
        "payload['trace']=s.trace() if hasattr(s,'trace') else [];"
        "payload['backend_name']=getattr(s,'backend_name',None);"
        "print(json.dumps(payload, sort_keys=True))"
    ) % (file_path, solver_name)
    return [sys.executable, "-c", init + run]


def run_one(backend: str, file_path: str, timeout_s: int) -> Tuple[str, float, Dict[str, object]]:
    """Execute one backend run and return verdict, time, and payload."""
    cmd = _solver_cmd(backend, file_path)
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=float(timeout_s),
            check=False,
        )
        elapsed = time.perf_counter() - start
        if proc.returncode != 0:
            return ("error", elapsed, {"verdict": "error", "stats": {}, "trace": []})
        try:
            payload = json.loads(proc.stdout.strip())
        except json.JSONDecodeError:
            return (
                "error",
                elapsed,
                {"verdict": "error", "stats": {}, "trace": [], "raw": proc.stdout},
            )
        return (str(payload.get("verdict", "unknown")), elapsed, payload)
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - start
        return ("timeout", elapsed, {"verdict": "timeout", "stats": {}, "trace": []})


def par2(score_times: List[float], timeout_s: int) -> float:
    """Compute average PAR-2 score for an already-penalized score list."""
    del timeout_s
    if not score_times:
        return 0.0
    return sum(score_times) / float(len(score_times))


def _majority_verdict(verdicts: List[str]) -> str:
    """Return deterministic majority verdict (stable tie-break)."""
    counts: Dict[str, int] = {}
    for verdict in verdicts:
        counts[verdict] = counts.get(verdict, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ordered[0][0]


def _aggregate_numeric_dict(dicts: List[Dict[str, object]]) -> Dict[str, float]:
    aggregate: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for stats in dicts:
        for key, value in stats.items():
            if isinstance(value, bool):
                numeric = float(int(value))
            elif isinstance(value, (int, float)):
                numeric = float(value)
            else:
                continue
            aggregate[key] = aggregate.get(key, 0.0) + numeric
            counts[key] = counts.get(key, 0) + 1
    return {
        key: aggregate[key] / float(counts[key])
        for key in sorted(aggregate)
        if counts.get(key, 0) > 0
    }


def main() -> int:
    """Run a deterministic FF benchmark campaign and write JSON results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-dir", required=True)
    parser.add_argument("--backends", default="bv,bv2,int,auto,perf")
    parser.add_argument("--timeouts", default="10,30,60")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    bench_files = sorted(pathlib.Path(args.bench_dir).rglob("*.smt2"))
    backends = [x.strip() for x in args.backends.split(",") if x.strip()]
    timeout_list = [int(x.strip()) for x in args.timeouts.split(",") if x.strip()]

    results: Dict[str, object] = {
        "bench_dir": str(pathlib.Path(args.bench_dir).resolve()),
        "backends": backends,
        "timeouts": timeout_list,
        "repetitions": args.repetitions,
        "instances": [str(p) for p in bench_files],
        "runs": {},
        "summary": {},
    }

    for timeout_s in timeout_list:
        t_key = str(timeout_s)
        results["runs"][t_key] = {}
        for backend in backends:
            instance_runs = {}
            solved = 0
            timeout_count = 0
            par2_times = []
            for bench in bench_files:
                verdicts = []
                times = []
                payloads = []
                for _ in range(args.repetitions):
                    verdict, elapsed, payload = run_one(backend, str(bench), timeout_s)
                    verdicts.append(verdict)
                    times.append(elapsed)
                    payloads.append(payload)
                median_t = statistics.median(times)
                majority_verdict = _majority_verdict(verdicts)
                majority_payloads = [
                    payloads[idx]
                    for idx, verdict in enumerate(verdicts)
                    if verdict == majority_verdict
                ]
                instance_runs[str(bench)] = {
                    "verdicts": verdicts,
                    "times": times,
                    "median_time": median_t,
                    "majority_verdict": majority_verdict,
                    "payloads": payloads,
                    "avg_stats": _aggregate_numeric_dict(
                        [
                            payload.get("stats", {})
                            for payload in majority_payloads
                            if isinstance(payload.get("stats", {}), dict)
                        ]
                    ),
                    "avg_trace_length": statistics.mean(
                        [
                            len(payload.get("trace", []))
                            for payload in majority_payloads
                            if isinstance(payload.get("trace", []), list)
                        ]
                        or [0]
                    ),
                }
                if majority_verdict in ("sat", "unsat"):
                    solved += 1
                    par2_times.append(median_t)
                else:
                    if majority_verdict == "timeout":
                        timeout_count += 1
                    par2_times.append(float(2 * timeout_s))

            results["runs"][t_key][backend] = instance_runs
            results["summary"].setdefault(t_key, {})
            results["summary"][t_key][backend] = {
                "solved": solved,
                "timeouts": timeout_count,
                "par2": par2(par2_times, timeout_s),
                "avg_stats": _aggregate_numeric_dict(
                    [
                        run_data.get("avg_stats", {})
                        for run_data in instance_runs.values()
                        if isinstance(run_data.get("avg_stats", {}), dict)
                    ]
                ),
                "avg_trace_length": statistics.mean(
                    [
                        float(run_data.get("avg_trace_length", 0.0))
                        for run_data in instance_runs.values()
                    ]
                    or [0.0]
                ),
            }

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Wrote %s" % out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
