#!/usr/bin/env python3
"""Run manifest-driven native concolic baseline comparisons."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

SCHEMA = "aria.concolic.benchmarks/v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare seed-only and native concolic campaigns"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmarks/concolic/manifest.json"),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--filter", help="Run case IDs containing this text")
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"unsupported benchmark schema {payload.get('schema')!r}")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("benchmark manifest must contain a non-empty case list")
    return cases


def run_case(case: Mapping[str, Any], timeout: float) -> List[Dict[str, Any]]:
    return [
        run_mode(case, "baseline", 1, timeout),
        run_mode(case, "concolic", int(case.get("iterations", 100)), timeout),
    ]


def run_mode(
    case: Mapping[str, Any],
    mode: str,
    iterations: int,
    timeout: float,
) -> Dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "aria.cli.concolic_cli",
        str(case["target"]),
        "--seed",
        json.dumps(case["seeds"], separators=(",", ":")),
        "--max-iterations",
        str(iterations),
        "--samples-per-frontier",
        str(case.get("samples_per_frontier", 4)),
        "--campaign-timeout",
        str(timeout),
        "--coverage",
    ]
    for package in case.get("instrument_packages", []):
        command.extend(("--instrument-package", str(package)))
    for source in case.get("coverage_sources", []):
        command.extend(("--coverage-source", str(source)))
    if case.get("direct", False):
        command.append("--direct")

    started = time.perf_counter()
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout + 15.0,
    )
    elapsed = time.perf_counter() - started
    if completed.returncode not in {0, 1, 3}:
        return {
            "case": case["id"],
            "mode": mode,
            "status": "error",
            "elapsed_seconds": elapsed,
            "error": completed.stderr.strip() or completed.stdout.strip(),
        }
    payload = json.loads(completed.stdout)
    coverage = payload.get("coverage") or {}
    cache = payload.get("search_stats", {}).get("sampler_cache", {})
    return {
        "case": case["id"],
        "mode": mode,
        "status": "ok",
        "elapsed_seconds": elapsed,
        "executions": payload.get("iterations", 0),
        "unique_paths": len(
            {trace.get("path_signature", "") for trace in payload.get("traces", [])}
        ),
        "failures": payload.get("failure_count", 0),
        "frontier_queries": cache.get("misses", 0),
        "cache_hits": cache.get("hits", 0),
        "line_coverage": _percent(coverage, "lines"),
        "function_coverage": _percent(coverage, "functions"),
        "branch_coverage": _percent(coverage, "branches"),
    }


def _percent(coverage: Mapping[str, Any], metric: str) -> float:
    return float(coverage.get(metric, {}).get("percent", 0.0))


def write_json(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema": SCHEMA, "results": rows}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cases = load_manifest(args.manifest)
        if args.filter:
            cases = [case for case in cases if args.filter in str(case.get("id", ""))]
        if not cases:
            raise ValueError("no benchmark cases matched")
        rows = [row for case in cases for row in run_case(case, args.timeout)]
        write_json(args.output_json, rows)
        write_csv(args.output_csv, rows)
    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
        subprocess.TimeoutExpired,
    ) as exc:
        print(f"concolic benchmark error: {exc}", file=sys.stderr)
        return 2
    return 1 if any(row.get("status") != "ok" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
