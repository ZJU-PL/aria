"""Command-line interface for native Python concolic testing."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence

from aria.concolic import ConcolicEngine, ConcolicOptions, ExplorationStrategy
from aria.concolic.artifacts import ArtifactStore
from aria.concolic.gaps import format_gap_report


def _target(value: str) -> Any:
    if ":" not in value:
        raise argparse.ArgumentTypeError("target must use MODULE:QUALIFIED_NAME syntax")
    module_name, qualified_name = value.split(":", 1)
    try:
        target: Any = importlib.import_module(module_name)
        for component in qualified_name.split("."):
            target = getattr(target, component)
    except (ImportError, AttributeError) as exc:
        raise argparse.ArgumentTypeError(
            f"cannot import target {value!r}: {exc}"
        ) from exc
    if not callable(target):
        raise argparse.ArgumentTypeError(f"target {value!r} is not callable")
    return target


def _seeds(value: str) -> List[dict[str, Any]]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid seed JSON: {exc}") from exc
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
        return payload
    raise argparse.ArgumentTypeError("seed JSON must be an object or a list of objects")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aria-concolic",
        description="Native coverage-guided concolic testing for Python functions",
    )
    parser.add_argument("target", help="Import target as MODULE:QUALIFIED_NAME")
    parser.add_argument(
        "--seed",
        required=True,
        help=(
            "JSON object, or list of objects, mapping parameter names to "
            "concrete values"
        ),
    )
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--samples-per-frontier", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--solver-timeout", type=float, default=5.0)
    parser.add_argument("--campaign-timeout", type=float)
    parser.add_argument("--max-path-depth", type=int, default=256)
    parser.add_argument("--max-constraint-nodes", type=int, default=10_000)
    parser.add_argument("--max-inputs-per-path", type=int, default=2)
    parser.add_argument(
        "--expand-input-shapes",
        action="store_true",
        help="Generate bounded list/dict/set/bytes length variants from seeds",
    )
    parser.add_argument("--max-shape-variants", type=int, default=32)
    parser.add_argument(
        "--explain-gaps",
        action="store_true",
        help="Rank opaque operations and blocked symbolic frontiers",
    )
    parser.add_argument("--gaps-json", type=Path)
    parser.add_argument(
        "--instrument-package",
        action="append",
        default=[],
        help="Instrument a package and propagate constraints across its calls",
    )
    parser.add_argument("--instrument-include", action="append", default=[])
    parser.add_argument("--instrument-omit", action="append", default=[])
    parser.add_argument(
        "--no-import-submodules",
        action="store_true",
        help="Only instrument package modules that are already imported",
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Calculate conventional line, statement, function, and branch coverage",
    )
    parser.add_argument(
        "--coverage-source",
        action="append",
        default=[],
        help="Package or directory to include in coverage totals; repeatable",
    )
    parser.add_argument("--coverage-include", action="append", default=[])
    parser.add_argument("--coverage-omit", action="append", default=[])
    parser.add_argument(
        "--coverage-json",
        type=Path,
        help="Write the conventional coverage report as standalone JSON",
    )
    parser.add_argument("--coverage-timeline-json", type=Path)
    parser.add_argument("--coverage-timeline-csv", type=Path)
    parser.add_argument("--fail-under-lines", type=float)
    parser.add_argument("--fail-under-statements", type=float)
    parser.add_argument("--fail-under-functions", type=float)
    parser.add_argument("--fail-under-branches", type=float)
    parser.add_argument("--memory-limit-mb", type=int)
    parser.add_argument("--max-output-chars", type=int, default=100_000)
    parser.add_argument("--random-seed", type=int)
    parser.add_argument(
        "--strategy",
        choices=[strategy.value for strategy in ExplorationStrategy],
        default=ExplorationStrategy.COVERAGE.value,
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Run in this process instead of an isolated worker",
    )
    parser.add_argument(
        "--artifact", type=Path, help="Write a versioned JSON campaign artifact"
    )
    parser.add_argument(
        "--pytest-output",
        type=Path,
        help="Generate pytest regression tests for discovered exceptions",
    )
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument(
        "--pretty", action="store_true", help="Pretty-print JSON output"
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        target = _target(args.target)
        seeds = _seeds(args.seed)
        module_name, qualified_name = args.target.split(":", 1)
        coverage_requested = bool(
            args.coverage
            or args.coverage_json
            or args.coverage_timeline_json
            or args.coverage_timeline_csv
            or args.fail_under_lines is not None
            or args.fail_under_statements is not None
            or args.fail_under_functions is not None
            or args.fail_under_branches is not None
        )
        options = ConcolicOptions(
            max_iterations=args.max_iterations,
            samples_per_frontier=args.samples_per_frontier,
            execution_timeout=args.timeout,
            random_seed=args.random_seed,
            strategy=ExplorationStrategy(args.strategy),
            isolate=not args.direct,
            stop_on_error=args.stop_on_error,
            memory_limit_mb=args.memory_limit_mb,
            max_output_chars=args.max_output_chars,
            solver_timeout=args.solver_timeout,
            campaign_timeout=args.campaign_timeout,
            max_path_depth=args.max_path_depth,
            max_constraint_nodes=args.max_constraint_nodes,
            max_inputs_per_path=args.max_inputs_per_path,
            expand_input_shapes=args.expand_input_shapes,
            max_shape_variants=args.max_shape_variants,
            explain_gaps=args.explain_gaps or args.gaps_json is not None,
            measure_coverage=coverage_requested,
            coverage_sources=tuple(args.coverage_source),
            coverage_include=tuple(args.coverage_include),
            coverage_omit=tuple(args.coverage_omit),
            instrument_packages=tuple(args.instrument_package),
            instrument_include=tuple(args.instrument_include),
            instrument_omit=tuple(args.instrument_omit),
            instrument_import_submodules=not args.no_import_submodules,
            coverage_fail_under_lines=args.fail_under_lines,
            coverage_fail_under_statements=args.fail_under_statements,
            coverage_fail_under_functions=args.fail_under_functions,
            coverage_fail_under_branches=args.fail_under_branches,
        )
        engine = ConcolicEngine(target, options)
        result = (
            asyncio.run(engine.aexplore(seeds))
            if inspect.iscoroutinefunction(target)
            else engine.explore(seeds)
        )
    except (TypeError, ValueError, RuntimeError, argparse.ArgumentTypeError) as exc:
        print(f"aria-concolic: error: {exc}", file=sys.stderr)
        return 2

    store = ArtifactStore()
    if args.artifact:
        store.write(args.artifact, args.target, result)
    if args.pytest_output:
        store.write_pytest_regressions(
            args.pytest_output,
            module_name,
            qualified_name,
            result.failures,
        )
    if args.coverage_json:
        if result.coverage is None:
            print(
                "aria-concolic: error: --coverage-json requires --coverage",
                file=sys.stderr,
            )
            return 2
        args.coverage_json.parent.mkdir(parents=True, exist_ok=True)
        args.coverage_json.write_text(
            json.dumps(
                result.coverage.to_dict(),
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    if args.coverage_timeline_json:
        args.coverage_timeline_json.parent.mkdir(parents=True, exist_ok=True)
        args.coverage_timeline_json.write_text(
            json.dumps(
                [snapshot.to_dict() for snapshot in result.coverage_timeline],
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    if args.coverage_timeline_csv:
        args.coverage_timeline_csv.parent.mkdir(parents=True, exist_ok=True)
        rows = ["iteration,lines,statements,functions,branches"]
        rows.extend(
            (
                f"{snapshot.iteration},{snapshot.lines:.6f},"
                f"{snapshot.statements:.6f},{snapshot.functions:.6f},"
                f"{snapshot.branches:.6f}"
            )
            for snapshot in result.coverage_timeline
        )
        args.coverage_timeline_csv.write_text("\n".join(rows) + "\n", encoding="utf-8")
    if args.gaps_json:
        args.gaps_json.parent.mkdir(parents=True, exist_ok=True)
        args.gaps_json.write_text(
            json.dumps(
                result.gaps.to_dict() if result.gaps else {},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    if args.explain_gaps and result.gaps is not None:
        print(format_gap_report(result.gaps), file=sys.stderr)
    print(
        json.dumps(
            result.to_dict(),
            indent=2 if args.pretty else None,
            sort_keys=args.pretty,
            ensure_ascii=False,
        )
    )
    if result.coverage_gate is not None and not result.coverage_gate.passed:
        return 3
    return 1 if result.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
