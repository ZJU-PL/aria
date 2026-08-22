"""Conventional coverage.py collection and campaign-level reporting."""

from __future__ import annotations

import ast
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import coverage

from .models import CoverageMetric, CoverageReport, FileCoverage


class CoverageSession:
    """Collect one target execution into a process-portable data blob."""

    def __init__(
        self,
        target_file: str,
        sources: Sequence[str] = (),
        include: Sequence[str] = (),
        omit: Sequence[str] = (),
    ) -> None:
        self.target_file = str(Path(target_file).resolve())
        self.sources = tuple(sources)
        self.include = tuple(include)
        self.omit = tuple(omit)
        self._coverage = _new_coverage(
            self.target_file,
            self.sources,
            self.include,
            self.omit,
        )

    def start(self) -> None:
        self._coverage.start()

    def stop(self) -> bytes:
        self._coverage.stop()
        data = self._coverage.get_data()
        try:
            return data.dumps()
        finally:
            data.close(force=True)


class CoverageAccumulator:
    """Merge direct or worker coverage blobs and calculate conventional totals."""

    def __init__(
        self,
        target_file: str,
        target_function: str,
        sources: Sequence[str] = (),
        include: Sequence[str] = (),
        omit: Sequence[str] = (),
    ) -> None:
        self.target_file = str(Path(target_file).resolve())
        self.target_function = target_function
        self.sources = tuple(sources)
        self.include = tuple(include)
        self.omit = tuple(omit)
        self._lines: Dict[str, set[int]] = defaultdict(set)
        self._arcs: Dict[str, set[tuple[int, int]]] = defaultdict(set)

    def add(self, data: Optional[bytes]) -> None:
        if not data:
            return
        partial = coverage.CoverageData(no_disk=True)
        try:
            partial.loads(data)
            for filename in partial.measured_files():
                self._lines[filename].update(partial.lines(filename) or ())
                self._arcs[filename].update(partial.arcs(filename) or ())
        finally:
            partial.close(force=True)

    def report(self) -> CoverageReport:
        reporter = _new_coverage(
            self.target_file,
            self.sources,
            self.include,
            self.omit,
        )
        report_data = reporter.get_data()
        if any(self._arcs.values()):
            report_data.add_arcs(
                {filename: sorted(arcs) for filename, arcs in self._arcs.items()}
            )
        elif self._lines:
            report_data.add_lines(
                {filename: sorted(lines) for filename, lines in self._lines.items()}
            )
        descriptor, report_path = tempfile.mkstemp(
            prefix="aria-concolic-coverage-", suffix=".json"
        )
        os.close(descriptor)
        try:
            reporter.json_report(outfile=report_path, pretty_print=False)
            payload = json.loads(Path(report_path).read_text(encoding="utf-8"))
        finally:
            report_data.close(force=True)
            try:
                os.unlink(report_path)
            except FileNotFoundError:
                pass
        return _coverage_report_from_json(
            payload,
            self.target_file,
            self.target_function,
        )


def _new_coverage(
    target_file: str,
    sources: Sequence[str],
    include: Sequence[str],
    omit: Sequence[str],
) -> coverage.Coverage:
    configured_include = list(include) if include and not sources else None
    if not sources and not configured_include:
        configured_include = [target_file]
    configured = coverage.Coverage(
        data_file=None,
        branch=True,
        config_file=False,
        source=list(sources) or None,
        include=configured_include,
        omit=list(omit) or None,
        messages=False,
    )
    configured.set_option(
        "run:disable_warnings",
        ["module-not-measured", "already-imported"],
    )
    return configured


def _coverage_report_from_json(
    payload: Mapping[str, Any],
    target_file: str,
    target_function: str,
) -> CoverageReport:
    files: Dict[str, FileCoverage] = {}
    total_functions = 0
    covered_functions = 0
    target_report: Optional[FileCoverage] = None

    for filename, entry in payload.get("files", {}).items():
        summary = entry["summary"]
        raw_function_entries = entry.get("functions")
        if raw_function_entries is None:
            raw_function_entries = _fallback_function_entries(filename, entry)
        function_entries = {
            name: details for name, details in raw_function_entries.items() if name
        }
        file_functions = len(function_entries)
        file_covered_functions = sum(
            bool(details.get("executed_lines")) for details in function_entries.values()
        )
        total_functions += file_functions
        covered_functions += file_covered_functions
        file_report = _file_coverage(
            filename,
            summary,
            file_covered_functions,
            file_functions,
            entry.get("missing_lines", []),
            entry.get("missing_branches", []),
        )
        files[filename] = file_report

        if Path(filename).resolve() == Path(target_file).resolve():
            details = function_entries.get(target_function)
            if details is not None:
                function_summary = details["summary"]
                target_report = _file_coverage(
                    f"{filename}:{target_function}",
                    function_summary,
                    int(bool(details.get("executed_lines"))),
                    1,
                    details.get("missing_lines", []),
                    details.get("missing_branches", []),
                )

    totals = payload.get("totals", {})
    total_statements = int(totals.get("num_statements", 0))
    covered_statements = int(totals.get("covered_lines", 0))
    total_branches = int(totals.get("num_branches", 0))
    covered_branches = int(totals.get("covered_branches", 0))
    statement_metric = CoverageMetric.from_counts(covered_statements, total_statements)
    return CoverageReport(
        lines=statement_metric,
        statements=statement_metric,
        functions=CoverageMetric.from_counts(covered_functions, total_functions),
        branches=CoverageMetric.from_counts(covered_branches, total_branches),
        files=files,
        target_function=target_report,
    )


def _file_coverage(
    filename: str,
    summary: Mapping[str, Any],
    covered_functions: int,
    total_functions: int,
    missing_lines: Iterable[int],
    missing_branches: Iterable[Sequence[int]],
) -> FileCoverage:
    total_statements = int(summary.get("num_statements", 0))
    covered_statements = int(summary.get("covered_lines", 0))
    total_branches = int(summary.get("num_branches", 0))
    covered_branches = int(summary.get("covered_branches", 0))
    statement_metric = CoverageMetric.from_counts(covered_statements, total_statements)
    return FileCoverage(
        filename=filename,
        lines=statement_metric,
        statements=statement_metric,
        functions=CoverageMetric.from_counts(covered_functions, total_functions),
        branches=CoverageMetric.from_counts(covered_branches, total_branches),
        missing_lines=tuple(int(line) for line in missing_lines),
        missing_branches=tuple(tuple(branch) for branch in missing_branches),
    )


def _fallback_function_entries(
    filename: str,
    file_entry: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Derive function regions when an older coverage JSON omits them."""
    try:
        source = Path(filename).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=filename)
    except (OSError, SyntaxError, UnicodeError):
        return {}

    regions = _collect_function_regions(tree)
    executed_lines = {int(line) for line in file_entry.get("executed_lines", [])}
    missing_lines = {int(line) for line in file_entry.get("missing_lines", [])}
    statement_lines = executed_lines | missing_lines
    executed_branches = [
        tuple(int(value) for value in branch)
        for branch in file_entry.get("executed_branches", [])
    ]
    missing_branches = [
        tuple(int(value) for value in branch)
        for branch in file_entry.get("missing_branches", [])
    ]

    def owner(line: int) -> Optional[str]:
        candidates = [region for region in regions if region[1] <= line <= region[2]]
        if not candidates:
            return None
        return min(candidates, key=lambda region: region[2] - region[1])[0]

    entries: Dict[str, Dict[str, Any]] = {}
    for name, start, end in regions:
        statements = {line for line in statement_lines if owner(line) == name}
        executed = sorted(statements & executed_lines)
        missing = sorted(statements & missing_lines)
        covered_arcs = [
            list(branch) for branch in executed_branches if owner(branch[0]) == name
        ]
        missing_arcs = [
            list(branch) for branch in missing_branches if owner(branch[0]) == name
        ]
        total_branches = len(covered_arcs) + len(missing_arcs)
        entries[name] = {
            "executed_lines": executed,
            "missing_lines": missing,
            "excluded_lines": [],
            "start_line": start,
            "executed_branches": covered_arcs,
            "missing_branches": missing_arcs,
            "summary": {
                "covered_lines": len(executed),
                "num_statements": len(statements),
                "num_branches": total_branches,
                "covered_branches": len(covered_arcs),
            },
        }
    return entries


def _collect_function_regions(tree: ast.AST) -> list[tuple[str, int, int]]:
    regions: list[tuple[str, int, int]] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.path: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.path.append(node.name)
            self.generic_visit(node)
            self.path.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._visit_function(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._visit_function(node)

        def _visit_function(self, node: Any) -> None:
            name = ".".join((*self.path, node.name))
            body_start = min(
                (getattr(statement, "lineno", node.lineno) for statement in node.body),
                default=node.lineno,
            )
            end = int(getattr(node, "end_lineno", body_start))
            regions.append((name, int(body_start), end))
            self.path.append(node.name)
            self.generic_visit(node)
            self.path.pop()

    Visitor().visit(tree)
    return regions
