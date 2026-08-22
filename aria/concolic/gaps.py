"""Rank and explain operations that block symbolic frontier generation."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from .models import ConcolicResult, GapHotspot, GapReport


def explain_gaps(result: ConcolicResult) -> GapReport:
    grouped: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    blocked_total = sum(
        branch.predicate is None for trace in result.traces for branch in trace.branches
    )
    diagnostic_count = 0

    for trace in result.traces:
        blocked_ids = {
            branch.branch_id for branch in trace.branches if branch.predicate is None
        }
        for branch in trace.branches:
            if branch.predicate is not None:
                continue
            key = (
                "opaque-branch",
                "branch predicate is opaque",
                branch.location.filename,
                branch.location.line,
            )
            entry = grouped.setdefault(
                key,
                {
                    "code": "opaque-branch",
                    "message": "branch predicate is opaque",
                    "occurrences": 0,
                    "blocked": 0,
                    "location": branch.location,
                    "branches": set(),
                    "types": {
                        name: type(value).__qualname__
                        for name, value in trace.inputs.items()
                    },
                    "examples": {
                        name: _bounded_repr(value)
                        for name, value in trace.inputs.items()
                    },
                },
            )
            entry["occurrences"] += 1
            entry["blocked"] += 1
            entry["branches"].add(branch.branch_id)
        for diagnostic in trace.diagnostics:
            diagnostic_count += 1
            location = diagnostic.location
            key = (
                diagnostic.code.value,
                diagnostic.message,
                location.filename if location else None,
                location.line if location else None,
            )
            entry = grouped.setdefault(
                key,
                {
                    "code": diagnostic.code.value,
                    "message": diagnostic.message,
                    "occurrences": 0,
                    "blocked": 0,
                    "location": location,
                    "branches": set(),
                    "types": {
                        name: type(value).__qualname__
                        for name, value in trace.inputs.items()
                    },
                    "examples": {
                        name: _bounded_repr(value)
                        for name, value in trace.inputs.items()
                    },
                },
            )
            entry["occurrences"] += 1
            if diagnostic.branch_id:
                entry["branches"].add(diagnostic.branch_id)
                if diagnostic.branch_id in blocked_ids:
                    entry["blocked"] += 1

    hotspots = [
        GapHotspot(
            code=entry["code"],
            message=entry["message"],
            occurrences=entry["occurrences"],
            blocked_frontiers=entry["blocked"],
            location=entry["location"],
            branch_ids=tuple(sorted(entry["branches"])),
            input_types=entry["types"],
            example_inputs=entry["examples"],
            recommendation=_recommendation(entry["code"], entry["message"]),
        )
        for entry in grouped.values()
    ]
    hotspots.sort(
        key=lambda hotspot: (
            -hotspot.blocked_frontiers,
            -hotspot.occurrences,
            hotspot.code,
            hotspot.message,
        )
    )
    return GapReport(tuple(hotspots), int(blocked_total), diagnostic_count)


def format_gap_report(report: GapReport) -> str:
    lines = [
        f"Blocked frontiers: {report.blocked_frontiers}",
        f"Diagnostics: {report.diagnostics}",
    ]
    for index, hotspot in enumerate(report.hotspots, 1):
        location = "unknown location"
        if hotspot.location:
            location = f"{hotspot.location.filename}:{hotspot.location.line}"
        lines.extend(
            [
                "",
                f"{index}. {location}",
                f"   code: {hotspot.code}",
                f"   operation: {hotspot.message}",
                f"   occurrences: {hotspot.occurrences}",
                f"   blocked frontiers: {hotspot.blocked_frontiers}",
                f"   input types: {dict(hotspot.input_types)!r}",
                f"   recommendation: {hotspot.recommendation}",
            ]
        )
    return "\n".join(lines)


def _recommendation(code: str, message: str) -> str:
    lowered = message.lower()
    if "call" in lowered or "method" in lowered:
        return "instrument the callee package or add a guarded symbolic model"
    if "attribute" in lowered or "descriptor" in lowered:
        return "instrument the owning class/property or add an object field model"
    if "input" in code:
        return "add a structured input encoding or domain-specific seed adapter"
    if "nested" in lowered or "frame" in lowered:
        return "add the owning module to --instrument-package"
    if "mismatch" in code:
        return "add a differential semantics test and refine the symbolic model"
    return "add an exact guarded model or extend AST/runtime instrumentation"


def _bounded_repr(value: Any, limit: int = 160) -> str:
    rendered = repr(value)
    return rendered if len(rendered) <= limit else rendered[: limit - 3] + "..."
