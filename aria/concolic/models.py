"""Public data models for native Python concolic execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence

import z3


class DiagnosticSeverity(str, Enum):
    """Severity assigned to an instrumentation or symbolic-semantics issue."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class DiagnosticCode(str, Enum):
    """Stable machine-readable diagnostic codes."""

    UNSUPPORTED_INPUT = "unsupported-input"
    UNSUPPORTED_EXPRESSION = "unsupported-expression"
    UNSUPPORTED_OPERATION = "unsupported-operation"
    MISSING_SYMBOLIC_VALUE = "missing-symbolic-value"
    SYMBOLIC_CONCRETE_MISMATCH = "symbolic-concrete-mismatch"
    NESTED_SYMBOLIC_FRAME = "nested-symbolic-frame"
    SAMPLER_FAILURE = "sampler-failure"
    INVALID_SAMPLE = "invalid-sample"
    EXECUTION_TIMEOUT = "execution-timeout"
    WORKER_FAILURE = "worker-failure"
    UNSERIALIZABLE_OUTPUT = "unserializable-output"
    GENERATOR_TRUNCATED = "generator-truncated"
    OUTPUT_TRUNCATED = "output-truncated"
    RESOURCE_LIMIT_UNAVAILABLE = "resource-limit-unavailable"
    DEFINEDNESS_MISMATCH = "definedness-mismatch"
    CAMPAIGN_TIMEOUT = "campaign-timeout"
    COVERAGE_FAILURE = "coverage-failure"


@dataclass(frozen=True)
class SourceLocation:
    """A source range associated with an instrumented operation."""

    filename: str
    line: int
    column: int
    end_line: Optional[int] = None
    end_column: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "line": self.line,
            "column": self.column,
            "end_line": self.end_line,
            "end_column": self.end_column,
        }


@dataclass(frozen=True)
class Diagnostic:
    """A structured explanation of lost precision or unsupported behavior."""

    code: DiagnosticCode
    message: str
    severity: DiagnosticSeverity = DiagnosticSeverity.WARNING
    location: Optional[SourceLocation] = None
    branch_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "severity": self.severity.value,
            "location": self.location.to_dict() if self.location else None,
            "branch_id": self.branch_id,
        }


@dataclass(frozen=True)
class SymbolicTerm:
    """A symbolic expression and conditions required for it to be defined."""

    expression: Optional[Any]
    guards: Sequence[z3.BoolRef] = ()
    reason: Optional[str] = None

    @property
    def is_symbolic(self) -> bool:
        return self.expression is not None


@dataclass(frozen=True)
class StructuredValue:
    """A fixed-shape symbolic Python container."""

    kind: str
    children: Mapping[Any, Any]


@dataclass(frozen=True)
class PythonConstant:
    """An exact non-SMT Python value carried through calls and tuples."""

    value: Any


@dataclass(frozen=True)
class CoverageMetric:
    """One conventional covered/total percentage."""

    covered: int
    total: int
    percent: float

    @classmethod
    def from_counts(cls, covered: int, total: int) -> "CoverageMetric":
        percent = 100.0 if total == 0 else (100.0 * covered / total)
        return cls(covered, total, percent)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "covered": self.covered,
            "total": self.total,
            "percent": self.percent,
        }


@dataclass(frozen=True)
class FileCoverage:
    """Conventional coverage metrics for one Python source file."""

    filename: str
    lines: CoverageMetric
    statements: CoverageMetric
    functions: CoverageMetric
    branches: CoverageMetric
    missing_lines: Sequence[int] = ()
    missing_branches: Sequence[Sequence[int]] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "lines": self.lines.to_dict(),
            "statements": self.statements.to_dict(),
            "functions": self.functions.to_dict(),
            "branches": self.branches.to_dict(),
            "missing_lines": list(self.missing_lines),
            "missing_branches": [list(branch) for branch in self.missing_branches],
        }


@dataclass(frozen=True)
class CoverageReport:
    """Merged line, statement, function, and branch coverage for a campaign."""

    lines: CoverageMetric
    statements: CoverageMetric
    functions: CoverageMetric
    branches: CoverageMetric
    files: Mapping[str, FileCoverage]
    target_function: Optional[FileCoverage] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lines": self.lines.to_dict(),
            "statements": self.statements.to_dict(),
            "functions": self.functions.to_dict(),
            "branches": self.branches.to_dict(),
            "files": {
                filename: report.to_dict() for filename, report in self.files.items()
            },
            "target_function": (
                self.target_function.to_dict() if self.target_function else None
            ),
        }


@dataclass(frozen=True)
class CoverageSnapshot:
    """Cumulative conventional coverage after one campaign iteration."""

    iteration: int
    lines: float
    statements: float
    functions: float
    branches: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "lines": self.lines,
            "statements": self.statements,
            "functions": self.functions,
            "branches": self.branches,
        }


@dataclass(frozen=True)
class CoverageGate:
    """CI threshold evaluation for aggregate conventional coverage."""

    passed: bool
    required: Mapping[str, float]
    actual: Mapping[str, float]
    failures: Sequence[str] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "required": dict(self.required),
            "actual": dict(self.actual),
            "failures": list(self.failures),
        }


@dataclass(frozen=True)
class GapHotspot:
    """One ranked cause of lost symbolic exploration."""

    code: str
    message: str
    occurrences: int
    blocked_frontiers: int
    location: Optional[SourceLocation]
    branch_ids: Sequence[str]
    input_types: Mapping[str, str]
    example_inputs: Mapping[str, str]
    recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "occurrences": self.occurrences,
            "blocked_frontiers": self.blocked_frontiers,
            "location": self.location.to_dict() if self.location else None,
            "branch_ids": list(self.branch_ids),
            "input_types": dict(self.input_types),
            "example_inputs": dict(self.example_inputs),
            "recommendation": self.recommendation,
        }


@dataclass(frozen=True)
class GapReport:
    """Ranked symbolic-precision gaps for a complete campaign."""

    hotspots: Sequence[GapHotspot]
    blocked_frontiers: int
    diagnostics: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "blocked_frontiers": self.blocked_frontiers,
            "diagnostics": self.diagnostics,
            "hotspots": [hotspot.to_dict() for hotspot in self.hotspots],
        }


@dataclass(frozen=True)
class BranchObservation:
    """One dynamic occurrence of an instrumented Python branch."""

    branch_id: str
    occurrence: int
    taken: bool
    location: SourceLocation
    predicate: Optional[z3.BoolRef]
    guards: Sequence[z3.BoolRef]
    path_prefix: Sequence[z3.BoolRef]

    @property
    def dynamic_id(self) -> str:
        return f"{self.branch_id}#{self.occurrence}"

    def frontier_formula(self) -> Optional[z3.BoolRef]:
        """Return a formula reaching this branch and taking its other edge."""
        if self.predicate is None:
            return None
        desired = z3.Not(self.predicate) if self.taken else self.predicate
        return z3.simplify(z3.And(*self.path_prefix, *self.guards, desired))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "branch_id": self.branch_id,
            "dynamic_id": self.dynamic_id,
            "occurrence": self.occurrence,
            "taken": self.taken,
            "location": self.location.to_dict(),
            "predicate": self.predicate.sexpr() if self.predicate is not None else None,
            "guards": [guard.sexpr() for guard in self.guards],
            "path_prefix": [constraint.sexpr() for constraint in self.path_prefix],
        }


class ExecutionOutcome(str, Enum):
    """How one concrete execution ended."""

    RETURNED = "returned"
    RAISED = "raised"
    TIMED_OUT = "timed-out"


@dataclass
class ExecutionTrace:
    """Complete trace and artifacts for one generated input."""

    inputs: Dict[str, Any]
    outcome: ExecutionOutcome
    branches: List[BranchObservation] = field(default_factory=list)
    diagnostics: List[Diagnostic] = field(default_factory=list)
    return_value: Any = None
    exception_type: Optional[str] = None
    exception_message: Optional[str] = None
    elapsed_seconds: float = 0.0
    stdout: str = ""
    stderr: str = ""
    input_symbols: Dict[str, z3.ExprRef] = field(default_factory=dict, repr=False)
    input_paths: Dict[str, tuple[str, tuple[Any, ...]]] = field(
        default_factory=dict, repr=False
    )
    coverage_data: Optional[bytes] = field(default=None, repr=False)

    @property
    def path_signature(self) -> str:
        return "/".join(
            f"{branch.dynamic_id}:{int(branch.taken)}" for branch in self.branches
        )

    @property
    def covered_edges(self) -> set[tuple[str, bool]]:
        return {(branch.branch_id, branch.taken) for branch in self.branches}

    @property
    def is_sound(self) -> bool:
        unsound_codes = {
            DiagnosticCode.SYMBOLIC_CONCRETE_MISMATCH,
        }
        return not any(
            diagnostic.code in unsound_codes for diagnostic in self.diagnostics
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "inputs": self.inputs,
            "outcome": self.outcome.value,
            "branches": [branch.to_dict() for branch in self.branches],
            "diagnostics": [diagnostic.to_dict() for diagnostic in self.diagnostics],
            "return_value": _json_safe(self.return_value),
            "exception_type": self.exception_type,
            "exception_message": self.exception_message,
            "elapsed_seconds": self.elapsed_seconds,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "path_signature": self.path_signature,
            "is_sound": self.is_sound,
        }

    def to_wire(self) -> Dict[str, Any]:
        """Return a process-safe representation that preserves Z3 formulas."""
        return {
            **self.to_dict(),
            "return_value": self.return_value,
            "input_symbol_sorts": {
                name: _sort_name(symbol) for name, symbol in self.input_symbols.items()
            },
            "input_paths": {
                name: (root, list(path))
                for name, (root, path) in self.input_paths.items()
            },
            "coverage_data": self.coverage_data,
        }

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "ExecutionTrace":
        """Reconstruct a trace received from an isolated worker process."""
        input_symbols = {
            name: _symbol_for_sort(name, sort_name)
            for name, sort_name in payload.get("input_symbol_sorts", {}).items()
        }
        input_paths = {
            name: (value[0], tuple(value[1]))
            for name, value in payload.get("input_paths", {}).items()
        }
        branches = []
        for branch_payload in payload.get("branches", []):
            location = SourceLocation(**branch_payload["location"])
            predicate = _parse_bool_expression(
                branch_payload.get("predicate"), input_symbols
            )
            guards = tuple(
                _parse_required_bool_expression(item, input_symbols)
                for item in branch_payload.get("guards", [])
            )
            prefix = tuple(
                _parse_required_bool_expression(item, input_symbols)
                for item in branch_payload.get("path_prefix", [])
            )
            branches.append(
                BranchObservation(
                    branch_id=branch_payload["branch_id"],
                    occurrence=int(branch_payload["occurrence"]),
                    taken=bool(branch_payload["taken"]),
                    location=location,
                    predicate=predicate,
                    guards=guards,
                    path_prefix=prefix,
                )
            )
        diagnostics = [
            Diagnostic(
                code=DiagnosticCode(item["code"]),
                message=item["message"],
                severity=DiagnosticSeverity(item["severity"]),
                location=(
                    SourceLocation(**item["location"]) if item.get("location") else None
                ),
                branch_id=item.get("branch_id"),
            )
            for item in payload.get("diagnostics", [])
        ]
        return cls(
            inputs=dict(payload["inputs"]),
            outcome=ExecutionOutcome(payload["outcome"]),
            branches=branches,
            diagnostics=diagnostics,
            return_value=payload.get("return_value"),
            exception_type=payload.get("exception_type"),
            exception_message=payload.get("exception_message"),
            elapsed_seconds=float(payload.get("elapsed_seconds", 0.0)),
            stdout=str(payload.get("stdout", "")),
            stderr=str(payload.get("stderr", "")),
            input_symbols=input_symbols,
            input_paths=input_paths,
            coverage_data=payload.get("coverage_data"),
        )


@dataclass(frozen=True)
class Frontier:
    """An alternative branch edge waiting to be sampled."""

    branch_id: str
    desired_taken: bool
    formula: z3.BoolRef
    input_symbols: Mapping[str, z3.ExprRef]
    input_paths: Mapping[str, tuple[str, tuple[Any, ...]]]
    base_inputs: Mapping[str, Any]
    depth: int

    @property
    def signature(self) -> str:
        base_key = tuple(
            sorted(
                (name, type(value).__qualname__, repr(value))
                for name, value in self.base_inputs.items()
            )
        )
        return (
            f"{self.branch_id}:{int(self.desired_taken)}:"
            f"{self.formula.sexpr()}:{base_key!r}"
        )


@dataclass
class ConcolicResult:
    """Aggregate result of a concolic exploration."""

    traces: List[ExecutionTrace]
    generated_inputs: List[Dict[str, Any]]
    exhausted: bool
    iterations: int
    pending_frontiers: int
    diagnostics: List[Diagnostic] = field(default_factory=list)
    search_stats: Dict[str, Any] = field(default_factory=dict)
    coverage: Optional[CoverageReport] = None
    coverage_timeline: Sequence[CoverageSnapshot] = ()
    coverage_gate: Optional[CoverageGate] = None
    gaps: Optional[GapReport] = None

    @property
    def covered_edges(self) -> set[tuple[str, bool]]:
        return (
            set().union(*(trace.covered_edges for trace in self.traces))
            if self.traces
            else set()
        )

    @property
    def failures(self) -> List[ExecutionTrace]:
        return [
            trace for trace in self.traces if trace.outcome == ExecutionOutcome.RAISED
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "traces": [trace.to_dict() for trace in self.traces],
            "generated_inputs": self.generated_inputs,
            "exhausted": self.exhausted,
            "iterations": self.iterations,
            "pending_frontiers": self.pending_frontiers,
            "covered_edges": sorted(
                [list(edge) for edge in self.covered_edges],
                key=lambda item: (item[0], item[1]),
            ),
            "failure_count": len(self.failures),
            "diagnostics": [diagnostic.to_dict() for diagnostic in self.diagnostics],
            "search_stats": self.search_stats,
            "coverage": self.coverage.to_dict() if self.coverage else None,
            "coverage_timeline": [
                snapshot.to_dict() for snapshot in self.coverage_timeline
            ],
            "coverage_gate": (
                self.coverage_gate.to_dict() if self.coverage_gate else None
            ),
            "gaps": self.gaps.to_dict() if self.gaps else None,
        }


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return repr(value)


def _sort_name(symbol: z3.ExprRef) -> str:
    if z3.is_bool(symbol):
        return "Bool"
    if z3.is_int(symbol):
        return "Int"
    if z3.is_real(symbol):
        return "Real"
    if z3.is_string(symbol):
        return "String"
    if z3.is_fp(symbol) and symbol.sort() == z3.Float64():
        return "Float64"
    raise ValueError(f"unsupported wire sort: {symbol.sort()}")


def _symbol_for_sort(name: str, sort_name: str) -> z3.ExprRef:
    factories = {
        "Bool": z3.Bool,
        "Int": z3.Int,
        "Real": z3.Real,
        "String": z3.String,
        "Float64": lambda value: z3.FP(value, z3.Float64()),
    }
    try:
        return factories[sort_name](name)
    except KeyError as exc:
        raise ValueError(f"unsupported wire sort: {sort_name}") from exc


def _parse_bool_expression(
    text: Optional[str], declarations: Mapping[str, z3.ExprRef]
) -> Optional[z3.BoolRef]:
    if text is None:
        return None
    return _parse_required_bool_expression(text, declarations)


def _parse_required_bool_expression(
    text: str, declarations: Mapping[str, z3.ExprRef]
) -> z3.BoolRef:
    parsed = z3.parse_smt2_string(f"(assert {text})", decls=dict(declarations))
    if len(parsed) != 1:
        raise ValueError(
            "wire expression did not contain exactly one Boolean assertion"
        )
    return parsed[0]
