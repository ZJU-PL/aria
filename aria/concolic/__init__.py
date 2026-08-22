"""Production-oriented native Python concolic testing for Aria."""

from .engine import (
    ConcolicEngine,
    ConcolicOptions,
    ExplorationStrategy,
    NativeConcolicEngine,
)
from .instrumentation import InstrumentationError, instrument_callable
from .invocation import build_call_arguments, invoke_target_async, invoke_target_sync
from .artifacts import ArtifactStore, ReplayMismatch, ReplayResult
from .models import (
    BranchObservation,
    ConcolicResult,
    CoverageMetric,
    CoverageGate,
    CoverageReport,
    CoverageSnapshot,
    Diagnostic,
    DiagnosticCode,
    DiagnosticSeverity,
    ExecutionOutcome,
    ExecutionTrace,
    FileCoverage,
    Frontier,
    SourceLocation,
    SymbolicTerm,
)
from .model_registry import (
    DEFAULT_MODEL_REGISTRY,
    SymbolicModelRegistry,
    register_function_model,
    register_method_model,
)
from .package import PackageInstrumentationReport, PackageInstrumentor
from .model_packs import (
    available_model_packs,
    install_default_model_packs,
    install_model_pack,
)
from .testing import (
    ConcolicAssertionError,
    ConcolicPropertyReport,
    PropertyViolation,
    assert_concolic,
    assert_concolic_async,
    evaluate_properties,
    hypothesis_strategy,
    pytest_parameters,
)
from .gaps import explain_gaps, format_gap_report

__all__ = [
    "BranchObservation",
    "ArtifactStore",
    "ConcolicEngine",
    "ConcolicOptions",
    "ConcolicResult",
    "CoverageMetric",
    "CoverageGate",
    "CoverageReport",
    "CoverageSnapshot",
    "Diagnostic",
    "DiagnosticCode",
    "DiagnosticSeverity",
    "ExecutionOutcome",
    "ExecutionTrace",
    "FileCoverage",
    "ExplorationStrategy",
    "Frontier",
    "InstrumentationError",
    "NativeConcolicEngine",
    "ReplayMismatch",
    "ReplayResult",
    "SourceLocation",
    "SymbolicTerm",
    "DEFAULT_MODEL_REGISTRY",
    "SymbolicModelRegistry",
    "register_function_model",
    "register_method_model",
    "PackageInstrumentationReport",
    "PackageInstrumentor",
    "available_model_packs",
    "install_default_model_packs",
    "install_model_pack",
    "ConcolicAssertionError",
    "ConcolicPropertyReport",
    "PropertyViolation",
    "assert_concolic",
    "assert_concolic_async",
    "evaluate_properties",
    "hypothesis_strategy",
    "pytest_parameters",
    "explain_gaps",
    "format_gap_report",
    "instrument_callable",
    "build_call_arguments",
    "invoke_target_async",
    "invoke_target_sync",
]
