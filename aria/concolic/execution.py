"""Isolated target execution with hard wall-clock limits."""

from __future__ import annotations

import importlib
import asyncio
import contextlib
import io
import inspect
import multiprocessing
import pickle
import time
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Any, Dict, Mapping, Optional

from .models import (
    Diagnostic,
    DiagnosticCode,
    DiagnosticSeverity,
    ExecutionOutcome,
    ExecutionTrace,
)
from .coverage import CoverageSession


class WorkerConfigurationError(RuntimeError):
    """Raised when a target cannot be resolved safely inside a worker."""


@dataclass(frozen=True)
class IsolatedExecutor:
    """Resolve and trace an importable function in a child process."""

    module_name: str
    qualified_name: str
    timeout_seconds: float
    start_method: Optional[str] = None
    memory_limit_mb: Optional[int] = None
    max_output_chars: int = 100_000
    measure_coverage: bool = False
    target_file: Optional[str] = None
    coverage_sources: tuple[str, ...] = ()
    coverage_include: tuple[str, ...] = ()
    coverage_omit: tuple[str, ...] = ()
    instrument_packages: tuple[str, ...] = ()
    instrument_include: tuple[str, ...] = ()
    instrument_omit: tuple[str, ...] = ()
    instrument_import_submodules: bool = True

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if "<locals>" in self.qualified_name:
            raise WorkerConfigurationError(
                "isolated execution requires a module-level importable function"
            )
        if self.memory_limit_mb is not None and self.memory_limit_mb <= 0:
            raise ValueError("memory_limit_mb must be positive when provided")
        if self.max_output_chars <= 0:
            raise ValueError("max_output_chars must be positive")

    def trace(self, inputs: Mapping[str, Any]) -> ExecutionTrace:
        method = self.start_method or _default_start_method()
        context = multiprocessing.get_context(method)
        parent, child = context.Pipe(duplex=False)
        process = context.Process(
            target=_worker_main,
            args=(
                child,
                self.module_name,
                self.qualified_name,
                dict(inputs),
                self.memory_limit_mb,
                self.max_output_chars,
                self.measure_coverage,
                self.target_file,
                self.coverage_sources,
                self.coverage_include,
                self.coverage_omit,
                self.instrument_packages,
                self.instrument_include,
                self.instrument_omit,
                self.instrument_import_submodules,
            ),
            daemon=False,
            name="aria-concolic-worker",
        )
        started = time.perf_counter()
        try:
            process.start()
        except (OSError, TypeError, ValueError, pickle.PickleError) as exc:
            parent.close()
            child.close()
            return _worker_failure_trace(
                inputs,
                f"could not start isolated worker: {exc}",
                time.perf_counter() - started,
            )
        child.close()
        try:
            if parent.poll(self.timeout_seconds):
                payload = parent.recv()
                process.join(timeout=1.0)
                if payload.get("kind") == "trace":
                    return ExecutionTrace.from_wire(payload["trace"])
                return _worker_failure_trace(
                    inputs,
                    payload.get("message", "worker failed without a message"),
                    time.perf_counter() - started,
                )
            process.terminate()
            process.join(timeout=1.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=1.0)
            return ExecutionTrace(
                inputs=dict(inputs),
                outcome=ExecutionOutcome.TIMED_OUT,
                diagnostics=[
                    Diagnostic(
                        DiagnosticCode.EXECUTION_TIMEOUT,
                        (
                            f"target exceeded the {self.timeout_seconds:g}s "
                            "execution limit"
                        ),
                        DiagnosticSeverity.ERROR,
                    )
                ],
                elapsed_seconds=time.perf_counter() - started,
            )
        except (EOFError, OSError, pickle.PickleError) as exc:
            return _worker_failure_trace(
                inputs, str(exc), time.perf_counter() - started
            )
        finally:
            parent.close()
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)


def _worker_main(
    connection: Connection,
    module_name: str,
    qualified_name: str,
    inputs: Dict[str, Any],
    memory_limit_mb: Optional[int],
    max_output_chars: int,
    measure_coverage: bool,
    target_file: Optional[str],
    coverage_sources: tuple[str, ...],
    coverage_include: tuple[str, ...],
    coverage_omit: tuple[str, ...],
    instrument_packages: tuple[str, ...],
    instrument_include: tuple[str, ...],
    instrument_omit: tuple[str, ...],
    instrument_import_submodules: bool,
) -> None:
    coverage_session = None
    try:
        resource_diagnostic = _apply_memory_limit(memory_limit_mb)
        module = importlib.import_module(module_name)
        target: Any = module
        for component in qualified_name.split("."):
            target = getattr(target, component)
        # Import here so a spawn worker does not initialize the engine before
        # its target module is fully loaded.
        from .engine import ConcolicOptions, NativeConcolicEngine

        engine = NativeConcolicEngine(
            target,
            ConcolicOptions(
                isolate=False,
                instrument_packages=instrument_packages,
                instrument_include=instrument_include,
                instrument_omit=instrument_omit,
                instrument_import_submodules=instrument_import_submodules,
            ),
        )
        if measure_coverage and target_file is not None:
            coverage_session = CoverageSession(
                target_file,
                coverage_sources,
                coverage_include,
                coverage_omit,
            )
            coverage_session.start()
        stdout = _BoundedTextBuffer(max_output_chars)
        stderr = _BoundedTextBuffer(max_output_chars)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            trace = (
                asyncio.run(engine.atrace(inputs))
                if inspect.iscoroutinefunction(target)
                else engine.trace(inputs)
            )
        trace.stdout = stdout.getvalue()
        trace.stderr = stderr.getvalue()
        if coverage_session is not None:
            trace.coverage_data = coverage_session.stop()
            coverage_session = None
        if stdout.truncated or stderr.truncated:
            trace.diagnostics.append(
                Diagnostic(
                    DiagnosticCode.OUTPUT_TRUNCATED,
                    (
                        f"worker output exceeded the {max_output_chars} character "
                        "capture limit"
                    ),
                )
            )
        if resource_diagnostic is not None:
            trace.diagnostics.append(resource_diagnostic)
        payload = trace.to_wire()
        try:
            pickle.dumps(payload)
        except (pickle.PickleError, TypeError, AttributeError):
            payload["return_value"] = repr(trace.return_value)
            payload["diagnostics"].append(
                Diagnostic(
                    DiagnosticCode.UNSERIALIZABLE_OUTPUT,
                    (
                        "return value was replaced by repr() because it is not "
                        "process-serializable"
                    ),
                ).to_dict()
            )
        connection.send({"kind": "trace", "trace": payload})
    # The worker must report initialization failures, including SystemExit.
    except BaseException as exc:
        if coverage_session is not None:
            try:
                coverage_session.stop()
            except Exception:
                pass
        try:
            connection.send(
                {
                    "kind": "failure",
                    "message": (
                        f"{type(exc).__module__}.{type(exc).__qualname__}: {exc}"
                    ),
                }
            )
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


def _worker_failure_trace(
    inputs: Mapping[str, Any], message: str, elapsed_seconds: float
) -> ExecutionTrace:
    return ExecutionTrace(
        inputs=dict(inputs),
        outcome=ExecutionOutcome.RAISED,
        diagnostics=[
            Diagnostic(
                DiagnosticCode.WORKER_FAILURE,
                message,
                DiagnosticSeverity.ERROR,
            )
        ],
        exception_type="aria.concolic.WorkerFailure",
        exception_message=message,
        elapsed_seconds=elapsed_seconds,
    )


def _default_start_method() -> str:
    available = multiprocessing.get_all_start_methods()
    return "spawn" if "spawn" in available else available[0]


class _BoundedTextBuffer(io.TextIOBase):
    """A write-only text stream that never stores more than its configured cap."""

    def __init__(self, limit: int) -> None:
        super().__init__()
        self.limit = limit
        self._parts: list[str] = []
        self._size = 0
        self.truncated = False

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        if not isinstance(text, str):
            raise TypeError("text output must be str")
        available = self.limit - self._size
        if available > 0:
            stored = text[:available]
            self._parts.append(stored)
            self._size += len(stored)
        if len(text) > available:
            self.truncated = True
        return len(text)

    def getvalue(self) -> str:
        return "".join(self._parts)


def _apply_memory_limit(memory_limit_mb: Optional[int]) -> Optional[Diagnostic]:
    if memory_limit_mb is None:
        return None
    try:
        import resource

        requested = memory_limit_mb * 1024 * 1024
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        effective = (
            requested if hard == resource.RLIM_INFINITY else min(requested, hard)
        )
        resource.setrlimit(resource.RLIMIT_AS, (effective, hard))
        return None
    except (ImportError, AttributeError, OSError, ValueError) as exc:
        return Diagnostic(
            DiagnosticCode.RESOURCE_LIMIT_UNAVAILABLE,
            f"could not apply the {memory_limit_mb} MiB address-space limit: {exc}",
            DiagnosticSeverity.WARNING,
        )
