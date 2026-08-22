"""Property-oracle, pytest, and Hypothesis integration helpers."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, List, Mapping, Optional, Sequence

from .engine import ConcolicOptions, NativeConcolicEngine
from .models import ConcolicResult, ExecutionTrace

PropertyOracle = Callable[[ExecutionTrace], Optional[bool]]


@dataclass(frozen=True)
class PropertyViolation:
    """One generated input that falsified a user property."""

    oracle: str
    inputs: Mapping[str, Any]
    message: str
    path_signature: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "oracle": self.oracle,
            "inputs": dict(self.inputs),
            "message": self.message,
            "path_signature": self.path_signature,
        }


@dataclass
class ConcolicPropertyReport:
    """Campaign result plus all property violations."""

    result: ConcolicResult
    violations: List[PropertyViolation]

    @property
    def passed(self) -> bool:
        return not self.violations


class ConcolicAssertionError(AssertionError):
    """Raised by the pytest-style assertion helper on generated failures."""

    def __init__(self, report: ConcolicPropertyReport) -> None:
        self.report = report
        details = "\n".join(
            f"  {violation.oracle}: {violation.message}; inputs={violation.inputs!r}"
            for violation in report.violations[:10]
        )
        super().__init__(
            f"{len(report.violations)} concolic property violation(s)\n{details}"
        )


def evaluate_properties(
    result: ConcolicResult,
    oracles: Sequence[PropertyOracle],
) -> ConcolicPropertyReport:
    violations: List[PropertyViolation] = []
    for trace in result.traces:
        for oracle in oracles:
            oracle_name = getattr(oracle, "__qualname__", repr(oracle))
            try:
                outcome = oracle(trace)
                if outcome is False:
                    raise AssertionError("property returned False")
            except Exception as exc:
                violations.append(
                    PropertyViolation(
                        oracle=oracle_name,
                        inputs=copy.deepcopy(trace.inputs),
                        message=str(exc) or "property assertion failed",
                        path_signature=trace.path_signature,
                    )
                )
    return ConcolicPropertyReport(result, violations)


def assert_concolic(
    target: Callable[..., Any],
    seeds: Sequence[Mapping[str, Any]],
    oracles: Sequence[PropertyOracle],
    options: Optional[ConcolicOptions] = None,
) -> ConcolicPropertyReport:
    """Run a campaign and fail a pytest test if any property is falsified."""
    report = evaluate_properties(
        NativeConcolicEngine(target, options).explore(seeds),
        oracles,
    )
    if not report.passed:
        raise ConcolicAssertionError(report)
    return report


async def assert_concolic_async(
    target: Callable[..., Any],
    seeds: Sequence[Mapping[str, Any]],
    oracles: Sequence[PropertyOracle],
    options: Optional[ConcolicOptions] = None,
) -> ConcolicPropertyReport:
    """Async equivalent of assert_concolic for coroutine targets."""
    result = await NativeConcolicEngine(target, options).aexplore(seeds)
    report = evaluate_properties(result, oracles)
    if not report.passed:
        raise ConcolicAssertionError(report)
    return report


def pytest_parameters(result: ConcolicResult):
    """Return deterministic pytest parameters for every generated input."""
    import pytest

    return [
        pytest.param(
            copy.deepcopy(trace.inputs),
            id=trace.path_signature or f"input-{index}",
        )
        for index, trace in enumerate(result.traces)
    ]


def hypothesis_strategy(result: ConcolicResult):
    """Build a Hypothesis sampled strategy from deduplicated generated inputs."""
    from hypothesis import strategies as st

    unique = []
    seen = set()
    for inputs in result.generated_inputs:
        key = repr(inputs)
        if key not in seen:
            seen.add(key)
            unique.append(copy.deepcopy(inputs))
    if not unique:
        raise ValueError("cannot create a Hypothesis strategy without generated inputs")
    return st.sampled_from(unique)
