"""
Backend dispatch table for arithmetic probabilistic inference.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import z3

from aria.prob.core.density import Density
from aria.prob.core.results import InferenceResult

from ._config import WMIMethod, WMIOptions
from ._exact_backend import _exact_discrete_mass
from ._sampling_backends import _bounded_support_monte_carlo, _importance_sampling


BackendFn = Callable[
    [z3.ExprRef, Density, WMIOptions, List[z3.ExprRef]],
    InferenceResult,
]


def _exact_discrete_backend(
    formula: z3.ExprRef, density: Density, options: WMIOptions, variables: List[z3.ExprRef]
) -> InferenceResult:
    del options
    return _exact_discrete_mass(formula, density, variables)


WMI_BACKENDS: Dict[WMIMethod, BackendFn] = {
    WMIMethod.EXACT_DISCRETE: _exact_discrete_backend,
    WMIMethod.BOUNDED_SUPPORT_MONTE_CARLO: _bounded_support_monte_carlo,
    WMIMethod.IMPORTANCE_SAMPLING: _importance_sampling,
}


__all__ = ["BackendFn", "WMI_BACKENDS"]
