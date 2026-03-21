"""
High-level probability and expectation queries.
"""

from __future__ import annotations

from dataclasses import replace
from typing import List, Optional, Sequence, Union

import z3
from pysat.formula import CNF

from aria.prob.core._helpers import merge_cnfs
from aria.prob.core.density import Density
from aria.prob.core.results import InferenceResult
from aria.prob.arithmetic.query import (
    probability as arithmetic_probability,
)
from aria.prob.boolean.base import LiteralWeights, WMCBackend, WMCOptions
from aria.prob.boolean.wmc import CompiledWMC, compile_wmc, wmc_count
from aria.prob.arithmetic.moments import moment, expectation, covariance, variance
from aria.prob.arithmetic.wmi import WMIOptions


def _strict_wmc_options(options: Optional[WMCOptions]) -> WMCOptions:
    opts = options or WMCOptions()
    if opts.strict_complements:
        return opts
    return replace(opts, strict_complements=True)


def _literal_sequence(value: Optional[Union[int, Sequence[int]]]) -> List[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    return [int(lit) for lit in value]


def probability(
    formula: Union[CNF, z3.ExprRef, int, Sequence[int]],
    model: Union[CompiledWMC, LiteralWeights, Density],
    evidence: Optional[Union[CNF, z3.ExprRef, int, Sequence[int]]] = None,
    options: Optional[Union[WMCOptions, WMIOptions]] = None,
) -> InferenceResult:
    """
    Compute P(formula | evidence) under a Boolean weighted model or arithmetic density.
    """

    if isinstance(model, CompiledWMC):
        if isinstance(formula, CNF):
            if evidence is not None and not isinstance(evidence, CNF):
                raise ValueError("CompiledWMC CNF queries require CNF evidence")
            return model.probability_cnf(formula, evidence_cnf=evidence)
        if isinstance(evidence, CNF):
            raise ValueError("CompiledWMC literal queries do not accept CNF evidence")
        return model.probability(
            query=_literal_sequence(formula),
            evidence=_literal_sequence(evidence),
        )

    if isinstance(formula, CNF):
        if not isinstance(model, dict):
            raise ValueError("CNF probability queries require a literal weight map")
        if evidence is not None and not isinstance(evidence, CNF):
            raise ValueError("CNF evidence must also be a CNF formula")

        opts = _strict_wmc_options(options if isinstance(options, WMCOptions) else None)
        if opts.backend == WMCBackend.DNNF:
            numerator_compiled = compile_wmc(
                merge_cnfs(formula, evidence), model, opts
            )
            numerator = numerator_compiled.count()
            if evidence is None:
                denominator = 1.0
            else:
                denominator = compile_wmc(evidence, model, opts).count()
            if denominator == 0.0:
                raise ValueError("Evidence CNF has zero probability under the weights")
            return InferenceResult(
                value=numerator / denominator,
                exact=True,
                backend="wmc-dnnf",
                stats={"numerator": numerator, "denominator": denominator},
                error_bound=0.0,
            )

        numerator = wmc_count(merge_cnfs(formula, evidence), model, opts)
        if evidence is None:
            denominator = 1.0
        else:
            denominator = wmc_count(evidence, model, opts)
        if denominator == 0.0:
            raise ValueError("Evidence CNF has zero probability under the weights")
        exact = opts.model_limit is None
        return InferenceResult(
            value=numerator / denominator,
            exact=exact,
            backend="wmc-enumeration",
            stats={
                "numerator": numerator,
                "denominator": denominator,
                "model_limit": opts.model_limit,
            },
            error_bound=None,
        )

    if not isinstance(formula, z3.ExprRef):
        raise ValueError("Arithmetic probability queries require a Z3 formula")
    if not isinstance(model, Density):
        raise ValueError("Arithmetic probability queries require a density model")

    wmi_options = options if isinstance(options, WMIOptions) else None
    return arithmetic_probability(formula, model, evidence=evidence, options=wmi_options)


def conditional_probability(
    query: Union[CNF, z3.ExprRef, int, Sequence[int]],
    evidence: Union[CNF, z3.ExprRef, int, Sequence[int]],
    model: Union[CompiledWMC, LiteralWeights, Density],
    options: Optional[Union[WMCOptions, WMIOptions]] = None,
) -> InferenceResult:
    return probability(query, model, evidence=evidence, options=options)
