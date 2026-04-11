"""Balanced call/return reachability helpers."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional

from aria.cflobdd.fixpoint import FixpointResult, least_fixed_point
from aria.cflobdd.relation import Relation


def balanced_reachability(
    intra: Relation,
    calls: Mapping[str, Relation],
    returns: Mapping[str, Relation],
    nodes: Iterable[int],
    include_identity: bool = True,
    max_iterations: int = 100,
    name: Optional[str] = None,
) -> FixpointResult:
    if len(intra.variables) != 2:
        raise ValueError("balanced_reachability requires binary relations")

    result_name = name if name is not None else "balanced"
    base = intra.with_wrapped_witness("balanced", {"rule": "intra"}, name=result_name)

    if include_identity:
        identity = Relation.identity(
            nodes,
            source=intra.variables[0],
            target=intra.variables[1],
            name=result_name,
        ).with_wrapped_witness("balanced", {"rule": "epsilon"}, name=result_name)
        base = base.union(identity, name=result_name)

    matched_labels = sorted(set(calls) & set(returns))

    def step(current: Relation) -> Relation:
        next_relation = base

        concat = current.binary_compose(current, name=result_name).with_wrapped_witness(
            "balanced", {"rule": "concat"}, name=result_name
        )
        next_relation = next_relation.union(concat, name=result_name)

        for label in matched_labels:
            summarized = calls[label].binary_compose(current, name=result_name)
            summarized = summarized.binary_compose(returns[label], name=result_name)
            summarized = summarized.with_wrapped_witness(
                "balanced",
                {"rule": "call_return", "label": label},
                name=result_name,
            )
            next_relation = next_relation.union(summarized, name=result_name)

        return next_relation

    return least_fixed_point(base, step, max_iterations=max_iterations)
