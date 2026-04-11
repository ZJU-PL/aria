"""Fixed-point helpers for reachability relations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Callable,
    Dict,
    Generic,
    Iterable,
    Optional,
    Protocol,
    Sized,
    Tuple,
    TypeVar,
)

from aria.cflobdd.relation import Relation


class SupportsFixedPoint(Protocol):
    @property
    def variables(self) -> Tuple[str, ...]:
        ...

    @property
    def facts(self) -> Sized:
        ...


T = TypeVar("T", bound=SupportsFixedPoint)


@dataclass(frozen=True)
class FixpointResult(Generic[T]):
    relation: T
    iterations: int
    converged: bool
    statistics: Dict[str, int]


def least_fixed_point(
    seed: T,
    step: Callable[[T], T],
    max_iterations: int = 100,
) -> FixpointResult[T]:
    current = seed
    for iteration in range(1, max_iterations + 1):
        next_relation = step(current)
        if next_relation.variables != current.variables:
            raise ValueError(
                "fixed-point step changed relation variables "
                f"from {current.variables} to {next_relation.variables}"
            )
        if next_relation.facts == current.facts:
            return FixpointResult(
                relation=next_relation,
                iterations=iteration,
                converged=True,
                statistics={"facts": len(next_relation.facts)},
            )
        current = next_relation

    return FixpointResult(
        relation=current,
        iterations=max_iterations,
        converged=False,
        statistics={"facts": len(current.facts)},
    )


def reflexive_closure(
    relation: Relation,
    nodes: Iterable[int],
    name: Optional[str] = None,
) -> Relation:
    if len(relation.variables) != 2:
        raise ValueError("reflexive_closure requires a binary relation")
    identity = Relation.identity(
        nodes,
        source=relation.variables[0],
        target=relation.variables[1],
    )
    return relation.union(identity, name=name)


def transitive_closure(
    relation: Relation,
    max_iterations: int = 100,
    name: Optional[str] = None,
) -> FixpointResult:
    if len(relation.variables) != 2:
        raise ValueError("transitive_closure requires a binary relation")

    def step(current: Relation) -> Relation:
        extended = current.binary_compose(relation, name=name)
        return current.union(extended, name=name)

    return least_fixed_point(relation, step, max_iterations=max_iterations)


def reflexive_transitive_closure(
    relation: Relation,
    nodes: Iterable[int],
    max_iterations: int = 100,
    name: Optional[str] = None,
) -> FixpointResult:
    seed = reflexive_closure(relation, nodes, name=name)

    def step(current: Relation) -> Relation:
        extended = current.binary_compose(seed, name=name)
        return current.union(extended, name=name)

    return least_fixed_point(seed, step, max_iterations=max_iterations)
