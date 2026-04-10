"""Kripke model and frame utilities for modal logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations, product
from types import MappingProxyType
from typing import FrozenSet, Hashable, Iterable, Literal, Mapping, Tuple, TypeVar, Union, cast


World = Hashable
RelationEdge = Tuple[World, World]
ModalLogic = Literal["K", "D", "T", "B", "K4", "S4", "S5"]
FrameLogicLike = Union[str, "FrameLogic"]
TItem = TypeVar("TItem", bound=Hashable)


class FrameLogic(str, Enum):
    K = "K"
    D = "D"
    T = "T"
    B = "B"
    K4 = "K4"
    S4 = "S4"
    S5 = "S5"


@dataclass(frozen=True)
class KripkeModel:
    """Immutable finite Kripke model."""

    worlds: FrozenSet[World]
    relation: FrozenSet[RelationEdge] = field(default_factory=frozenset)
    valuation: Mapping[str, FrozenSet[World]] = field(default_factory=dict, compare=False)
    _valuation_items: Tuple[Tuple[str, FrozenSet[World]], ...] = field(
        init=False, repr=False
    )
    _successors: Mapping[World, FrozenSet[World]] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        worlds = frozenset(self.worlds)
        if not worlds:
            raise ValueError("Kripke models must declare at least one world")
        normalized_relation = _normalize_relation(worlds, self.relation)
        normalized_valuation = _normalize_valuation(worlds, self.valuation)
        successors = {world: set() for world in worlds}

        for source, target in normalized_relation:
            successors[source].add(target)

        object.__setattr__(self, "worlds", worlds)
        object.__setattr__(self, "relation", normalized_relation)
        object.__setattr__(self, "valuation", MappingProxyType(dict(normalized_valuation)))
        object.__setattr__(
            self,
            "_valuation_items",
            tuple(sorted(normalized_valuation.items(), key=lambda item: item[0])),
        )
        object.__setattr__(
            self,
            "_successors",
            MappingProxyType(
                {world: frozenset(targets) for world, targets in successors.items()}
            ),
        )

    def successors(self, world: World) -> FrozenSet[World]:
        self._validate_world(world)
        return self._successors[world]

    def truth_set(self, atom: str) -> FrozenSet[World]:
        return self.valuation.get(atom, frozenset())

    def predecessors(self, world: World) -> FrozenSet[World]:
        self._validate_world(world)
        return frozenset(source for source, target in self.relation if target == world)

    def reachable_worlds(self, root: World) -> FrozenSet[World]:
        self._validate_world(root)
        pending = [root]
        seen = {root}

        while pending:
            world = pending.pop()
            for successor in self.successors(world):
                if successor not in seen:
                    seen.add(successor)
                    pending.append(successor)

        return frozenset(seen)

    def restrict_to_worlds(self, worlds: Iterable[World]) -> "KripkeModel":
        world_set = frozenset(worlds)
        if not world_set:
            raise ValueError("Submodels must contain at least one world")

        unknown_worlds = world_set - self.worlds
        if unknown_worlds:
            unknown_world = next(iter(unknown_worlds))
            raise ValueError(f"Unknown world: {unknown_world!r}")

        relation = frozenset(
            (source, target)
            for source, target in self.relation
            if source in world_set and target in world_set
        )
        valuation = {
            atom: frozenset(world for world in atom_worlds if world in world_set)
            for atom, atom_worlds in self.valuation.items()
        }
        return KripkeModel(worlds=world_set, relation=relation, valuation=valuation)

    def generated_submodel(self, root: World) -> "KripkeModel":
        return self.restrict_to_worlds(self.reachable_worlds(root))

    def is_reflexive(self) -> bool:
        return _is_reflexive(_ordered_worlds(self.worlds), self.relation)

    def is_symmetric(self) -> bool:
        return _is_symmetric(self.relation)

    def is_serial(self) -> bool:
        return _is_serial(_ordered_worlds(self.worlds), self.relation)

    def is_transitive(self) -> bool:
        return _is_transitive(_ordered_worlds(self.worlds), self.relation)

    def _validate_world(self, world: World) -> None:
        if world not in self.worlds:
            raise ValueError(f"Unknown world: {world!r}")


@dataclass(frozen=True)
class CountermodelWitness:
    model: KripkeModel
    world: World


@dataclass(frozen=True)
class ModelWitness:
    model: KripkeModel
    world: World


def _normalize_relation(
    worlds: FrozenSet[World], relation: Iterable[RelationEdge]
) -> FrozenSet[RelationEdge]:
    normalized_relation = set()

    for edge in relation:
        try:
            source, target = edge
        except (TypeError, ValueError) as error:
            raise ValueError("Each relation edge must contain exactly two worlds") from error

        if source not in worlds:
            raise ValueError(f"Relation source {source!r} is not a declared world")
        if target not in worlds:
            raise ValueError(f"Relation target {target!r} is not a declared world")

        normalized_relation.add((source, target))

    return frozenset(normalized_relation)


def _normalize_valuation(
    worlds: FrozenSet[World], valuation: Mapping[str, Iterable[World]]
) -> Mapping[str, FrozenSet[World]]:
    normalized_valuation = {}

    for atom, assigned_worlds in valuation.items():
        if not isinstance(atom, str):
            raise ValueError(f"Valuation atoms must be strings, got {atom!r}")

        world_set = frozenset(assigned_worlds)
        unknown_worlds = world_set - worlds
        if unknown_worlds:
            unknown_world = next(iter(unknown_worlds))
            raise ValueError(
                f"Valuation for atom {atom!r} mentions unknown world {unknown_world!r}"
            )

        normalized_valuation[atom] = world_set

    return normalized_valuation


def _enumerate_models(
    atoms: Iterable[str], logic: FrameLogicLike, max_worlds: int
) -> Iterable[KripkeModel]:
    normalized_logic = _normalize_logic(logic)
    if max_worlds < 1:
        raise ValueError("max_worlds must be at least 1")

    atom_list = tuple(sorted(set(atoms)))
    for world_count in range(1, max_worlds + 1):
        worlds = tuple(f"w{index}" for index in range(world_count))
        for relation in _enumerate_relations(worlds, normalized_logic):
            for valuation in _enumerate_valuations(worlds, atom_list):
                yield KripkeModel(
                    worlds=frozenset(worlds),
                    relation=relation,
                    valuation=valuation,
                )


def _enumerate_relations(
    worlds: Tuple[World, ...], logic: ModalLogic
) -> Iterable[FrozenSet[RelationEdge]]:
    all_edges = tuple((source, target) for source in worlds for target in worlds)
    for edge_subset in _powerset(all_edges):
        relation = frozenset(edge_subset)
        if _relation_satisfies_logic(worlds, relation, logic):
            yield relation


def _enumerate_valuations(
    worlds: Tuple[World, ...], atoms: Tuple[str, ...]
) -> Iterable[Mapping[str, FrozenSet[World]]]:
    if not atoms:
        yield {}
        return

    world_subsets = tuple(frozenset(subset) for subset in _powerset(worlds))
    for assignment in product(world_subsets, repeat=len(atoms)):
        yield {atom: truth_set for atom, truth_set in zip(atoms, assignment)}


def _powerset(items: Tuple[TItem, ...]) -> Iterable[Tuple[TItem, ...]]:
    for subset_size in range(len(items) + 1):
        yield from combinations(items, subset_size)


def _relation_satisfies_logic(
    worlds: Tuple[World, ...], relation: FrozenSet[RelationEdge], logic: ModalLogic
) -> bool:
    if logic == "K":
        return True
    if logic == "D":
        return _is_serial(worlds, relation)
    if logic == "T":
        return _is_reflexive(worlds, relation)
    if logic == "B":
        return _is_reflexive(worlds, relation) and _is_symmetric(relation)
    if logic == "K4":
        return _is_transitive(worlds, relation)
    if logic == "S4":
        return _is_reflexive(worlds, relation) and _is_transitive(worlds, relation)
    return (
        _is_reflexive(worlds, relation)
        and _is_symmetric(relation)
        and _is_transitive(worlds, relation)
    )


def _is_reflexive(worlds: Tuple[World, ...], relation: FrozenSet[RelationEdge]) -> bool:
    return all((world, world) in relation for world in worlds)


def _is_symmetric(relation: FrozenSet[RelationEdge]) -> bool:
    return all((target, source) in relation for source, target in relation)


def _is_serial(worlds: Tuple[World, ...], relation: FrozenSet[RelationEdge]) -> bool:
    return all(any(source == world for source, _ in relation) for world in worlds)


def _is_transitive(worlds: Tuple[World, ...], relation: FrozenSet[RelationEdge]) -> bool:
    return all(
        (source, target) not in relation
        or (target, successor) not in relation
        or (source, successor) in relation
        for source in worlds
        for target in worlds
        for successor in worlds
    )


def _normalize_logic(logic: FrameLogicLike) -> ModalLogic:
    if isinstance(logic, FrameLogic):
        normalized_logic = logic.value
    elif isinstance(logic, str):
        normalized_logic = logic.upper()
    else:
        raise ValueError(
            f"Unsupported modal logic {logic!r}; expected one of K, D, T, B, K4, S4, S5"
        )
    if normalized_logic not in {"K", "D", "T", "B", "K4", "S4", "S5"}:
        raise ValueError(
            f"Unsupported modal logic {logic!r}; expected one of K, D, T, B, K4, S4, S5"
        )
    return cast(ModalLogic, normalized_logic)


def validate_frame(model: KripkeModel, logic: FrameLogicLike) -> bool:
    return _relation_satisfies_logic(
        _ordered_worlds(model.worlds), model.relation, _normalize_logic(logic)
    )


def _ordered_worlds(worlds: FrozenSet[World]) -> Tuple[World, ...]:
    return tuple(sorted(worlds, key=repr))


__all__ = [
    "World",
    "RelationEdge",
    "ModalLogic",
    "FrameLogicLike",
    "FrameLogic",
    "KripkeModel",
    "CountermodelWitness",
    "ModelWitness",
    "validate_frame",
    "_enumerate_models",
    "_normalize_logic",
]
