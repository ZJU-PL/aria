"""Finite-model modal checking and bounded countermodel search."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from importlib import import_module
from itertools import combinations, product
from types import MappingProxyType
from typing import (
    FrozenSet,
    Hashable,
    Iterable,
    Literal,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
)


class Formula:
    """Base class for modal formulas."""

    def atoms(self) -> set[str]:
        raise NotImplementedError


@dataclass(frozen=True)
class Constant(Formula):
    value: bool

    def __post_init__(self) -> None:
        if not isinstance(self.value, bool):
            raise TypeError(f"Modal constants must be bools, got {self.value!r}")

    def atoms(self) -> set[str]:
        return set()


@dataclass(frozen=True)
class Atom(Formula):
    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError(f"Atom names must be strings, got {self.name!r}")

    def atoms(self) -> set[str]:
        return {self.name}


@dataclass(frozen=True)
class UnaryFormula(Formula):
    operand: Formula

    def __post_init__(self) -> None:
        if not isinstance(self.operand, Formula):
            raise TypeError(
                f"Unary modal operands must be Formula instances, got {self.operand!r}"
            )

    def atoms(self) -> set[str]:
        return self.operand.atoms()


@dataclass(frozen=True)
class Not(UnaryFormula):
    pass


@dataclass(frozen=True)
class BinaryFormula(Formula):
    left: Formula
    right: Formula

    def __post_init__(self) -> None:
        if not isinstance(self.left, Formula):
            raise TypeError(
                f"Binary modal left operand must be a Formula, got {self.left!r}"
            )
        if not isinstance(self.right, Formula):
            raise TypeError(
                f"Binary modal right operand must be a Formula, got {self.right!r}"
            )

    def atoms(self) -> set[str]:
        return self.left.atoms() | self.right.atoms()


@dataclass(frozen=True)
class And(BinaryFormula):
    pass


@dataclass(frozen=True)
class Or(BinaryFormula):
    pass


@dataclass(frozen=True)
class Implies(BinaryFormula):
    pass


@dataclass(frozen=True)
class Box(UnaryFormula):
    pass


@dataclass(frozen=True)
class Diamond(UnaryFormula):
    pass


World = Hashable
RelationEdge = Tuple[World, World]
ModalLogic = Literal["K", "T", "S4", "S5"]
FrameLogicLike = Union[str, "FrameLogic"]
TItem = TypeVar("TItem", bound=Hashable)


class FrameLogic(str, Enum):
    K = "K"
    T = "T"
    S4 = "S4"
    S5 = "S5"


@dataclass(frozen=True)
class KripkeModel:
    """Immutable finite Kripke model."""

    worlds: FrozenSet[World]
    relation: FrozenSet[RelationEdge] = field(default_factory=frozenset)
    valuation: Mapping[str, FrozenSet[World]] = field(
        default_factory=dict, compare=False
    )
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
        normalized_relation = self._normalize_relation(worlds)
        normalized_valuation = self._normalize_valuation(worlds)
        successors = {world: set() for world in worlds}

        for source, target in normalized_relation:
            successors[source].add(target)

        object.__setattr__(self, "worlds", worlds)
        object.__setattr__(self, "relation", normalized_relation)
        object.__setattr__(
            self,
            "valuation",
            MappingProxyType(dict(normalized_valuation)),
        )
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
        """Return the worlds accessible from ``world``."""

        self._validate_world(world)
        return self._successors[world]

    def truth_set(self, atom: str) -> FrozenSet[World]:
        """Return the worlds where ``atom`` is true."""

        return self.valuation.get(atom, frozenset())

    def is_reflexive(self) -> bool:
        return _is_reflexive(_ordered_worlds(self.worlds), self.relation)

    def is_symmetric(self) -> bool:
        return _is_symmetric(self.relation)

    def is_transitive(self) -> bool:
        return _is_transitive(_ordered_worlds(self.worlds), self.relation)

    def _validate_world(self, world: World) -> None:
        if world not in self.worlds:
            raise ValueError(f"Unknown world: {world!r}")

    def _normalize_relation(
        self, worlds: FrozenSet[World]
    ) -> FrozenSet[RelationEdge]:
        normalized_relation = set()

        for edge in self.relation:
            try:
                source, target = edge
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "Each relation edge must contain exactly two worlds"
                ) from error

            if source not in worlds:
                raise ValueError(
                    f"Relation source {source!r} is not a declared world"
                )
            if target not in worlds:
                raise ValueError(
                    f"Relation target {target!r} is not a declared world"
                )

            normalized_relation.add((source, target))

        return frozenset(normalized_relation)

    def _normalize_valuation(
        self, worlds: FrozenSet[World]
    ) -> Mapping[str, FrozenSet[World]]:
        normalized_valuation = {}

        for atom, assigned_worlds in self.valuation.items():
            if not isinstance(atom, str):
                raise ValueError(
                    f"Valuation atoms must be strings, got {atom!r}"
                )

            world_set = frozenset(assigned_worlds)
            unknown_worlds = world_set - worlds
            if unknown_worlds:
                unknown_world = next(iter(unknown_worlds))
                raise ValueError(
                    f"Valuation for atom {atom!r} mentions unknown world "
                    f"{unknown_world!r}"
                )

            normalized_valuation[atom] = world_set

        return normalized_valuation


@dataclass(frozen=True)
class CountermodelWitness:
    """A bounded finite countermodel witness rooted at one world."""

    model: KripkeModel
    world: World


def satisfies(model: KripkeModel, world: World, formula: Formula) -> bool:
    """Return whether ``formula`` holds at ``world`` in ``model``."""

    if world not in model.worlds:
        raise ValueError(f"Unknown world: {world!r}")

    if isinstance(formula, Constant):
        return formula.value
    if isinstance(formula, Atom):
        return world in model.truth_set(formula.name)
    if isinstance(formula, Not):
        return not satisfies(model, world, formula.operand)
    if isinstance(formula, And):
        return satisfies(model, world, formula.left) and satisfies(
            model, world, formula.right
        )
    if isinstance(formula, Or):
        return satisfies(model, world, formula.left) or satisfies(
            model, world, formula.right
        )
    if isinstance(formula, Implies):
        return (not satisfies(model, world, formula.left)) or satisfies(
            model, world, formula.right
        )
    if isinstance(formula, Box):
        return all(
            satisfies(model, successor, formula.operand)
            for successor in model.successors(world)
        )
    if isinstance(formula, Diamond):
        return any(
            satisfies(model, successor, formula.operand)
            for successor in model.successors(world)
        )

    raise TypeError(f"Unsupported modal formula: {formula!r}")


def is_valid(
    model: KripkeModel,
    formula: Formula,
    logic: Optional[FrameLogicLike] = None,
) -> bool:
    """Return whether ``formula`` holds at every world in ``model``."""

    if logic is not None and not validate_frame(model, logic):
        raise ValueError(f"Model does not satisfy the frame conditions for {logic!r}")

    return all(satisfies(model, world, formula) for world in model.worlds)


def entails(
    model: KripkeModel,
    premises: Iterable[Formula],
    conclusion: Formula,
    logic: Optional[FrameLogicLike] = None,
) -> bool:
    """Return whether the premises entail the conclusion in ``model``."""

    if logic is not None and not validate_frame(model, logic):
        raise ValueError(f"Model does not satisfy the frame conditions for {logic!r}")

    premise_list = tuple(premises)
    for world in model.worlds:
        if all(satisfies(model, world, premise) for premise in premise_list):
            if not satisfies(model, world, conclusion):
                return False
    return True


def find_countermodel(
    formula: Formula,
    logic: FrameLogicLike = "K",
    max_worlds: int = 1,
) -> Optional[CountermodelWitness]:
    """Search for a bounded finite countermodel to ``formula``.

    The search is exhaustive only over finite models up to ``max_worlds`` whose
    frames satisfy the selected modal logic. Returning ``None`` means that no
    countermodel was found within the bound.
    """

    if not isinstance(formula, Formula):
        raise TypeError(f"Expected a modal Formula, got {formula!r}")

    for model in _enumerate_models(formula.atoms(), logic, max_worlds):
        for world in _ordered_worlds(model.worlds):
            if not satisfies(model, world, formula):
                return CountermodelWitness(model=model, world=world)
    return None


def find_entailment_countermodel(
    premises: Iterable[Formula],
    conclusion: Formula,
    logic: FrameLogicLike = "K",
    max_worlds: int = 1,
) -> Optional[CountermodelWitness]:
    """Search for a bounded finite countermodel to an entailment.

    The returned witness contains a model and a world where every premise holds
    but the conclusion fails. Returning ``None`` means no such witness was found
    within the given finite search bound.
    """

    premise_list = tuple(premises)
    for premise in premise_list:
        if not isinstance(premise, Formula):
            raise TypeError(f"Expected modal premises, got {premise!r}")
    if not isinstance(conclusion, Formula):
        raise TypeError(f"Expected a modal Formula, got {conclusion!r}")

    atoms = set(conclusion.atoms())
    for premise in premise_list:
        atoms |= premise.atoms()

    for model in _enumerate_models(atoms, logic, max_worlds):
        for world in _ordered_worlds(model.worlds):
            if all(satisfies(model, world, premise) for premise in premise_list):
                if not satisfies(model, world, conclusion):
                    return CountermodelWitness(model=model, world=world)
    return None


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
    worlds: Tuple[World, ...],
    relation: FrozenSet[RelationEdge],
    logic: ModalLogic,
) -> bool:
    if logic == "K":
        return True
    if logic == "T":
        return _is_reflexive(worlds, relation)
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
            f"Unsupported modal logic {logic!r}; expected one of K, T, S4, S5"
        )
    if normalized_logic not in {"K", "T", "S4", "S5"}:
        raise ValueError(
            f"Unsupported modal logic {logic!r}; expected one of K, T, S4, S5"
        )
    return cast(ModalLogic, normalized_logic)


def validate_frame(model: KripkeModel, logic: FrameLogicLike) -> bool:
    """Return whether ``model`` satisfies the selected modal frame conditions."""

    return _relation_satisfies_logic(
        _ordered_worlds(model.worlds), model.relation, _normalize_logic(logic)
    )


def _ordered_worlds(worlds: FrozenSet[World]) -> Tuple[World, ...]:
    return tuple(sorted(worlds, key=repr))


__all__ = [
    "Formula",
    "Constant",
    "Atom",
    "Not",
    "And",
    "Or",
    "Implies",
    "Box",
    "Diamond",
    "FrameLogic",
    "KripkeModel",
    "CountermodelWitness",
    "ModalSyntaxError",
    "parse_formula",
    "eliminate_implications",
    "to_nnf",
    "satisfies",
    "is_valid",
    "entails",
    "validate_frame",
    "find_countermodel",
    "find_entailment_countermodel",
]


def __getattr__(name: str):
    if name == "ModalSyntaxError":
        from aria.bool.modal.parser import ModalSyntaxError

        return ModalSyntaxError
    if name == "parse_formula":
        from aria.bool.modal.parser import parse_formula

        return parse_formula
    if name == "eliminate_implications":
        from aria.bool.modal.normalization import eliminate_implications

        return eliminate_implications
    if name == "to_nnf":
        from aria.bool.modal.normalization import to_nnf

        return to_nnf
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
