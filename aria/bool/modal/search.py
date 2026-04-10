"""Bounded witness search for modal formulas."""

from __future__ import annotations

from importlib import import_module
from typing import Iterable, Literal, Optional, Tuple, cast

from .formula import Formula
from .model import (
    CountermodelWitness,
    FrameLogicLike,
    ModelWitness,
    _enumerate_models,
)
from .semantics import satisfies


SearchBackend = Literal["auto", "exhaustive", "z3"]


def find_model(
    formula: Formula,
    logic: FrameLogicLike = "K",
    max_worlds: int = 1,
    backend: SearchBackend = "auto",
) -> Optional[ModelWitness]:
    if not isinstance(formula, Formula):
        raise TypeError(f"Expected a modal Formula, got {formula!r}")

    witness = _find_model_with_backend(formula, logic, max_worlds, backend)
    if backend == "z3" or witness is not None:
        return witness

    for model in _enumerate_models(formula.atoms(), logic, max_worlds):
        for world in sorted(model.worlds, key=repr):
            if satisfies(model, world, formula):
                return ModelWitness(model=model, world=world)
    return None


def find_countermodel(
    formula: Formula,
    logic: FrameLogicLike = "K",
    max_worlds: int = 1,
    backend: SearchBackend = "auto",
) -> Optional[CountermodelWitness]:
    if not isinstance(formula, Formula):
        raise TypeError(f"Expected a modal Formula, got {formula!r}")

    witness = _find_countermodel_with_backend(formula, logic, max_worlds, backend)
    if backend == "z3" or witness is not None:
        return witness

    for model in _enumerate_models(formula.atoms(), logic, max_worlds):
        for world in sorted(model.worlds, key=repr):
            if not satisfies(model, world, formula):
                return CountermodelWitness(model=model, world=world)
    return None


def find_entailment_countermodel(
    premises: Iterable[Formula],
    conclusion: Formula,
    logic: FrameLogicLike = "K",
    max_worlds: int = 1,
    backend: SearchBackend = "auto",
) -> Optional[CountermodelWitness]:
    premise_list = tuple(premises)
    for premise in premise_list:
        if not isinstance(premise, Formula):
            raise TypeError(f"Expected modal premises, got {premise!r}")
    if not isinstance(conclusion, Formula):
        raise TypeError(f"Expected a modal Formula, got {conclusion!r}")

    atoms = set(conclusion.atoms())
    for premise in premise_list:
        atoms |= premise.atoms()

    witness = _find_entailment_countermodel_with_backend(
        premise_list, conclusion, logic, max_worlds, backend
    )
    if backend == "z3" or witness is not None:
        return witness

    for model in _enumerate_models(atoms, logic, max_worlds):
        for world in sorted(model.worlds, key=repr):
            if all(satisfies(model, world, premise) for premise in premise_list):
                if not satisfies(model, world, conclusion):
                    return CountermodelWitness(model=model, world=world)
    return None


def _find_model_with_backend(
    formula: Formula,
    logic: FrameLogicLike,
    max_worlds: int,
    backend: SearchBackend,
) -> Optional[ModelWitness]:
    if backend == "exhaustive":
        return None
    if backend not in {"auto", "z3"}:
        raise ValueError(
            f"Unsupported search backend {backend!r}; expected auto, exhaustive, or z3"
        )

    try:
        solver = import_module("aria.bool.modal.solver")
    except ModuleNotFoundError:
        if backend == "z3":
            raise
        return None

    return cast(
        Optional[ModelWitness],
        solver.find_model_with_z3(formula, logic=logic, max_worlds=max_worlds),
    )


def _find_countermodel_with_backend(
    formula: Formula,
    logic: FrameLogicLike,
    max_worlds: int,
    backend: SearchBackend,
) -> Optional[CountermodelWitness]:
    if backend == "exhaustive":
        return None
    if backend not in {"auto", "z3"}:
        raise ValueError(
            f"Unsupported search backend {backend!r}; expected auto, exhaustive, or z3"
        )

    try:
        solver = import_module("aria.bool.modal.solver")
    except ModuleNotFoundError:
        if backend == "z3":
            raise
        return None

    return cast(
        Optional[CountermodelWitness],
        solver.find_countermodel_with_z3(formula, logic=logic, max_worlds=max_worlds),
    )


def _find_entailment_countermodel_with_backend(
    premises: Tuple[Formula, ...],
    conclusion: Formula,
    logic: FrameLogicLike,
    max_worlds: int,
    backend: SearchBackend,
) -> Optional[CountermodelWitness]:
    if backend == "exhaustive":
        return None
    if backend not in {"auto", "z3"}:
        raise ValueError(
            f"Unsupported search backend {backend!r}; expected auto, exhaustive, or z3"
        )

    try:
        solver = import_module("aria.bool.modal.solver")
    except ModuleNotFoundError:
        if backend == "z3":
            raise
        return None

    return cast(
        Optional[CountermodelWitness],
        solver.find_entailment_countermodel_with_z3(
            premises, conclusion, logic=logic, max_worlds=max_worlds
        ),
    )


__all__ = [
    "SearchBackend",
    "find_model",
    "find_countermodel",
    "find_entailment_countermodel",
]
