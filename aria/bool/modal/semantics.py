"""Semantic evaluation for modal formulas."""

from __future__ import annotations

from typing import Iterable, Optional

from .formula import And, Atom, Box, Constant, Diamond, Formula, Iff, Implies, Not, Or
from .model import FrameLogicLike, KripkeModel, validate_frame


def satisfies(model: KripkeModel, world, formula: Formula) -> bool:
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
    if isinstance(formula, Iff):
        return satisfies(model, world, formula.left) == satisfies(
            model, world, formula.right
        )
    if isinstance(formula, Box):
        return all(satisfies(model, successor, formula.operand) for successor in model.successors(world))
    if isinstance(formula, Diamond):
        return any(satisfies(model, successor, formula.operand) for successor in model.successors(world))

    raise TypeError(f"Unsupported modal formula: {formula!r}")


def is_valid(
    model: KripkeModel, formula: Formula, logic: Optional[FrameLogicLike] = None
) -> bool:
    if logic is not None and not validate_frame(model, logic):
        raise ValueError(f"Model does not satisfy the frame conditions for {logic!r}")
    return all(satisfies(model, world, formula) for world in model.worlds)


def entails(
    model: KripkeModel,
    premises: Iterable[Formula],
    conclusion: Formula,
    logic: Optional[FrameLogicLike] = None,
) -> bool:
    if logic is not None and not validate_frame(model, logic):
        raise ValueError(f"Model does not satisfy the frame conditions for {logic!r}")

    premise_list = tuple(premises)
    for world in model.worlds:
        if all(satisfies(model, world, premise) for premise in premise_list):
            if not satisfies(model, world, conclusion):
                return False
    return True


__all__ = ["satisfies", "is_valid", "entails"]
