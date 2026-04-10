"""Solver-backed bounded search for modal witnesses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from z3 import And, Bool, BoolVal, Implies, Not, Or, Solver, is_true, sat

from .formula import (
    And as AndFormula,
    Atom,
    Box,
    Constant,
    Diamond,
    Formula,
    Iff,
    Implies as ImpliesFormula,
    Not as NotFormula,
    Or as OrFormula,
)
from .model import CountermodelWitness, FrameLogicLike, KripkeModel, ModelWitness, World, _normalize_logic
from .utils import subformulas


ConditionBuilder = Callable[[Mapping[Formula, Sequence[object]], int], object]


@dataclass(frozen=True)
class _Encoding:
    worlds: Tuple[World, ...]
    relation: Tuple[Tuple[object, ...], ...]
    atoms: Mapping[str, Tuple[object, ...]]
    values: Dict[Formula, Tuple[object, ...]]


def find_model_with_z3(
    formula: Formula,
    logic: FrameLogicLike,
    max_worlds: int,
) -> Optional[ModelWitness]:
    return _search_with_z3(
        formulas=(formula,),
        atoms=formula.atoms(),
        logic=logic,
        max_worlds=max_worlds,
        condition_builder=lambda values, index: values[formula][index],
        witness_type=ModelWitness,
    )


def find_countermodel_with_z3(
    formula: Formula,
    logic: FrameLogicLike,
    max_worlds: int,
) -> Optional[CountermodelWitness]:
    return _search_with_z3(
        formulas=(formula,),
        atoms=formula.atoms(),
        logic=logic,
        max_worlds=max_worlds,
        condition_builder=lambda values, index: Not(values[formula][index]),
        witness_type=CountermodelWitness,
    )


def find_entailment_countermodel_with_z3(
    premises: Iterable[Formula],
    conclusion: Formula,
    logic: FrameLogicLike,
    max_worlds: int,
) -> Optional[CountermodelWitness]:
    premise_list = tuple(premises)
    atoms = set(conclusion.atoms())
    for premise in premise_list:
        atoms |= premise.atoms()

    return _search_with_z3(
        formulas=premise_list + (conclusion,),
        atoms=atoms,
        logic=logic,
        max_worlds=max_worlds,
        condition_builder=lambda values, index: And(
            *[values[premise][index] for premise in premise_list],
            Not(values[conclusion][index]),
        ),
        witness_type=CountermodelWitness,
    )


def _search_with_z3(
    formulas: Sequence[Formula],
    atoms: Iterable[str],
    logic: FrameLogicLike,
    max_worlds: int,
    condition_builder: ConditionBuilder,
    witness_type,
):
    if max_worlds < 1:
        raise ValueError("max_worlds must be at least 1")

    normalized_logic = _normalize_logic(logic)
    for world_count in range(1, max_worlds + 1):
        encoding = _build_encoding(tuple(sorted(set(atoms))), world_count)
        solver = Solver()
        solver.add(_frame_constraints(encoding, normalized_logic))
        solver.add(_formula_constraints(encoding, formulas))

        world_conditions = [
            condition_builder(encoding.values, index) for index in range(world_count)
        ]
        solver.add(Or(*world_conditions))
        if solver.check() != sat:
            continue

        model = solver.model()
        kripke_model = _extract_model(encoding, model)
        for index, condition in enumerate(world_conditions):
            if is_true(model.evaluate(condition, model_completion=True)):
                return witness_type(model=kripke_model, world=encoding.worlds[index])
    return None


def _build_encoding(atom_names: Tuple[str, ...], world_count: int) -> _Encoding:
    worlds = tuple(f"w{index}" for index in range(world_count))
    relation = tuple(
        tuple(Bool(f"r_{source}_{target}") for target in range(world_count))
        for source in range(world_count)
    )
    atoms = {
        atom: tuple(Bool(f"atom_{atom}_{index}") for index in range(world_count))
        for atom in atom_names
    }
    values: Dict[Formula, Tuple[object, ...]] = {}
    return _Encoding(worlds=worlds, relation=relation, atoms=atoms, values=values)


def _formula_constraints(
    encoding: _Encoding, formulas: Sequence[Formula]
) -> List[object]:
    constraints: List[object] = []
    formula_set = set()
    for formula in formulas:
        formula_set |= set(subformulas(formula))
    all_formulas = sorted(formula_set, key=repr)
    for formula in all_formulas:
        encoding.values[formula] = tuple(
            Bool(f"holds_{abs(hash(formula))}_{index}")
            for index in range(len(encoding.worlds))
        )

    for formula in all_formulas:
        for index in range(len(encoding.worlds)):
            constraints.append(
                encoding.values[formula][index]
                == _formula_semantics(encoding, formula, index)
            )
    return constraints


def _formula_semantics(encoding: _Encoding, formula: Formula, index: int):
    if isinstance(formula, Constant):
        return BoolVal(formula.value)
    if isinstance(formula, Atom):
        return encoding.atoms[formula.name][index]
    if isinstance(formula, NotFormula):
        return Not(encoding.values[formula.operand][index])
    if isinstance(formula, AndFormula):
        return And(
            encoding.values[formula.left][index],
            encoding.values[formula.right][index],
        )
    if isinstance(formula, OrFormula):
        return Or(
            encoding.values[formula.left][index],
            encoding.values[formula.right][index],
        )
    if isinstance(formula, ImpliesFormula):
        return Implies(
            encoding.values[formula.left][index],
            encoding.values[formula.right][index],
        )
    if isinstance(formula, Iff):
        return (
            encoding.values[formula.left][index]
            == encoding.values[formula.right][index]
        )
    if isinstance(formula, Box):
        return And(
            *[
                Implies(
                    encoding.relation[index][target],
                    encoding.values[formula.operand][target],
                )
                for target in range(len(encoding.worlds))
            ]
        )
    if isinstance(formula, Diamond):
        return Or(
            *[
                And(
                    encoding.relation[index][target],
                    encoding.values[formula.operand][target],
                )
                for target in range(len(encoding.worlds))
            ]
        )
    raise TypeError(f"Unsupported modal formula: {formula!r}")


def _frame_constraints(encoding: _Encoding, logic: str) -> List[object]:
    constraints: List[object] = []
    world_count = len(encoding.worlds)
    relation = encoding.relation

    if logic in {"D"}:
        for source in range(world_count):
            constraints.append(Or(*relation[source]))
    if logic in {"T", "B", "S4", "S5"}:
        for world in range(world_count):
            constraints.append(relation[world][world])
    if logic in {"B", "S5"}:
        for source in range(world_count):
            for target in range(world_count):
                constraints.append(relation[source][target] == relation[target][source])
    if logic in {"K4", "S4", "S5"}:
        for source in range(world_count):
            for target in range(world_count):
                for successor in range(world_count):
                    constraints.append(
                        Implies(
                            And(relation[source][target], relation[target][successor]),
                            relation[source][successor],
                        )
                    )
    return constraints


def _extract_model(encoding: _Encoding, model) -> KripkeModel:
    relation = set()
    for source, source_world in enumerate(encoding.worlds):
        for target, target_world in enumerate(encoding.worlds):
            if is_true(
                model.evaluate(encoding.relation[source][target], model_completion=True)
            ):
                relation.add((source_world, target_world))

    valuation = {}
    for atom, worlds in encoding.atoms.items():
        truth_worlds = frozenset(
            encoding.worlds[index]
            for index, variable in enumerate(worlds)
            if is_true(model.evaluate(variable, model_completion=True))
        )
        valuation[atom] = truth_worlds

    return KripkeModel(
        worlds=frozenset(encoding.worlds),
        relation=frozenset(relation),
        valuation=valuation,
    )
