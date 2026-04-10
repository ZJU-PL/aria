"""Finite-model modal checking and bounded countermodel search."""

from aria.bool.modal.formula import (
    And,
    Atom,
    BinaryFormula,
    Box,
    Constant,
    Diamond,
    Formula,
    Iff,
    Implies,
    Not,
    Or,
    UnaryFormula,
)
from aria.bool.modal.model import (
    CountermodelWitness,
    FrameLogic,
    FrameLogicLike,
    KripkeModel,
    ModalLogic,
    ModelWitness,
    RelationEdge,
    World,
    validate_frame,
)
from aria.bool.modal.normalization import eliminate_implications, simplify, to_nnf
from aria.bool.modal.parser import ModalSyntaxError, parse_formula
from aria.bool.modal.search import (
    SearchBackend,
    find_countermodel,
    find_entailment_countermodel,
    find_model,
)
from aria.bool.modal.semantics import entails, is_valid, satisfies
from aria.bool.modal.utils import format_formula, formula_size, modal_depth, subformulas

__all__ = [
    "Formula",
    "Constant",
    "Atom",
    "UnaryFormula",
    "Not",
    "BinaryFormula",
    "And",
    "Or",
    "Implies",
    "Iff",
    "Box",
    "Diamond",
    "World",
    "RelationEdge",
    "ModalLogic",
    "FrameLogicLike",
    "SearchBackend",
    "FrameLogic",
    "KripkeModel",
    "CountermodelWitness",
    "ModelWitness",
    "ModalSyntaxError",
    "parse_formula",
    "eliminate_implications",
    "simplify",
    "to_nnf",
    "formula_size",
    "modal_depth",
    "subformulas",
    "format_formula",
    "satisfies",
    "is_valid",
    "entails",
    "validate_frame",
    "find_model",
    "find_countermodel",
    "find_entailment_countermodel",
]
