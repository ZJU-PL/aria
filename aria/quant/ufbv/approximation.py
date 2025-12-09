"""
Approximation utilities and recursive traversal of Z3 formulas for UFBV.

This module contains the logic to:
- Track the maximum encountered bit-width across variables in a formula
- Determine quantifier kind at a position based on polarity
- Recreate Z3 ASTs while applying bit-width reduction to targeted variables

The functions here are pure in spirit except for maintaining a module-level
`max_bit_width` that is updated while traversing, mirroring the original design.
"""

from __future__ import annotations

import z3

from .enums import Quantification, Polarity, ReductionType
from .reduction_types import (
    zero_extension,
    one_extension,
    sign_extension,
    right_zero_extension,
    right_one_extension,
    right_sign_extension,
)


# Global tracker for the maximum bit width seen during traversal.
max_bit_width: int = 0


def approximate(formula: z3.AstRef, reduction_type: ReductionType, bit_places: int) -> z3.AstRef:
    """Apply a single bit-vector projection according to `reduction_type`.

    Only reduces bit-width when the current size is strictly greater than
    `bit_places` to avoid accidental widening.
    """
    if formula.size() > bit_places:
        if reduction_type == ReductionType.ZERO_EXTENSION:
            return zero_extension(formula, bit_places)
        if reduction_type == ReductionType.ONE_EXTENSION:
            return one_extension(formula, bit_places)
        if reduction_type == ReductionType.SIGN_EXTENSION:
            return sign_extension(formula, bit_places)
        if reduction_type == ReductionType.RIGHT_ZERO_EXTENSION:
            return right_zero_extension(formula, bit_places)
        if reduction_type == ReductionType.RIGHT_ONE_EXTENSION:
            return right_one_extension(formula, bit_places)
        if reduction_type == ReductionType.RIGHT_SIGN_EXTENSION:
            return right_sign_extension(formula, bit_places)
        raise ValueError("Unknown ReductionType")
    return formula


def recreate_vars(new_vars: list[z3.AstRef], quant: z3.QuantifierRef) -> None:
    """Populate `new_vars` with freshly created bound vars from `quant`.

    Preserves names and sorts, instantiating either `z3.BitVec` or `z3.Bool`.
    """
    for i in range(quant.num_vars()):
        name = quant.var_name(i)
        if z3.is_bv_sort(quant.var_sort(i)):
            size = quant.var_sort(i).size()
            new_vars.append(z3.BitVec(name, size))
        elif quant.var_sort(i).is_bool():
            new_vars.append(z3.Bool(name))
        else:
            raise ValueError("Unknown sort for bound variable: %r" % quant.var_sort(i))


def get_q_type(quant: z3.QuantifierRef, polarity: Polarity) -> Quantification:
    """Return the effective quantifier kind under the given polarity.

    Universal at positive polarity (and existential at negative) is treated as
    UNIVERSAL, and the opposite as EXISTENTIAL, matching Skolemization/duality.
    """
    if (quant.is_forall() and (polarity == Polarity.POSITIVE)) or (
        (not quant.is_forall()) and (polarity == Polarity.NEGATIVE)
    ):
        return Quantification.UNIVERSAL
    return Quantification.EXISTENTIAL


def update_vars(quant: z3.QuantifierRef, var_list: list[tuple[str, Quantification]], polarity: Polarity) -> tuple[list[z3.AstRef], z3.QuantifierRef]:
    """Recreate bound vars and extend `var_list` with their names/kinds.

    Also merges consecutive quantifiers with the same kind to keep traversal
    simple and avoid deep nesting.
    """
    new_vars: list[z3.AstRef] = []
    quantification = get_q_type(quant, polarity)

    for i in range(quant.num_vars()):
        var_list.append((quant.var_name(i), quantification))
    recreate_vars(new_vars, quant)

    # Merge subsequent quantifiers of the same shape
    while (isinstance(quant.body(), z3.QuantifierRef) and (
        (quant.is_forall() and quant.body().is_forall()) or
        ((not quant.is_forall()) and (not quant.body().is_forall()))
    )):
        for i in range(quant.body().num_vars()):
            var_list.append((quant.body().var_name(i), quantification))
        recreate_vars(new_vars, quant.body())
        quant = quant.body()

    return new_vars, quant


def qform_process(quant: z3.QuantifierRef,
                  var_list: list[tuple[str, Quantification]],
                  reduction_type: ReductionType,
                  q_type: Quantification,
                  bit_places: int,
                  polarity: Polarity) -> z3.AstRef:
    """Rebuild a quantifier node with a transformed body.

    The body is recursively processed; bound variables are recreated by name and
    sort to keep a proper Z3 AST.
    """
    new_vars, quant = update_vars(quant, var_list, polarity)
    new_body = rec_go(
        quant.body(),
        list(var_list),
        reduction_type,
        q_type,
        bit_places,
        polarity,
    )
    return z3.ForAll(new_vars, new_body) if quant.is_forall() else z3.Exists(new_vars, new_body)


def cform_process(node: z3.AstRef,
                  var_list: list[tuple[str, Quantification]],
                  reduction_type: ReductionType,
                  q_type: Quantification,
                  bit_places: int,
                  polarity: Polarity) -> z3.AstRef:
    """Process a compound node, adjusting polarity and rebuilding children.
    """
    new_children: list[z3.AstRef] = []
    var_list_copy = list(var_list)

    # Negation flips polarity
    if node.decl().name() == "not":
        polarity = Polarity(not polarity.value)

    # Implication flips polarity in the antecedent
    elif node.decl().name() == "=>":
        left_polarity = Polarity(not polarity.value)
        new_children.append(
            rec_go(node.children()[0], var_list_copy, reduction_type, q_type, bit_places, left_polarity)
        )
        new_children.append(
            rec_go(node.children()[1], var_list_copy, reduction_type, q_type, bit_places, polarity)
        )
        return z3.Implies(*new_children)

    # Regular children processing
    for child in node.children():
        new_children.append(
            rec_go(child, var_list_copy, reduction_type, q_type, bit_places, polarity)
        )

    # Normalize n-ary ops to canonical Z3 constructors
    if node.decl().name() == "and":
        return z3.And(*new_children)
    if node.decl().name() == "or":
        return z3.Or(*new_children)
    if node.decl().name() == "bvadd":
        acc = new_children[0]
        for ch in new_children[1:]:
            acc = acc + ch
        return acc
    return node.decl().__call__(*new_children)


def rec_go(node: z3.AstRef,
           var_list: list[tuple[str, Quantification]],
           reduction_type: ReductionType,
           q_type: Quantification,
           bit_places: int,
           polarity: Polarity) -> z3.AstRef:
    """Traverse `node` and apply approximations to targeted BV variables.

    Updates the global `max_bit_width` tracker when encountering BV variables.
    Only variables whose quantifier kind matches `q_type` are reduced.
    """
    # Constant
    if z3.is_const(node):
        return node

    # Bound variable reference
    if z3.is_var(node):
        order = -z3.get_var_index(node) - 1
        if isinstance(node, z3.BitVecRef):
            global max_bit_width
            if max_bit_width < node.size():
                max_bit_width = node.size()
            if var_list and var_list[order][1] == q_type:
                return approximate(node, reduction_type, bit_places)
        return node

    # Quantified form
    if isinstance(node, z3.QuantifierRef):
        return qform_process(node, list(var_list), reduction_type, q_type, bit_places, polarity)

    # Compound form
    return cform_process(node, list(var_list), reduction_type, q_type, bit_places, polarity)


def get_max_bit_width() -> int:
    """Return the maximum bit-width observed during the last traversal."""
    return max_bit_width


def extract_max_bits_for_formula(fml: z3.AstRef) -> int:
    """Compute maximal bit-width by a traversal without altering the formula."""
    # Use a benign setup that walks the tree to populate the tracker
    reduction_type = ReductionType.ONE_EXTENSION
    q_type = Quantification.EXISTENTIAL
    bit_places = 1
    polarity = Polarity.POSITIVE
    rec_go(fml, [], reduction_type, q_type, bit_places, polarity)
    return get_max_bit_width()


def next_approx(reduction_type: ReductionType, bit_places: int) -> tuple[ReductionType, int]:
    """Alternate extension side and advance bit-width search frontier.

    Swaps left/right by negating the enum value; grows by 1 then 2-step strides
    to cover both parities.
    """
    reduction_type = ReductionType(-reduction_type.value)
    if reduction_type.value < 0:
        bit_places = bit_places + 1 if bit_places == 1 else bit_places + 2
    return reduction_type, bit_places


__all__ = [
    "approximate",
    "recreate_vars",
    "get_q_type",
    "update_vars",
    "qform_process",
    "cform_process",
    "rec_go",
    "get_max_bit_width",
    "extract_max_bits_for_formula",
    "next_approx",
]
