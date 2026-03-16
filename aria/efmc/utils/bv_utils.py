"""Facilities for bit-vector related operations"""

from enum import Enum

import z3
from z3 import BitVecVal, Concat, Extract
from aria.efmc.utils.z3_expr_utils import get_variables


class Signedness(Enum):
    SIGNED = 0
    UNSIGNED = 1
    UNKNOWN = 2


def zero_extension(formula: z3.BitVecRef, bit_places: int) -> z3.BitVecRef:
    """Set the rest of bits on the left to 0."""
    complement = BitVecVal(0, formula.size() - bit_places)
    formula = z3.Concat(complement, (Extract(bit_places - 1, 0, formula)))
    return formula


def one_extension(formula: z3.BitVecRef, bit_places: int) -> z3.BitVecRef:
    """Set the rest of bits on the left to 1."""
    complement = BitVecVal(0, formula.size() - bit_places) - 1
    formula = Concat(complement, (Extract(bit_places - 1, 0, formula)))
    return formula


def sign_extension(formula: z3.BitVecRef, bit_places: int) -> z3.BitVecRef:
    """Set the rest of bits on the left to the value of the sign bit."""
    sign_bit = Extract(bit_places - 1, bit_places - 1, formula)

    complement = sign_bit
    for _ in range(formula.size() - bit_places - 1):
        complement = Concat(sign_bit, complement)

    formula = Concat(complement, (Extract(bit_places - 1, 0, formula)))
    return formula


def right_zero_extension(formula: z3.BitVecRef, bit_places: int) -> z3.BitVecRef:
    """Set the rest of bits on the right to 0."""
    complement = BitVecVal(0, formula.size() - bit_places)
    formula = Concat(
        Extract(formula.size() - 1, formula.size() - bit_places, formula), complement
    )
    return formula


def right_one_extension(formula: z3.BitVecRef, bit_places: int) -> z3.BitVecRef:
    """Set the rest of bits on the right to 1."""
    complement = BitVecVal(0, formula.size() - bit_places) - 1
    formula = Concat(
        Extract(formula.size() - 1, formula.size() - bit_places, formula), complement
    )
    return formula


def right_sign_extension(formula: z3.BitVecRef, bit_places: int) -> z3.BitVecRef:
    """Set the rest of bits on the right to the value of the sign bit."""
    sign_bit_position = formula.size() - bit_places
    sign_bit = Extract(sign_bit_position, sign_bit_position, formula)

    complement = sign_bit
    for _ in range(sign_bit_position - 1):
        complement = Concat(sign_bit, complement)

    formula = Concat(
        Extract(formula.size() - 1, sign_bit_position, formula), complement
    )
    return formula


def get_signedness(formula: z3.BitVecRef) -> Signedness:
    """Get the signedness of bit-vector variables in a bit-vector formula.
    Returns UNSIGNED if unsigned comparisons are used,
    SIGNED if signed comparisons are used,
    and UNKNOWN if no clear signedness indicators are found.

    TODO: by LLM, to check if this is correct
      What if we have UG(x + 1, 3)? Shoud we regard x as unsigned?
      Need to understand the SMT-LIB2 stanard about the signedness so that we can make this file right!!!
    """
    # get all variables in the formula
    variables = get_variables(formula)

    def check_signedness_recursive(expr: z3.ExprRef) -> Signedness:
        if z3.is_const(expr):
            return Signedness.UNKNOWN

        # Check operation kind
        kind = expr.decl().kind()
        op_name = expr.decl().name()

        # Unsigned operations - check by operation name
        if op_name in ["bvult", "bvule", "bvugt", "bvuge"]:
            return Signedness.UNSIGNED

        # Signed operations - check by operation name
        if op_name in ["bvslt", "bvsle", "bvsgt", "bvsge"]:
            return Signedness.SIGNED

        # Recurse on children
        for child in expr.children():
            result = check_signedness_recursive(child)
            if result != Signedness.UNKNOWN:
                return result

        return Signedness.UNKNOWN

    return check_signedness_recursive(formula)
