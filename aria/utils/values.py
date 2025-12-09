"""Utils for manipulating values (especially BV and FP)
  - bool_to_BitVec: Convert a boolean expression to a 1-bit bitvector.
  - bv_log2: Compute the floor of log2 of a bitvector value.
  - zext_or_trunc: Zero-extend or truncate a bitvector to a target width.
  - ctlz: Count leading zeros in a bitvector.
  - cttz: Count trailing zeros in a bitvector.
  - ComputeNumSignBits: Compute the number of consecutive sign bits in a bitvector.
  - fpUEQ: Unordered floating-point equality.
  - fpMod: Floating-point modulo operation.
  - fpMod_using_fpRem: Implement fpMod using fpRem.
  - fpRem_trampoline: Trampoline to Z3's fpRem.
  - set_bit: Set the index:th bit of v to x, and return the new value.
  - twos_complement: Returns the 2-complemented value of val assuming bits word width
  - convert_smtlib_models_to_python_value: For converting SMT-LIB models to Python values
  - zero_extension: Set the rest of bits on the left to 0.
  - one_extension: Set the rest of bits on the left to 1.
  - sign_extension: Set the rest of bits on the left to the value of the sign bit.
  - right_zero_extension: Set the rest of bits on the right to 0.
  - right_one_extension: Set the rest of bits on the right to 1.
  - right_sign_extension: Set the rest of bits on the right to the value of the sign bit.
  - absolute_value_bv: Absolute value for bitvector encoding.
"""

import re
import z3
from z3 import BitVecVal, Concat, Extract

# Regular expression for extracting values from SMT-LIB strings
RE_GET_EXPR_VALUE_ALL = re.compile(
    r"\(([a-zA-Z0-9_]*)[ \n\s]*(#b[0-1]*|#x[0-9a-fA-F]*|[(]?_ bv[0-9]* [0-9]*|true|false|[-+]?[0-9]+|[-+]?[0-9]*\.[0-9]+|\"[^\"]*\")\)"
)

def bool_to_BitVec(b):
  """Convert a boolean expression to a 1-bit bitvector.

  Args:
      b: Z3 boolean expression

  Returns:
      1-bit bitvector: 1 if b is true, 0 if b is false
  """
  return z3.If(b, z3.BitVecVal(1, 1), z3.BitVecVal(0, 1))

def bv_log2(bitwidth, v):
  """Compute the floor of log2 of a bitvector value.

  Args:
      bitwidth: Width of the output bitvector
      v: Input bitvector value

  Returns:
      Bitvector of specified bitwidth representing floor(log2(v))
  """
  def rec(h, l):
    if h <= l:
      return z3.BitVecVal(l, bitwidth)
    mid = l+int((h-l)/2)
    return z3.If(z3.Extract(h,mid+1,v) != 0, rec(h, mid+1), rec(mid, l))
  return rec(v.size()-1, 0)


def zext_or_trunc(v, src, tgt):
  """Zero-extend or truncate a bitvector to a target width.

  Args:
      v: Input bitvector
      src: Source bitwidth
      tgt: Target bitwidth

  Returns:
      Bitvector of target width (zero-extended if tgt > src, truncated if tgt < src)
  """
  if tgt == src:
    return v
  if tgt > src:
    return z3.ZeroExt(tgt - src, v)

  return z3.Extract(tgt-1, 0, v)


def ctlz(output_width, v):
  """Count leading zeros in a bitvector.

  Args:
      output_width: Width of the output bitvector
      v: Input bitvector

  Returns:
      Bitvector representing the number of leading zeros in v
  """
  size = v.size()
  def rec(i):
    if i < 0:
      return z3.BitVecVal(size, output_width)
    return z3.If(z3.Extract(i,i,v) == z3.BitVecVal(1, 1),
              z3.BitVecVal(size-1-i, output_width),
              rec(i-1))
  return rec(size-1)


def cttz(output_width, v):
  """Count trailing zeros in a bitvector.

  Args:
      output_width: Width of the output bitvector
      v: Input bitvector

  Returns:
      Bitvector representing the number of trailing zeros in v
  """
  size = v.size()
  def rec(i):
    if i == size:
      return z3.BitVecVal(size, output_width)
    return z3.If(z3.Extract(i,i,v) == z3.BitVecVal(1, 1),
              z3.BitVecVal(i, output_width),
              rec(i+1))
  return rec(0)

def ComputeNumSignBits(bitwidth, v):
  """Compute the number of consecutive sign bits in a bitvector.

  This counts how many leading bits match the sign bit (the most significant bit).

  Args:
      bitwidth: Width of the output bitvector
      v: Input bitvector

  Returns:
      Bitvector representing the number of consecutive sign bits
  """
  size = v.size()
  size1 = size - 1
  sign = z3.Extract(size1, size1, v)

  def rec(i):
    if i < 0:
      return z3.BitVecVal(size, bitwidth)
    return z3.If(z3.Extract(i,i,v) == sign,
              rec(i-1),
              z3.BitVecVal(size1-i, bitwidth))
  return rec(size - 2)

def fpUEQ(x, y):
  """Unordered floating-point equality.

  Returns true if x == y or either operand is NaN.

  Args:
      x: First floating-point value
      y: Second floating-point value

  Returns:
      Boolean expression representing unordered equality
  """
  return z3.Or(z3.fpEQ(x,y), z3.fpIsNaN(x), z3.fpIsNaN(y))


# Z3 4.4 incorrectly implemented fpRem as fpMod, where fpMod(x,y) has the same
# sign as x. In particular fpMod(3,2) = 1, but fpRem(3,2) = -1.
def detect_fpMod():
  """Determine whether Z3's fpRem is correct, and set fpMod accordingly.
  """
  import logging
  log = logging.getLogger(__name__)
  log.debug('Setting fpMod')

  if z3.is_true(z3.simplify(z3.FPVal(3, z3.Float32()) % 2 < 0)):
    log.debug('Correct fpRem detected')
    fpMod.__code__ = fpMod_using_fpRem.__code__
  else:
    log.debug('fpRem = fpMod')
    fpMod.__code__ = fpRem_trampoline.__code__


# Wait until fpMod is called, then determine which implementation it should
# have. Subsequent calls will use that implementation directly.
def fpMod(x, y, ctx=None):
  """Floating-point modulo operation.

  Lazily determines the correct implementation based on Z3's behavior,
  then uses that implementation for subsequent calls.

  Args:
      x: Dividend
      y: Divisor
      ctx: Optional Z3 context

  Returns:
      Floating-point remainder with sign matching x
  """
  detect_fpMod()
  return fpMod(x, y, ctx)

# It would be great if this had a less complicated implementation
def fpMod_using_fpRem(x, y, ctx=None):
  """Implement fpMod using fpRem.

  Used when Z3's fpRem is correct. This converts fpRem to fpMod semantics.

  Args:
      x: Dividend
      y: Divisor
      ctx: Optional Z3 context

  Returns:
      Floating-point modulo with sign matching x
  """
  y = z3.fpAbs(y)
  z = z3.fpRem(z3.fpAbs(x), y, ctx)
  r = z3.If(z3.fpIsNegative(z), z + y, z, ctx) # does rounding mode matter here?
  return z3.If(
    z3.Not(z3.fpIsNegative(x) == z3.fpIsNegative(r), ctx), z3.fpNeg(r), r, ctx)

# synonym for fpRem, but located in this module (i.e., same globals in scope)
def fpRem_trampoline(x, y, ctx=None):
  """Trampoline to Z3's fpRem.

  Used when Z3's fpRem is actually fpMod. This is a synonym for fpRem
  located in this module so it has the same globals in scope.

  Args:
      x: Dividend
      y: Divisor
      ctx: Optional Z3 context (unused)

  Returns:
      Result of Z3's fpRem operation
  """
  return z3.fpRem(x, y)


# Bit Operations
def set_bit(v: int, index: int, x: int):
    """Set the index:th bit of v to x, and return the new value."""
    mask = 1 << index
    if x:
        v |= mask
    else:
        v &= ~mask
    return v


def twos_complement(val: int, bits: int):
    """Returns the 2-complemented value of val assuming bits word width"""
    if (val & (1 << (bits - 1))) != 0:  # if sign bit is set
        val = val - (2 ** bits)  # compute negative value
    return val


def convert_smtlib_models_to_python_value(v):
    """
    For converting SMT-LIB models to Python values
    Supports: boolean, bitvectors, integers, reals, strings
    """
    r = None
    if v == "true":
        r = True
    elif v == "false":
        r = False
    elif v.startswith("#b"):
        r = int(v[2:], 2)
    elif v.startswith("#x"):
        r = int(v[2:], 16)
    elif v.startswith("_ bv"):
        # Extract the number part between "_ bv" and the space
        parts = v[len("_ bv"):].split(" ")
        r = int(parts[0], 10)
    elif v.startswith("(_ bv"):
        v = v[len("(_ bv"):]
        r = int(v[: v.find(" ")], 10)
    elif v.startswith('"') and v.endswith('"'):
        r = v[1:-1]  # Remove quotes for string values
    elif "." in v:  # Real number
        r = float(v)
    elif v.lstrip('-+').isdigit():  # Integer (possibly with sign)
        r = int(v)

    assert r is not None
    return r


def zero_extension(formula: z3.BitVecRef, bit_places: int) -> z3.BitVecRef:
    """Set the rest of bits on the left to 0.
    """
    complement = BitVecVal(0, formula.size() - bit_places)
    formula = z3.Concat(complement, (Extract(bit_places - 1, 0, formula)))
    return formula


def one_extension(formula: z3.BitVecRef, bit_places: int):
    """Set the rest of bits on the left to 1.

    Args:
        formula: Input bitvector
        bit_places: Number of bits to preserve from the right

    Returns:
        Bitvector with leftmost bits set to 1, rightmost bit_places bits preserved
    """
    complement = BitVecVal(0, formula.size() - bit_places) - 1
    formula = Concat(complement, (Extract(bit_places - 1, 0, formula)))
    return formula


def sign_extension(formula: z3.BitVecRef, bit_places: int) -> z3.BitVecRef:
    """Set the rest of bits on the left to the value of the sign bit.

    Args:
        formula: Input bitvector
        bit_places: Number of bits to preserve from the right

    Returns:
        Bitvector with leftmost bits set to the sign bit value,
        rightmost bit_places bits preserved
    """
    sign_bit = Extract(bit_places - 1, bit_places - 1, formula)

    complement = sign_bit
    for _ in range(formula.size() - bit_places - 1):
        complement = Concat(sign_bit, complement)

    formula = Concat(complement, (Extract(bit_places - 1, 0, formula)))
    return formula


def right_zero_extension(formula: z3.BitVecRef, bit_places: int) -> z3.BitVecRef:
    """Set the rest of bits on the right to 0.

    Args:
        formula: Input bitvector
        bit_places: Number of bits to preserve from the left

    Returns:
        Bitvector with rightmost bits set to 0, leftmost bit_places bits preserved
    """
    complement = BitVecVal(0, formula.size() - bit_places)
    formula = Concat(Extract(formula.size() - 1,
                             formula.size() - bit_places,
                             formula),
                     complement)
    return formula


def right_one_extension(formula: z3.BitVecRef, bit_places: int) -> z3.BitVecRef:
    """Set the rest of bits on the right to 1.

    Args:
        formula: Input bitvector
        bit_places: Number of bits to preserve from the left

    Returns:
        Bitvector with rightmost bits set to 1, leftmost bit_places bits preserved
    """
    complement = BitVecVal(0, formula.size() - bit_places) - 1
    formula = Concat(Extract(formula.size() - 1,
                             formula.size() - bit_places,
                             formula),
                     complement)
    return formula


def right_sign_extension(formula: z3.BitVecRef, bit_places: int) -> z3.BitVecRef:
    """Set the rest of bits on the right to the value of the sign bit.

    Args:
        formula: Input bitvector
        bit_places: Number of bits to preserve from the left

    Returns:
        Bitvector with rightmost bits set to the sign bit value,
        leftmost bit_places bits preserved
    """
    sign_bit_position = formula.size() - bit_places
    sign_bit = Extract(sign_bit_position, sign_bit_position, formula)

    complement = sign_bit
    for _ in range(sign_bit_position - 1):
        complement = Concat(sign_bit, complement)

    formula = Concat(Extract(formula.size() - 1,
                             sign_bit_position,
                             formula),
                     complement)
    return formula


def absolute_value_bv(bv: z3.BitVecRef):
    """Compute absolute value of a bitvector using bit manipulation.

    Based on: https://graphics.stanford.edu/~seander/bithacks.html#IntegerAbs

    Algorithm:
        Desired behavior is by definition (bv < 0) ? -bv : bv
        Now let mask := bv >> (bv.size() - 1)
        Note because of sign extension:
            bv >> (bv.size() - 1) == (bv < 0) ? -1 : 0
        Recall:
            -x == ~x + 1 => ~(x - 1) == -(x - 1) -1 == -x
            ~x == -1^x
             x ==  0^x
        now if bv < 0 then -bv == -1^(bv - 1) == mask ^ (bv + mask)
        else bv == 0^(bv + 0) == mask^(bv + mask)
        hence for all bv, absolute_value(bv) == mask ^ (bv + mask)

    Args:
        bv: Input bitvector

    Returns:
        Bitvector representing the absolute value of bv
    """
    mask = bv >> (bv.size() - 1)
    return mask ^ (bv + mask)


def absolute_value_int(val):
    """Compute absolute value of an integer using Z3's conditional.

    Args:
        val: Z3 integer expression

    Returns:
        Z3 expression representing the absolute value of val
    """
    return z3.If(val >= 0, val, -val)
