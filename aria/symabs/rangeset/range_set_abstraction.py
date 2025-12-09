from __future__ import annotations

from typing import List, Tuple

import z3


def _bv_width(x: z3.ExprRef) -> int:
    if not z3.is_bv(x):
        raise TypeError("x must be a bit-vector")
    return x.size()


def _to_python_int(val: int, width: int, signed: bool) -> int:
    if not signed:
        return val
    # Interpret val as two's complement of given width
    sign_bit = 1 << (width - 1)
    mask = (1 << width) - 1
    v = val & mask
    return (v ^ sign_bit) - sign_bit


def _from_python_int(v: int, width: int) -> z3.BitVecNumRef:
    mask = (1 << width) - 1
    return z3.BitVecVal(v & mask, width)


def _assume_and_check(s: z3.Solver, assumptions: List[z3.ExprRef]) -> z3.CheckSatResult:
    s.push(); s.add(*assumptions); res = s.check(); s.pop()
    return res


def minimum(fml: z3.BoolRef, x: z3.BitVecRef, signed: bool = False) -> int:
    """Compute minimal integer value of bit-vector x satisfying fml.

    Returns a Python int under the chosen signedness.
    """
    width = _bv_width(x)
    s = z3.Solver()
    s.add(fml)

    # Build result bits from MSB to LSB as in the paper.
    result = 0
    for i in reversed(range(width)):
        bit_val_zero_assumptions: List[z3.BoolRef] = []

        # Impose previously fixed higher bits on x.
        for j in range(i + 1, width):
            bit = (result >> j) & 1
            bit_constraint = z3.Extract(j, j, x) == z3.BitVecVal(bit, 1)
            bit_val_zero_assumptions.append(bit_constraint)

        # Decide which polarity to try first
        prefer_zero = True
        if signed and i == width - 1:
            # For signed minimum, prefer MSB=1 (negative numbers)
            prefer_zero = False

        try_first = 0 if prefer_zero else 1
        try_expr = z3.Extract(i, i, x) == z3.BitVecVal(try_first, 1)
        res = _assume_and_check(s, bit_val_zero_assumptions + [try_expr])
        if res == z3.sat:
            # Keep first choice
            if try_first == 1:
                result |= (1 << i)
            continue
        # Otherwise flip
        flipped = 1 - try_first
        if flipped == 1:
            result |= (1 << i)

    return _to_python_int(result, width, signed)


def maximum(fml: z3.BoolRef, x: z3.BitVecRef, signed: bool = False) -> int:
    """Compute maximal integer value of bit-vector x satisfying fml."""
    width = _bv_width(x)
    s = z3.Solver()
    s.add(fml)

    result = 0
    for i in reversed(range(width)):
        assumptions: List[z3.BoolRef] = []
        for j in range(i + 1, width):
            bit = (result >> j) & 1
            assumptions.append(z3.Extract(j, j, x) == z3.BitVecVal(bit, 1))

        # For signed maximum, prefer MSB=0 (non-negative)
        prefer_one = True
        if signed and i == width - 1:
            prefer_one = False

        try_first = 1 if prefer_one else 0
        try_expr = z3.Extract(i, i, x) == z3.BitVecVal(try_first, 1)
        res = _assume_and_check(s, assumptions + [try_expr])
        if res == z3.sat:
            if try_first == 1:
                result |= (1 << i)
        else:
            flipped = 1 - try_first
            if flipped == 1:
                result |= (1 << i)

    return _to_python_int(result, width, signed)


def range_abstraction(
    fml: z3.BoolRef, x: z3.BitVecRef, signed: bool = False
) -> Tuple[int, int]:
    """Return (min, max) integers of x under fml with selected signedness."""
    # Clamp with comparator constraints to speed convergence is not required for min/max
    return minimum(fml, x, signed), maximum(fml, x, signed)


def _value(bv_int: int, width: int, signed: bool) -> int:
    return _to_python_int(bv_int, width, signed)


def _bounds_to_bv_constraints(
    x: z3.BitVecRef, lower_bits: int, upper_bits: int, signed: bool
) -> Tuple[z3.BoolRef, z3.BoolRef]:
    width = _bv_width(x)
    l = _from_python_int(lower_bits, width)
    u = _from_python_int(upper_bits, width)
    if signed:
        return (z3.SLE(l, x), z3.SLE(x, u))
    return (z3.ULE(l, x), z3.ULE(x, u))


def set_abstraction(
    fml: z3.BoolRef,
    x: z3.BitVecRef,
    signed: bool = False,
    steps: int = -1,
) -> List[Tuple[int, int]]:
    """Compute a set abstraction for x:

    - If steps < 0: exact (run until convergence)
    - If steps >= 0: run that many refinements; odd steps yield an over-approx,
      even steps yield an under-approx.
    Returns a list of inclusive intervals [(a, b), ...] in Python ints.
    """
    width = _bv_width(x)

    # Initial bounds respecting signedness
    if signed:
        # l = 100..0 (MSB=1 negative min), u = 011..1 (MSB=0)
        l_bits = 1 << (width - 1)
        u_bits = (1 << (width - 1)) - 1
    else:
        l_bits = 0
        u_bits = (1 << width) - 1

    S: List[Tuple[int, int]] = []
    p_add = True
    remaining = steps

    current_fml = fml
    while True:
        # Early stop
        if remaining == 0:
            break

        # Compute [l, u] within current bounds
        s_l = z3.Solver()
        s_l.add(current_fml)
        l_assumptions = list(_bounds_to_bv_constraints(x, l_bits, u_bits, signed))
        l_min_val = None
        # Search min by bits with additional bound assumptions
        result_l = 0
        for i in reversed(range(width)):
            assumps = [
                z3.Extract(j, j, x) == z3.BitVecVal((result_l >> j) & 1, 1)
                for j in range(i + 1, width)
            ]
            assumps += l_assumptions
            assumps.append(z3.Extract(i, i, x) == z3.BitVecVal(0, 1))
            if _assume_and_check(s_l, assumps) == z3.sat:
                pass
            else:
                result_l |= (1 << i)
        l_min_val = result_l

        s_u = z3.Solver()
        s_u.add(current_fml)
        u_assumptions = list(_bounds_to_bv_constraints(x, l_bits, u_bits, signed))
        result_u = 0
        for i in reversed(range(width)):
            assumps = [
                z3.Extract(j, j, x) == z3.BitVecVal((result_u >> j) & 1, 1)
                for j in range(i + 1, width)
            ]
            assumps += u_assumptions
            assumps.append(z3.Extract(i, i, x) == z3.BitVecVal(1, 1))
            if _assume_and_check(s_u, assumps) == z3.sat:
                result_u |= (1 << i)
            else:
                pass

        if _value(l_min_val, width, signed) >= _value(result_u, width, signed):
            break

        a = _value(l_min_val, width, signed)
        b = _value(result_u, width, signed)

        if p_add:
            # add interval
            S.append((a, b))
        else:
            # subtract interval from S
            S = _intervals_subtract(S, (a, b))

        # Alternate and flip search space: restrict to complement by toggling fml
        p_add = not p_add
        current_fml = z3.Not(current_fml)
        remaining = remaining - 1 if remaining is not None and remaining > 0 else remaining

        # Narrow bounds for next iteration
        if p_add:
            # After a removal step, reset to full range
            if signed:
                l_bits = 1 << (width - 1)
                u_bits = (1 << (width - 1)) - 1
            else:
                l_bits = 0
                u_bits = (1 << width) - 1
        else:
            # After an addition, try to look outside the just-added block by
            # splitting, but for simplicity we keep global bounds.
            pass

    # Merge adjacent intervals
    S = _intervals_normalize(S)
    return S


def _intervals_subtract(S: List[Tuple[int, int]], rem: Tuple[int, int]) -> List[Tuple[int, int]]:
    a, b = rem
    res: List[Tuple[int, int]] = []
    for (l, u) in S:
        if u < a or b < l:
            res.append((l, u))
        else:
            if l < a:
                res.append((l, min(u, a - 1)))
            if b < u:
                res.append((max(l, b + 1), u))
    return res


def _intervals_normalize(S: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not S:
        return S
    S = sorted(S)
    merged: List[Tuple[int, int]] = [S[0]]
    for l, u in S[1:]:
        ml, mu = merged[-1]
        if l <= mu + 1:
            merged[-1] = (ml, max(mu, u))
        else:
            merged.append((l, u))
    return merged
