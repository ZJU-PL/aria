"""
Arbitrary precision integer operations for SRK.

This module provides arbitrary precision integer arithmetic operations,
equivalent to the OCaml zZ.ml module but using Python's built-in
arbitrary precision integers.
"""

from __future__ import annotations
from typing import Optional
import math

# Type alias for clarity
t = int

def pp(formatter, x: int) -> None:
    """Pretty print an integer."""
    print(x, file=formatter, end="")

def show(x: int) -> str:
    """Convert integer to string representation."""
    return str(x)

def hash(x: int) -> int:
    """Hash function for integers."""
    return hash(str(x))

def compare(x: int, y: int) -> int:
    """Compare two integers: returns -1, 0, or 1."""
    if x < y:
        return -1
    elif x > y:
        return 1
    else:
        return 0

def equal(x: int, y: int) -> bool:
    """Check if two integers are equal."""
    return x == y

def leq(x: int, y: int) -> bool:
    """Check if x <= y."""
    return x <= y

def lt(x: int, y: int) -> bool:
    """Check if x < y."""
    return x < y

def add(x: int, y: int) -> int:
    """Add two integers."""
    return x + y

def sub(x: int, y: int) -> int:
    """Subtract two integers."""
    return x - y

def mul(x: int, y: int) -> int:
    """Multiply two integers."""
    return x * y

def negate(x: int) -> int:
    """Negate an integer."""
    return -x

def one() -> int:
    """Return the integer 1."""
    return 1

def zero() -> int:
    """Return the integer 0."""
    return 0

def modulo(x: int, y: int) -> int:
    """Compute x mod y (remainder)."""
    if y == 0:
        raise ZeroDivisionError("Modulo by zero")
    return x % y

def div(x: int, y: int) -> int:
    """Integer division (floor division)."""
    if y == 0:
        raise ZeroDivisionError("Division by zero")
    return x // y

def gcd(x: int, y: int) -> int:
    """Greatest common divisor."""
    x = abs_val(x)
    y = abs_val(y)
    while y != 0:
        x, y = y, x % y
    return x

def lcm(x: int, y: int) -> int:
    """Least common multiple."""
    if x == 0 or y == 0:
        return 0
    return abs(x * y) // gcd(x, y)

def of_int(x: int) -> int:
    """Convert int to arbitrary precision integer."""
    return x

def of_string(s: str) -> int:
    """Parse string as integer."""
    return int(s)

def to_int(x: int) -> Optional[int]:
    """Convert to regular int if possible, None otherwise."""
    # In Python, all ints are arbitrary precision, so this always succeeds
    # but we'll simulate the OCaml behavior for very large numbers
    try:
        # Check if it fits in a 64-bit signed integer
        if -2**63 <= x <= 2**63 - 1:
            return int(x)
        else:
            return None
    except OverflowError:
        return None

def min(x: int, y: int) -> int:
    """Return the minimum of two integers."""
    return min(x, y)

def max(x: int, y: int) -> int:
    """Return the maximum of two integers."""
    return max(x, y)

def abs_val(x: int) -> int:
    """Absolute value."""
    return abs(x) if x >= 0 else -x
