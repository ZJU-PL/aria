"""
Interval arithmetic for rational numbers.

This module provides interval arithmetic operations over rational numbers,
useful for abstract interpretation and numerical analysis.
"""

from __future__ import annotations
from typing import Dict, List, Set, Tuple, Optional, Union, Any
from fractions import Fraction
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Type aliases
QQ = Fraction  # Rational numbers


@dataclass(frozen=True)
class Interval:
    """Represents a closed interval over rational numbers."""

    lower: Optional[QQ]  # Lower bound (None for -∞)
    upper: Optional[QQ]  # Upper bound (None for +∞)

    def __post_init__(self):
        """Normalize the interval after creation."""
        if (self.lower is not None and self.upper is not None and
            self.lower > self.upper):
            # Invalid interval becomes bottom (empty)
            object.__setattr__(self, 'lower', QQ(1))
            object.__setattr__(self, 'upper', QQ(0))

    @staticmethod
    def bottom() -> Interval:
        """Empty interval (⊥)."""
        return Interval(QQ(1), QQ(0))

    @staticmethod
    def top() -> Interval:
        """Universal interval (⊤) covering all rationals."""
        return Interval(None, None)

    @staticmethod
    def const(k: QQ) -> Interval:
        """Create a point interval [k, k]."""
        return Interval(k, k)

    @staticmethod
    def zero() -> Interval:
        """Create the interval [0, 0]."""
        return Interval.const(QQ(0))

    @staticmethod
    def one() -> Interval:
        """Create the interval [1, 1]."""
        return Interval.const(QQ(1))

    @staticmethod
    def make_bounded(lower: QQ, upper: QQ) -> Interval:
        """Create a bounded interval [lower, upper]."""
        return Interval(lower, upper)

    @staticmethod
    def make(lower: Optional[QQ], upper: Optional[QQ]) -> Interval:
        """Create an interval with given bounds."""
        return Interval(lower, upper)

    def __eq__(self, other: object) -> bool:
        """Check if two intervals are equal."""
        if not isinstance(other, Interval):
            return False
        return (self.lower == other.lower and
                self.upper == other.upper)

    def __hash__(self) -> int:
        """Hash function for intervals."""
        return hash((self.lower, self.upper))

    def __str__(self) -> str:
        """String representation of the interval."""
        lower_str = "-∞" if self.lower is None else str(self.lower)
        upper_str = "+∞" if self.upper is None else str(self.upper)
        return f"[{lower_str}, {upper_str}]"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"Interval(lower={self.lower}, upper={self.upper})"

    def is_bottom(self) -> bool:
        """Check if this interval is empty (⊥)."""
        return (self.lower is not None and self.upper is not None and
                self.lower > self.upper)

    def is_top(self) -> bool:
        """Check if this interval is the universal set (⊤)."""
        return self.lower is None and self.upper is None

    def is_point(self) -> bool:
        """Check if this interval represents a single point."""
        return (self.lower is not None and self.upper is not None and
                self.lower == self.upper)

    def contains(self, x: QQ) -> bool:
        """Check if this interval contains the given rational number."""
        if self.is_bottom():
            return False
        if self.lower is not None and x < self.lower:
            return False
        if self.upper is not None and x > self.upper:
            return False
        return True

    def __neg__(self) -> Interval:
        """Negate the interval."""
        return Interval(
            None if self.upper is None else -self.upper,
            None if self.lower is None else -self.lower
        )

    def __add__(self, other: Interval) -> Interval:
        """Add two intervals."""
        if self.is_bottom() or other.is_bottom():
            return Interval.bottom()

        lower = None
        if self.lower is not None and other.lower is not None:
            lower = self.lower + other.lower
        elif self.lower is None and other.lower is not None:
            lower = None  # -∞ + c = -∞
        elif self.lower is not None and other.lower is None:
            lower = None  # c + (-∞) = -∞

        upper = None
        if self.upper is not None and other.upper is not None:
            upper = self.upper + other.upper
        elif self.upper is None and other.upper is not None:
            upper = None  # +∞ + c = +∞
        elif self.upper is not None and other.upper is None:
            upper = None  # c + (+∞) = +∞

        return Interval(lower, upper)

    def __sub__(self, other: Interval) -> Interval:
        """Subtract two intervals."""
        return self + (-other)

    def __mul__(self, other: Interval) -> Interval:
        """Multiply two intervals."""
        if self.is_bottom() or other.is_bottom():
            return Interval.bottom()

        # For simplicity, we'll implement a basic version
        # A complete implementation would consider all sign combinations
        candidates = []

        # Compute all possible products of bounds
        bounds_self = [self.lower, self.upper]
        bounds_other = [other.lower, other.upper]

        for a in bounds_self:
            for b in bounds_other:
                if a is not None and b is not None:
                    candidates.append(a * b)

        if not candidates:
            return Interval.top()

        return Interval(min(candidates), max(candidates))

    def __truediv__(self, other: Interval) -> Interval:
        """Divide two intervals."""
        if self.is_bottom() or other.is_bottom():
            return Interval.bottom()

        # Division by interval containing 0 is problematic
        if other.contains(QQ(0)):
            return Interval.top()  # Could be more precise

        return self * other.reciprocal()

    def reciprocal(self) -> Interval:
        """Compute the reciprocal (1/x) of this interval."""
        if self.is_bottom():
            return Interval.bottom()

        if self.contains(QQ(0)):
            return Interval.top()

        if self.lower is None and self.upper is None:
            return Interval.top()

        # Reciprocal of [a, b] is [1/b, 1/a] (assuming a, b > 0 or a, b < 0)
        if self.lower is not None and self.upper is not None:
            # Handle case where interval crosses zero (should return top)
            if self.lower < 0 < self.upper:
                return Interval.top()
            return Interval(QQ(1)/self.upper, QQ(1)/self.lower)

        # Handle infinite bounds
        if self.lower is None:
            if self.upper is not None:
                if self.upper > 0:
                    return Interval(QQ(0), QQ(1)/self.upper)
                else:
                    return Interval(QQ(1)/self.upper, QQ(0))
            return Interval.top()
        if self.upper is None:
            if self.lower is not None:
                if self.lower > 0:
                    return Interval(QQ(1)/self.lower, QQ(0))
                else:
                    return Interval(QQ(0), QQ(1)/self.lower)
            return Interval.top()

        return Interval.top()

    def union(self, other: Interval) -> Interval:
        """Compute the union of two intervals."""
        if self.is_bottom():
            return other
        if other.is_bottom():
            return self

        lower = None
        if self.lower is not None and other.lower is not None:
            lower = min(self.lower, other.lower)
        elif self.lower is None:
            lower = other.lower
        elif other.lower is None:
            lower = self.lower

        upper = None
        if self.upper is not None and other.upper is not None:
            upper = max(self.upper, other.upper)
        elif self.upper is None:
            upper = other.upper
        elif other.upper is None:
            upper = self.upper

        return Interval(lower, upper)

    def intersection(self, other: Interval) -> Interval:
        """Compute the intersection of two intervals."""
        if self.is_bottom() or other.is_bottom():
            return Interval.bottom()

        lower = None
        if self.lower is not None and other.lower is not None:
            lower = max(self.lower, other.lower)
        elif self.lower is None:
            lower = other.lower
        elif other.lower is None:
            lower = self.lower

        upper = None
        if self.upper is not None and other.upper is not None:
            upper = min(self.upper, other.upper)
        elif self.upper is None:
            upper = other.upper
        elif other.upper is None:
            upper = self.upper

        return Interval(lower, upper)

    def widen(self, other: Interval) -> Interval:
        """Widen this interval to include another interval."""
        return self.union(other)

    def narrow(self, other: Interval) -> Interval:
        """Narrow this interval to the intersection with another interval."""
        return self.intersection(other)

    def __pow__(self, other: Interval) -> Interval:
        """Compute interval power operation."""
        if self.is_bottom() or other.is_bottom():
            return Interval.bottom()

        # For now, implement a basic version
        # A complete implementation would handle all cases properly
        if self.is_point() and other.is_point():
            # Both are single points
            base = self.lower if self.lower is not None else QQ(0)
            exponent = other.lower if other.lower is not None else QQ(0)
            try:
                result = base ** exponent
                return Interval(result, result)
            except:
                return Interval.top()
        else:
            # Conservative approximation for non-point intervals
            return Interval.top()

    def sqrt(self) -> Interval:
        """Compute square root of interval (for non-negative intervals)."""
        if self.is_bottom():
            return Interval.bottom()

        if self.lower is not None and self.lower < 0:
            return Interval.top()  # Square root of negative numbers

        if self.lower is None and self.upper is None:
            return Interval.top()

        if self.lower is not None and self.upper is not None:
            if self.lower == self.upper and self.lower == QQ(0):
                return Interval(QQ(0), QQ(0))

            import math
            lower_sqrt = math.sqrt(float(self.lower)) if self.lower is not None else 0
            upper_sqrt = math.sqrt(float(self.upper)) if self.upper is not None else float('inf')
            return Interval(QQ(lower_sqrt), QQ(upper_sqrt))

        # Handle infinite bounds
        if self.lower is None:
            return Interval(QQ(0), QQ(float('inf')))
        if self.upper is None:
            return Interval(QQ(0), QQ(float('inf')))

        return Interval(QQ(0), QQ(float('inf')))

    def abs(self) -> Interval:
        """Compute absolute value of interval."""
        if self.is_bottom():
            return Interval.bottom()

        if self.lower is None and self.upper is None:
            return Interval(QQ(0), None)  # [0, ∞)

        if self.lower is not None and self.upper is not None:
            if self.lower >= 0:
                return Interval(self.lower, self.upper)
            elif self.upper <= 0:
                return Interval(-self.upper, -self.lower)
            else:
                # Interval crosses zero
                return Interval(QQ(0), max(-self.lower, self.upper))

        # Handle infinite bounds
        if self.lower is None:
            return Interval(QQ(0), self.upper if self.upper is not None else None)
        if self.upper is None:
            return Interval(QQ(0), None)

        return Interval(QQ(0), None)

    def midpoint(self) -> Optional[QQ]:
        """Compute midpoint of interval."""
        if self.is_bottom() or self.lower is None or self.upper is None:
            return None
        return (self.lower + self.upper) / QQ(2)

    def width(self) -> Optional[QQ]:
        """Compute width of interval."""
        if self.is_bottom() or self.lower is None or self.upper is None:
            return None
        return self.upper - self.lower

    def split(self) -> Tuple[Interval, Interval]:
        """Split interval at midpoint."""
        if self.is_bottom() or self.lower is None or self.upper is None:
            return (self, self)

        mid = self.midpoint()
        if mid is None:
            return (self, self)

        return (Interval(self.lower, mid), Interval(mid, self.upper))