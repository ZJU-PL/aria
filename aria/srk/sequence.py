"""
Sequence analysis for program analysis and verification.

This module provides tools for analyzing sequences, including:
- Ultimately periodic sequences
- Periodic sequences
- Sequence transformations
- Period detection using Brent's algorithm
- Sequence arithmetic

This follows the OCaml sequence.ml module implementation.
"""

from __future__ import annotations
from typing import Dict, List, Set, Tuple, Optional, Union, Any, Iterator, Callable, TypeVar, Generic
from fractions import Fraction
from dataclasses import dataclass
from abc import ABC, abstractmethod
import itertools
import math

T = TypeVar('T')
U = TypeVar('U')


def lcm(x: int, y: int) -> int:
    """Compute least common multiple of two integers."""
    if x == 0 or y == 0:
        return 0
    return abs(x * y) // math.gcd(x, y)


@dataclass(frozen=True)
class UltimatelyPeriodicSequence(Generic[T]):
    """
    Represents an ultimately periodic sequence: transient prefix + periodic suffix.
    
    An ultimately periodic sequence has the form:
        t[0], t[1], ..., t[n-1], p[0], p[1], ..., p[m-1], p[0], p[1], ...
    where t is the transient part and p repeats infinitely.
    """

    transient: Tuple[T, ...]  # Initial transient part
    periodic: Tuple[T, ...]   # Periodic part that repeats

    def __post_init__(self):
        if not self.periodic:
            raise ValueError("Periodic part cannot be empty")

    def length(self) -> int:
        """Get the length of the transient part."""
        return len(self.transient)

    def period(self) -> int:
        """Get the period length."""
        return len(self.periodic)

    def transient_len(self) -> int:
        """Get the length of the transient part."""
        return len(self.transient)

    def period_len(self) -> int:
        """Get the period length."""
        return len(self.periodic)

    def get(self, k: int) -> T:
        """Get the element at the given index."""
        if k < len(self.transient):
            return self.transient[k]
        else:
            periodic_index = (k - len(self.transient)) % len(self.periodic)
            return self.periodic[periodic_index]

    def nth(self, k: int) -> T:
        """Get the element at the given index."""
        return self.get(k)

    def __getitem__(self, index: int) -> T:
        """Get element at index using [] notation."""
        return self.nth(index)

    def take(self, n: int) -> List[T]:
        """Get the first n elements of the sequence."""
        return [self.nth(i) for i in range(n)]

    def enum(self) -> Iterator[T]:
        """Create an infinite iterator over the sequence."""
        # Yield transient elements
        for x in self.transient:
            yield x
        # Yield periodic elements infinitely
        while True:
            for x in self.periodic:
                yield x

    def map(self, func: Callable[[T], U]) -> UltimatelyPeriodicSequence[U]:
        """Apply a function to each element of the sequence."""
        return UltimatelyPeriodicSequence(
            tuple(func(x) for x in self.transient),
            tuple(func(x) for x in self.periodic)
        )

    def map2(self, other: UltimatelyPeriodicSequence[T], 
             func: Callable[[T, T], U]) -> UltimatelyPeriodicSequence[U]:
        """
        Combine two sequences element-wise using a binary function.
        
        The result has a transient length equal to the maximum of the two
        transients, and a period equal to the LCM of the two periods.
        """
        transient = max(self.transient_len(), other.transient_len())
        period = lcm(self.period_len(), other.period_len())
        
        # Compute combined transient
        combined_transient = tuple(
            func(self.nth(i), other.nth(i)) for i in range(transient)
        )
        
        # Compute combined period
        combined_period = tuple(
            func(self.nth(transient + i), other.nth(transient + i)) 
            for i in range(period)
        )
        
        return UltimatelyPeriodicSequence(combined_transient, combined_period)

    @staticmethod
    def mapn(sequences: List[UltimatelyPeriodicSequence[T]], 
             func: Callable[[List[T]], U]) -> UltimatelyPeriodicSequence[U]:
        """
        Combine multiple sequences element-wise using a function.
        
        Args:
            sequences: List of sequences to combine
            func: Function that takes a list of elements and returns a result
        """
        if not sequences:
            raise ValueError("Cannot mapn over empty list of sequences")
        
        # Compute transient length (max of all transients)
        transient = max(seq.transient_len() for seq in sequences)
        
        # Compute period (LCM of all periods)
        period = 1
        for seq in sequences:
            period = lcm(period, seq.period_len())
        
        # Compute combined transient
        combined_transient = tuple(
            func([seq.nth(i) for seq in sequences]) 
            for i in range(transient)
        )
        
        # Compute combined period
        combined_period = tuple(
            func([seq.nth(transient + i) for seq in sequences]) 
            for i in range(period)
        )
        
        return UltimatelyPeriodicSequence(combined_transient, combined_period)

    def equal(self, other: UltimatelyPeriodicSequence[T], 
              equal_func: Optional[Callable[[T, T], bool]] = None) -> bool:
        """
        Check if two sequences are equal.
        
        Two ultimately periodic sequences are equal if they agree on a prefix
        of length (max transient + LCM of periods).
        """
        if equal_func is None:
            equal_func = lambda x, y: x == y
        
        transient = max(self.transient_len(), other.transient_len())
        period = lcm(self.period_len(), other.period_len())
        check_length = transient + period
        
        for i in range(check_length):
            if not equal_func(self.nth(i), other.nth(i)):
                return False
        
        return True

    def filter(self, predicate: Callable[[T], bool]) -> UltimatelyPeriodicSequence[T]:
        """Filter sequence elements based on predicate."""
        # Filter transient part
        filtered_transient = tuple(x for x in self.transient if predicate(x))

        # Filter periodic part
        filtered_periodic = tuple(x for x in self.periodic if predicate(x))

        # If no elements remain in periodic part, create a simple constant sequence
        if not filtered_periodic:
            # Return a sequence with just the filtered transient elements
            # This is a simplified implementation
            return UltimatelyPeriodicSequence(filtered_transient, (filtered_transient[0] if filtered_transient else 0,))

        return UltimatelyPeriodicSequence(filtered_transient, filtered_periodic)

    def __add__(self, other: UltimatelyPeriodicSequence[T]) -> UltimatelyPeriodicSequence[T]:
        """Concatenate two sequences."""
        return UltimatelyPeriodicSequence(
            self.transient + other.transient,
            self.periodic  # For simplicity, use first sequence's periodic part
        )

    def concat(self, prefix: List[T]) -> UltimatelyPeriodicSequence[T]:
        """Prepend a prefix to the sequence."""
        return UltimatelyPeriodicSequence(
            tuple(prefix) + self.transient,
            self.periodic
        )

    def __str__(self) -> str:
        """String representation of the sequence."""
        transient_str = ', '.join(str(x) for x in self.transient)
        periodic_str = ', '.join(str(x) for x in self.periodic)
        return f"[{transient_str}]({periodic_str})^ω"

    @staticmethod
    def unfold(func: Callable[[T], T], init: T, 
               equal_func: Optional[Callable[[T, T], bool]] = None) -> UltimatelyPeriodicSequence[T]:
        """
        Unfold a sequence from an initial state using Brent's cycle detection.
        
        This algorithm detects when the sequence becomes periodic by finding
        the first repetition in the state sequence.
        
        Args:
            func: Function to generate next state
            init: Initial state
            equal_func: Equality function (default: ==)
        """
        if equal_func is None:
            equal_func = lambda x, y: x == y
        
        # Brent's cycle detection algorithm
        def find_period() -> int:
            """Find the period length using Brent's algorithm."""
            tortoise = init
            hare = func(init)
            power = 1
            period_len = 1
            
            while not equal_func(tortoise, hare):
                if power == period_len:
                    tortoise = hare
                    power *= 2
                    period_len = 0
                hare = func(hare)
                period_len += 1
            
            return period_len
        
        period_len = find_period()
        
        # Build the sequence up to and including the periodic part
        def construct_seq(n: int, x: T) -> Tuple[T, List[T]]:
            """Construct n elements starting from x."""
            seq = []
            current = x
            for _ in range(n):
                seq.append(current)
                current = func(current)
            return (current, seq)
        
        # Find where the repetition starts
        def find_transient_start() -> List[T]:
            """Find the transient part before the period starts."""
            tortoise = init
            hare, period_seq = construct_seq(period_len, init)
            seq = []
            
            while not equal_func(tortoise, hare):
                seq.append(tortoise)
                tortoise = func(tortoise)
                hare = func(hare)
            
            return seq
        
        transient = find_transient_start()
        
        # Build the periodic part
        periodic = []
        current = func(transient[-1]) if transient else init
        for _ in range(period_len):
            periodic.append(current)
            current = func(current)
        
        return UltimatelyPeriodicSequence(tuple(transient), tuple(periodic))


@dataclass(frozen=True)
class PeriodicSequence(Generic[T]):
    """
    Represents a purely periodic sequence (no transient part).
    
    A periodic sequence has the form:
        p[0], p[1], ..., p[n-1], p[0], p[1], ..., p[n-1], ...
    """

    periodic: Tuple[T, ...]  # Periodic part that repeats

    def __post_init__(self):
        if not self.periodic:
            raise ValueError("Periodic part cannot be empty")

    def period_len(self) -> int:
        """Get the period length."""
        return len(self.periodic)

    def nth(self, k: int) -> T:
        """Get the element at the given index."""
        return self.periodic[k % len(self.periodic)]

    def __getitem__(self, index: int) -> T:
        """Get element at index using [] notation."""
        return self.nth(index)

    def enum(self) -> Iterator[T]:
        """Create an infinite iterator over the sequence."""
        while True:
            for x in self.periodic:
                yield x

    def map(self, func: Callable[[T], U]) -> PeriodicSequence[U]:
        """Apply a function to each element of the sequence."""
        return PeriodicSequence(tuple(func(x) for x in self.periodic))

    def map2(self, other: PeriodicSequence[T], 
             func: Callable[[T, T], U]) -> PeriodicSequence[U]:
        """Combine two sequences element-wise using a binary function."""
        period = lcm(self.period_len(), other.period_len())
        combined = tuple(
            func(self.nth(i), other.nth(i)) for i in range(period)
        )
        return PeriodicSequence(combined)

    @staticmethod
    def mapn(sequences: List[PeriodicSequence[T]], 
             func: Callable[[List[T]], U]) -> PeriodicSequence[U]:
        """Combine multiple sequences element-wise."""
        if not sequences:
            raise ValueError("Cannot mapn over empty list of sequences")
        
        period = 1
        for seq in sequences:
            period = lcm(period, seq.period_len())
        
        combined = tuple(
            func([seq.nth(i) for seq in sequences]) 
            for i in range(period)
        )
        
        return PeriodicSequence(combined)

    def equal(self, other: PeriodicSequence[T], 
              equal_func: Optional[Callable[[T, T], bool]] = None) -> bool:
        """Check if two periodic sequences are equal."""
        if equal_func is None:
            equal_func = lambda x, y: x == y
        
        period = lcm(self.period_len(), other.period_len())
        
        for i in range(period):
            if not equal_func(self.nth(i), other.nth(i)):
                return False
        
        return True

    def __str__(self) -> str:
        """String representation of the sequence."""
        periodic_str = ', '.join(str(x) for x in self.periodic)
        return f"({periodic_str})^ω"


# Utility functions

def periodic_approx(up: UltimatelyPeriodicSequence[T]) -> PeriodicSequence[T]:
    """
    Approximate an ultimately periodic sequence as a purely periodic one.
    
    This rotates the periodic part so that the resulting periodic sequence
    has the same long-term behavior.
    """
    transient = up.transient_len()
    period = up.period_len()
    
    # Compute rotation amount
    n = (period - (transient % period)) % period
    
    # Rotate the periodic part
    periodic_list = list(up.periodic)
    rotated = periodic_list[n:] + periodic_list[:n]
    
    return PeriodicSequence(tuple(rotated))


# Factory functions

def make_constant_sequence(value: T) -> UltimatelyPeriodicSequence[T]:
    """Create a constant sequence."""
    return UltimatelyPeriodicSequence((), (value,))


def make_arithmetic_sequence(start: int, step: int) -> UltimatelyPeriodicSequence[int]:
    """Create an arithmetic sequence (unbounded, so we approximate)."""
    # For an arithmetic sequence, we can't truly represent it as ultimately periodic
    # unless it's constant (step=0). This is a helper for bounded cases.
    if step == 0:
        return make_constant_sequence(start)
    # Return a small sample - in practice, use itertools or generators
    sample = tuple(start + i * step for i in range(10))
    return UltimatelyPeriodicSequence(sample[:-1], (sample[-1],))


def make_periodic(period: List[T]) -> PeriodicSequence[T]:
    """Create a periodic sequence from a period."""
    if not period:
        raise ValueError("Cannot make periodic sequence with empty period")
    return PeriodicSequence(tuple(period))


def make_ultimately_periodic(transient: List[T], period: List[T]) -> UltimatelyPeriodicSequence[T]:
    """Create an ultimately periodic sequence."""
    if not period:
        raise ValueError("Cannot make ultimately periodic sequence with empty period")
    return UltimatelyPeriodicSequence(tuple(transient), tuple(period))


class SequenceAnalyzer:
    """Analyzer for detecting patterns in sequences."""

    @staticmethod
    def detect_period(sequence: List[T]) -> Optional[int]:
        """Detect the period of a sequence using autocorrelation."""
        n = len(sequence)
        if n < 2:
            return None

        # Compute autocorrelation for different lags
        max_corr = 0
        best_period = None

        for period in range(1, n // 2 + 1):
            corr = 0
            count = 0
            for i in range(n - period):
                if sequence[i] == sequence[i + period]:
                    corr += 1
                count += 1

            corr_rate = corr / count if count > 0 else 0
            if corr_rate > max_corr:
                max_corr = corr_rate
                best_period = period

        # Only return period if correlation is high enough
        return best_period if max_corr > 0.8 else None

    @staticmethod
    def find_repeating_pattern(sequence: List[T]) -> Optional[Tuple[int, List[T]]]:
        """Find the first repeating pattern in a sequence."""
        n = len(sequence)
        if n < 2:
            return None

        # Try different pattern lengths
        for length in range(1, n // 2 + 1):
            pattern = sequence[:length]
            pattern_found = True

            # Check if pattern repeats throughout the sequence
            for i in range(length, n, length):
                if i + length <= n:
                    if sequence[i:i + length] != pattern:
                        pattern_found = False
                        break

            if pattern_found:
                return (0, pattern)

        return None

    @staticmethod
    def compute_autocorrelation(sequence: List[T]) -> List[float]:
        """Compute autocorrelation coefficients for different lags."""
        n = len(sequence)
        if n < 2:
            return []

        correlations = []
        for lag in range(1, n):
            corr = 0
            count = 0
            for i in range(n - lag):
                if sequence[i] == sequence[i + lag]:
                    corr += 1
                count += 1

            corr_rate = corr / count if count > 0 else 0
            correlations.append(corr_rate)

        return correlations

    @staticmethod
    def detect_ultimately_periodic(sequence: List[T]) -> Optional[UltimatelyPeriodicSequence[T]]:
        """Detect if sequence is ultimately periodic and return the structure."""
        if len(sequence) < 2:
            return None

        # Try shorter transients first (start from the beginning)
        for transient_len in range(len(sequence)):
            suffix = sequence[transient_len:]

            # Try different periods for this suffix
            for period in range(1, len(suffix) // 2 + 1):
                if len(suffix) < 2 * period:
                    continue

                pattern = suffix[:period]
                remainder = suffix[period:]

                # Check if remainder matches pattern repeated
                if len(remainder) == 0:
                    # No remainder means the candidate is exactly one period long
                    return UltimatelyPeriodicSequence(
                        tuple(sequence[:transient_len]),
                        tuple(pattern)
                    )

                # Check if remainder matches pattern repeated
                matches = True
                for i, val in enumerate(remainder):
                    if val != pattern[i % period]:
                        matches = False
                        break

                if matches:
                    return UltimatelyPeriodicSequence(
                        tuple(sequence[:transient_len]),
                        tuple(pattern)
                    )

        return None


def fibonacci_sequence() -> List[int]:
    """Create a Fibonacci sequence."""
    # Generate Fibonacci sequence: 1, 1, 2, 3, 5, 8, 13, 21, 34, ...
    fib = [1, 1]
    for i in range(2, 20):  # Generate first 20 elements
        fib.append(fib[i-1] + fib[i-2])
    return fib


def arithmetic_sequence(start: int, step: int, length: int) -> List[int]:
    """Create an arithmetic sequence."""
    return [start + i * step for i in range(length)]


def geometric_sequence(start: int, ratio: int, length: int) -> List[int]:
    """Create a geometric sequence."""
    return [start * (ratio ** i) for i in range(length)]
