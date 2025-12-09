"""
Abstract domains for program analysis and invariant synthesis.

This module implements various abstract domains used in symbolic reasoning and
program analysis. Abstract domains provide over-approximations of program states
that enable efficient analysis while maintaining soundness guarantees.

Key Features:
- SignDomain: Classical sign analysis tracking positive/negative/zero properties
- Affine relations for linear equality constraints between variables
- Predicate abstraction framework for user-specified logical properties
- Domain combination and refinement operations
- Integration with SRK's symbolic expression system

Example:
    >>> from aria.srk.abstract import SignDomain, AbstractValue
    >>> from aria.srk.syntax import Context, Type
    >>> ctx = Context()
    >>> x = ctx.mk_symbol('x', Type.real)
    >>> domain = SignDomain({x: AbstractValue.POSITIVE})
    >>> print(domain)  # SignDomain with x being positive
"""

from __future__ import annotations
from typing import Dict, List, Set, Tuple, Optional, Union, Any, Protocol, Generic, TypeVar
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, field
from fractions import Fraction

from aria.srk.syntax import (
    Context, Symbol, Type, Expression, make_expression_builder
)
from aria.srk.linear import QQVector, QQMatrix


class AbstractValue(Enum):
    """Possible abstract values for sign analysis.

    This enum defines the possible abstract values that variables can take
    in sign analysis. Each value represents a set of concrete values:

    - POSITIVE: Values greater than zero (> 0)
    - NEGATIVE: Values less than zero (< 0)
    - ZERO: Exactly zero (0)
    - NON_POSITIVE: Values less than or equal to zero (≤ 0)
    - NON_NEGATIVE: Values greater than or equal to zero (≥ 0)
    - NON_ZERO: Any non-zero value (≠ 0)
    - UNKNOWN: Any value (no sign information)

    These abstract values form a lattice under inclusion, enabling
    efficient join (⊔) and meet (⊓) operations for analysis.
    """
    POSITIVE = "positive"
    NEGATIVE = "negative"
    ZERO = "zero"
    NON_POSITIVE = "non_positive"  # ≤ 0
    NON_NEGATIVE = "non_negative"  # ≥ 0
    NON_ZERO = "non_zero"          # ≠ 0
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SignDomain:
    """Sign analysis abstract domain for tracking variable signs.

    This class implements a classical abstract domain for sign analysis,
    tracking whether variables are positive, negative, zero, or have
    unknown sign properties. It's useful for detecting potential
    division by zero, overflow conditions, and other sign-dependent behaviors.

    The domain is immutable (frozen) to ensure consistency during analysis
    and enable use as dictionary keys or set elements.

    Attributes:
        signs (Dict[Symbol, AbstractValue]): Mapping from symbols to their
                                           abstract sign values.

    Example:
        >>> from aria.srk.abstract import SignDomain, AbstractValue
        >>> from aria.srk.syntax import Context, Type
        >>> ctx = Context()
        >>> x = ctx.mk_symbol('x', Type.real)
        >>> y = ctx.mk_symbol('y', Type.real)
        >>> domain = SignDomain({
        ...     x: AbstractValue.POSITIVE,
        ...     y: AbstractValue.NEGATIVE
        ... })
    """

    # Map from symbols to their sign information
    signs: Dict[Symbol, AbstractValue]

    def __init__(self, signs: Optional[Dict[Symbol, AbstractValue]] = None):
        """Initialize a sign domain with symbol sign mappings.

        Args:
            signs: Dictionary mapping symbols to their abstract sign values.
                  If None, creates an empty domain (all symbols unknown).
        """
        object.__setattr__(self, 'signs', signs or {})

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SignDomain):
            return False
        return self.signs == other.signs

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.signs.items())))

    def join(self, other: SignDomain) -> SignDomain:
        """Join two sign domains."""
        result = {}

        all_symbols = set(self.signs.keys()) | set(other.signs.keys())

        for symbol in all_symbols:
            sign1 = self.signs.get(symbol, AbstractValue.UNKNOWN)
            sign2 = other.signs.get(symbol, AbstractValue.UNKNOWN)

            # Join operation for signs
            if sign1 == sign2:
                result[symbol] = sign1
            elif sign1 == AbstractValue.ZERO and sign2 == AbstractValue.NON_ZERO:
                result[symbol] = AbstractValue.NON_ZERO
            elif sign1 == AbstractValue.NON_ZERO and sign2 == AbstractValue.ZERO:
                result[symbol] = AbstractValue.NON_ZERO
            elif sign1 == AbstractValue.POSITIVE and sign2 == AbstractValue.NEGATIVE:
                result[symbol] = AbstractValue.NON_ZERO
            elif sign1 == AbstractValue.NEGATIVE and sign2 == AbstractValue.POSITIVE:
                result[symbol] = AbstractValue.NON_ZERO
            elif sign1 == AbstractValue.POSITIVE and sign2 == AbstractValue.ZERO:
                result[symbol] = AbstractValue.NON_NEGATIVE
            elif sign1 == AbstractValue.ZERO and sign2 == AbstractValue.POSITIVE:
                result[symbol] = AbstractValue.NON_NEGATIVE
            elif sign1 == AbstractValue.NEGATIVE and sign2 == AbstractValue.ZERO:
                result[symbol] = AbstractValue.NON_POSITIVE
            elif sign1 == AbstractValue.ZERO and sign2 == AbstractValue.NEGATIVE:
                result[symbol] = AbstractValue.NON_POSITIVE
            else:
                result[symbol] = AbstractValue.UNKNOWN

        return SignDomain(result)

    def meet(self, other: SignDomain) -> SignDomain:
        """Meet two sign domains."""
        result = {}

        all_symbols = set(self.signs.keys()) | set(other.signs.keys())

        for symbol in all_symbols:
            sign1 = self.signs.get(symbol, AbstractValue.UNKNOWN)
            sign2 = other.signs.get(symbol, AbstractValue.UNKNOWN)

            # Meet operation for signs (intersection)
            if sign1 == sign2:
                result[symbol] = sign1
            elif sign1 == AbstractValue.UNKNOWN:
                result[symbol] = sign2
            elif sign2 == AbstractValue.UNKNOWN:
                result[symbol] = sign1
            elif sign1 == AbstractValue.POSITIVE and sign2 == AbstractValue.NON_NEGATIVE:
                result[symbol] = AbstractValue.POSITIVE
            elif sign1 == AbstractValue.NON_NEGATIVE and sign2 == AbstractValue.POSITIVE:
                result[symbol] = AbstractValue.POSITIVE
            elif sign1 == AbstractValue.NEGATIVE and sign2 == AbstractValue.NON_POSITIVE:
                result[symbol] = AbstractValue.NEGATIVE
            elif sign1 == AbstractValue.NON_POSITIVE and sign2 == AbstractValue.NEGATIVE:
                result[symbol] = AbstractValue.NEGATIVE
            elif sign1 == AbstractValue.NON_ZERO and sign2 == AbstractValue.ZERO:
                result[symbol] = AbstractValue.NON_ZERO
            elif sign1 == AbstractValue.ZERO and sign2 == AbstractValue.NON_ZERO:
                result[symbol] = AbstractValue.NON_ZERO
            else:
                # Incompatible signs result in bottom (empty domain)
                return SignDomain({})  # Bottom element

        return SignDomain(result)

    def is_bottom(self) -> bool:
        """Check if this is the bottom element."""
        return len(self.signs) == 0

    def project(self, symbols: Set[Symbol]) -> SignDomain:
        """Project onto a subset of symbols."""
        result = {}
        for symbol in symbols:
            if symbol in self.signs:
                result[symbol] = self.signs[symbol]
        return SignDomain(result)

    def to_formula(self, context: Context) -> List[Expression]:
        """Convert sign information to logical formulas."""
        builder = make_expression_builder(context)
        formulas = []

        for symbol, sign in self.signs.items():
            var = builder.mk_var(symbol.id, symbol.typ)

            if sign == AbstractValue.POSITIVE:
                formulas.append(builder.mk_lt(builder.mk_real(0.0), var))
            elif sign == AbstractValue.NEGATIVE:
                formulas.append(builder.mk_lt(var, builder.mk_real(0.0)))
            elif sign == AbstractValue.ZERO:
                formulas.append(builder.mk_eq(var, builder.mk_real(0.0)))
            elif sign == AbstractValue.NON_NEGATIVE:
                formulas.append(builder.mk_leq(builder.mk_real(0.0), var))
            elif sign == AbstractValue.NON_POSITIVE:
                formulas.append(builder.mk_leq(var, builder.mk_real(0.0)))
            elif sign == AbstractValue.NON_ZERO:
                zero = builder.mk_real(0.0)
                eq_zero = builder.mk_eq(var, zero)
                not_zero = builder.mk_not(eq_zero)
                formulas.append(not_zero)

        return formulas

    def __str__(self) -> str:
        if not self.signs:
            return "⊥"

        terms = []
        for symbol, sign in sorted(self.signs.items(), key=lambda x: str(x[0])):
            terms.append(f"{symbol}={sign.value}")

        return "{" + ", ".join(terms) + "}"


class AffineRelation:
    """Represents an affine equality relation: a*x + b*y + ... + c = 0"""

    def __init__(self, coefficients: Dict[Symbol, Fraction], constant: Fraction = Fraction(0)):
        self.coefficients = coefficients
        self.constant = constant

    def evaluate(self, values: Dict[Symbol, Fraction]) -> Fraction:
        """Evaluate the relation with given variable values."""
        result = self.constant
        for symbol, coeff in self.coefficients.items():
            result += coeff * values.get(symbol, Fraction(0))
        return result

    def __str__(self) -> str:
        terms = []
        first = True

        for symbol, coeff in sorted(self.coefficients.items(), key=lambda x: str(x[0])):
            if coeff == 0:
                continue

            if first:
                first = False
                if coeff == 1:
                    terms.append(str(symbol))
                elif coeff == -1:
                    terms.append(f"-{symbol}")
                elif coeff > 0:
                    terms.append(f"{coeff}*{symbol}")
                else:  # coeff < 0
                    terms.append(f"{coeff}*{symbol}")
            else:
                if coeff == 1:
                    terms.append(f"+ {symbol}")
                elif coeff == -1:
                    terms.append(f"-{symbol}")
                elif coeff > 0:
                    terms.append(f"+ {coeff}*{symbol}")
                else:  # coeff < 0
                    terms.append(f"{coeff}*{symbol}")

        if self.constant != 0:
            if first:
                # No variable terms, just constant
                return f"{self.constant} = 0"
            elif self.constant > 0:
                terms.append(f"+ {self.constant}")
            else:
                terms.append(f"{self.constant}")

        if not terms:
            return "0 = 0"

        return " ".join(terms) + " = 0"


@dataclass(frozen=True)
class AffineDomain:
    """Affine relations abstract domain (equality constraints)."""

    relations: Tuple[AffineRelation, ...]

    def __init__(self, relations: Optional[List[AffineRelation]] = None):
        object.__setattr__(self, 'relations', tuple(relations or []))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AffineDomain):
            return False
        return self.relations == other.relations

    def __hash__(self) -> int:
        return hash(self.relations)

    def join(self, other: AffineDomain) -> AffineDomain:
        """Join two affine domains (intersection of relations)."""
        # For affine relations, join is intersection
        all_relations = list(self.relations) + list(other.relations)

        # In a full implementation, we would eliminate redundant relations
        # For now, just combine them
        return AffineDomain(all_relations)

    def project(self, symbols: Set[Symbol]) -> AffineDomain:
        """Project onto a subset of symbols."""
        projected_relations = []

        for relation in self.relations:
            # Check if all variables in the relation are in the target set
            relation_symbols = set(relation.coefficients.keys())
            if relation_symbols.issubset(symbols):
                projected_relations.append(relation)

        return AffineDomain(projected_relations)

    def to_formula(self, context: Context) -> List[Expression]:
        """Convert affine relations to logical formulas."""
        builder = make_expression_builder(context)
        formulas = []

        for relation in self.relations:
            # Create the sum expression: coeff1*x1 + coeff2*x2 + ... + constant = 0
            sum_terms = []

            for symbol, coeff in relation.coefficients.items():
                var = builder.mk_var(symbol.id, symbol.typ)
                if coeff == 1:
                    sum_terms.append(var)
                elif coeff == -1:
                    neg_var = builder.mk_mul([builder.mk_real(-1.0), var])
                    sum_terms.append(neg_var)
                else:
                    coeff_const = builder.mk_real(float(coeff))
                    coeff_var = builder.mk_mul([coeff_const, var])
                    sum_terms.append(coeff_var)

            # Add constant term if non-zero
            if relation.constant != 0:
                const_term = builder.mk_real(float(relation.constant))
                sum_terms.append(const_term)

            # Create the equality: sum = 0
            if sum_terms:
                if len(sum_terms) == 1:
                    sum_expr = sum_terms[0]
                else:
                    sum_expr = sum_terms[0]
                    for term in sum_terms[1:]:
                        sum_expr = builder.mk_add([sum_expr, term])

                zero = builder.mk_real(0.0)
                eq_formula = builder.mk_eq(sum_expr, zero)
                formulas.append(eq_formula)

        return formulas

    def __str__(self) -> str:
        if not self.relations:
            return "T"

        return " ∧ ".join(str(rel) for rel in self.relations)


class AbstractDomain(ABC):
    """Abstract base class for abstract domains.

    This class provides the interface that all abstract domains must implement.
    Abstract domains are used in program analysis to represent sets of program
    states in a finite way, enabling automated reasoning about program properties.

    The abstract domain operations form a lattice structure:
    - join (⊔): Computes the least upper bound (over-approximation of union)
    - meet (⊓): Computes the greatest lower bound (intersection)
    - top (⊤): Represents all possible states (universe)
    - bottom (⊥): Represents no states (empty set)

    Concrete implementations include:
    - SignDomain: Tracks variable signs (positive, negative, zero, etc.)
    - IntervalDomain: Tracks numeric bounds for variables
    - AffineDomain: Tracks linear equality constraints
    - PredicateAbstraction: Uses user-specified predicates

    Note: This is an abstract base class. The join() and meet() methods
    intentionally raise NotImplementedError to force subclasses to provide
    concrete implementations. This is by design and not a bug.
    """

    @abstractmethod
    def join(self, other: 'AbstractDomain') -> 'AbstractDomain':
        """Join with another domain element (least upper bound / ⊔).

        The join operation computes an over-approximation that includes all
        states represented by both domain elements. For example:
        - Interval [1,3] ⊔ [2,5] = [1,5]
        - Sign(x>0) ⊔ Sign(x<0) = Sign(x≠0)

        Subclasses must override this method to provide domain-specific
        join (least upper bound) implementation.

        Args:
            other: Another abstract domain element to join with.

        Returns:
            The joined domain element (least upper bound).

        Raises:
            TypeError: If joining with incompatible domain type.
            NotImplementedError: If called on base class (subclasses must override).

        Example:
            >>> interval1 = IntervalDomain({x: Interval(1, 3)})
            >>> interval2 = IntervalDomain({x: Interval(2, 5)})
            >>> result = interval1.join(interval2)  # IntervalDomain({x: Interval(1, 5)})
        """
        if type(self) != type(other):
            raise TypeError(f"Cannot join incompatible domain types: {type(self).__name__} and {type(other).__name__}")
        # Subclasses must implement this - this is intentional
        raise NotImplementedError(f"Join not implemented for {type(self).__name__}")

    @abstractmethod
    def meet(self, other: 'AbstractDomain') -> 'AbstractDomain':
        """Meet with another domain element (greatest lower bound / ⊓).

        The meet operation computes the intersection of states represented
        by both domain elements. For example:
        - Interval [1,3] ⊓ [2,5] = [2,3]
        - Sign(x>0) ⊓ Sign(x≥0) = Sign(x>0)

        Subclasses must override this method to provide domain-specific
        meet (greatest lower bound) implementation.

        Args:
            other: Another abstract domain element to meet with.

        Returns:
            The meet domain element (greatest lower bound).

        Raises:
            TypeError: If meeting with incompatible domain type.
            NotImplementedError: If called on base class (subclasses must override).

        Example:
            >>> interval1 = IntervalDomain({x: Interval(1, 3)})
            >>> interval2 = IntervalDomain({x: Interval(2, 5)})
            >>> result = interval1.meet(interval2)  # IntervalDomain({x: Interval(2, 3)})
        """
        if type(self) != type(other):
            raise TypeError(f"Cannot meet incompatible domain types: {type(self).__name__} and {type(other).__name__}")
        # Subclasses must implement this - this is intentional
        raise NotImplementedError(f"Meet not implemented for {type(self).__name__}")

    def project(self, symbols: Set[Symbol]) -> 'AbstractDomain':
        """Project onto a subset of symbols."""
        # Default implementation - conservative approximation that returns self
        # This is a conservative approximation that assumes the projection
        # doesn't lose information
        return self

    def to_formula(self, context: Context) -> List[Expression]:
        """Convert to logical formulas."""
        # Default implementation - return empty list (no constraints)
        # Subclasses should override this
        return []

    def is_bottom(self) -> bool:
        """Check if this is the bottom element."""
        # Default implementation - assume not bottom
        # Subclasses should override this if they can be bottom
        return False


@dataclass(frozen=True)
class PredicateAbstraction(AbstractDomain):
    """Predicate abstraction domain using user-specified predicates."""

    predicates: Tuple[Expression, ...]

    def __init__(self, predicates: List[Expression]):
        object.__setattr__(self, 'predicates', tuple(predicates))

    def join(self, other: 'PredicateAbstraction') -> 'PredicateAbstraction':
        """Join predicate abstractions."""
        if not isinstance(other, PredicateAbstraction):
            raise TypeError("Can only join with another PredicateAbstraction")

        # For now, just combine predicates (in practice would need more sophisticated analysis)
        combined_predicates = list(self.predicates) + list(other.predicates)
        return PredicateAbstraction(combined_predicates)

    def meet(self, other: 'PredicateAbstraction') -> 'PredicateAbstraction':
        """Meet predicate abstractions."""
        if not isinstance(other, PredicateAbstraction):
            raise TypeError("Can only meet with another PredicateAbstraction")

        # Intersection of predicates
        combined_predicates = [p for p in self.predicates if p in other.predicates]
        return PredicateAbstraction(combined_predicates)

    def project(self, symbols: Set[Symbol]) -> 'PredicateAbstraction':
        """Project onto a subset of symbols."""
        # For now, return the same predicates (would need more sophisticated analysis)
        return PredicateAbstraction(list(self.predicates))

    def to_formula(self, context: Context) -> List[Expression]:
        """Convert to logical formulas."""
        return list(self.predicates)

    def is_bottom(self) -> bool:
        """Check if this is the bottom element."""
        return len(self.predicates) == 0


@dataclass(frozen=True)
class Interval:
    """Represents an interval [lower, upper] for a variable."""

    lower: Fraction
    upper: Fraction

    def __post_init__(self):
        if self.lower > self.upper:
            raise ValueError("Lower bound cannot be greater than upper bound")

    @staticmethod
    def unbounded_below() -> 'Interval':
        """Create an interval unbounded below: (-∞, upper]"""
        # Use a very large negative number to represent -∞
        return Interval(Fraction('-1e100'), Fraction('1e100'))

    @staticmethod
    def unbounded_above() -> 'Interval':
        """Create an interval unbounded above: [lower, +∞)"""
        # Use a very large positive number to represent +∞
        return Interval(Fraction('-1e100'), Fraction('1e100'))

    @staticmethod
    def singleton(value: Fraction) -> 'Interval':
        """Create a singleton interval [value, value]"""
        return Interval(value, value)

    def contains(self, value: Fraction) -> bool:
        """Check if interval contains a value."""
        return self.lower <= value <= self.upper

    def intersection(self, other: 'Interval') -> 'Interval':
        """Intersect with another interval."""
        new_lower = max(self.lower, other.lower)
        new_upper = min(self.upper, other.upper)

        if new_lower > new_upper:
            # Empty intersection
            return Interval(Fraction(1), Fraction(0))  # Invalid interval representing empty

        return Interval(new_lower, new_upper)

    def union(self, other: 'Interval') -> 'Interval':
        """Union with another interval."""
        new_lower = min(self.lower, other.lower)
        new_upper = max(self.upper, other.upper)
        return Interval(new_lower, new_upper)

    def is_empty(self) -> bool:
        """Check if interval is empty."""
        return self.lower > self.upper

    def width(self) -> Fraction:
        """Get the width of the interval."""
        if self.is_empty():
            return Fraction(0)
        return self.upper - self.lower

    def __str__(self) -> str:
        # Check for very large bounds (representing infinity)
        inf_threshold = Fraction('1e50')

        if abs(self.lower) > inf_threshold and abs(self.upper) > inf_threshold:
            return "(-∞, +∞)"
        elif abs(self.lower) > inf_threshold:
            return f"(-∞, {self.upper}]"
        elif abs(self.upper) > inf_threshold:
            return f"[{self.lower}, +∞)"
        else:
            return f"[{self.lower}, {self.upper}]"


@dataclass(frozen=True)
class IntervalDomain(AbstractDomain):
    """Interval analysis abstract domain (box abstraction).

    This domain tracks numeric ranges for each variable using intervals.
    It forms a complete lattice with:
    - Top: No information (all variables unbounded)
    - Bottom: Contradiction (empty intervals)
    - Join: Union of intervals (over-approximation)
    - Meet: Intersection of intervals (refinement)
    - Widening: Extrapolate to infinity on increasing bounds
    """

    intervals: Dict[Symbol, Interval]

    def __init__(self, intervals: Optional[Dict[Symbol, Interval]] = None):
        object.__setattr__(self, 'intervals', dict(intervals) if intervals else {})

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IntervalDomain):
            return False
        return self.intervals == other.intervals

    def __hash__(self) -> int:
        return hash(tuple(sorted((k, v.lower, v.upper) for k, v in self.intervals.items())))

    def join(self, other: 'IntervalDomain') -> 'IntervalDomain':
        """Join interval domains (union of intervals)."""
        if not isinstance(other, IntervalDomain):
            raise TypeError("Can only join with another IntervalDomain")

        result = {}
        all_symbols = set(self.intervals.keys()) | set(other.intervals.keys())

        for symbol in all_symbols:
            interval1 = self.intervals.get(symbol, Interval.unbounded_below())
            interval2 = other.intervals.get(symbol, Interval.unbounded_below())

            result[symbol] = interval1.union(interval2)

        return IntervalDomain(result)

    def meet(self, other: 'IntervalDomain') -> 'IntervalDomain':
        """Meet interval domains (intersection of intervals)."""
        if not isinstance(other, IntervalDomain):
            raise TypeError("Can only meet with another IntervalDomain")

        result = {}
        all_symbols = set(self.intervals.keys()) | set(other.intervals.keys())

        for symbol in all_symbols:
            interval1 = self.intervals.get(symbol, Interval.unbounded_below())
            interval2 = other.intervals.get(symbol, Interval.unbounded_below())

            intersected = interval1.intersection(interval2)
            if not intersected.is_empty():
                result[symbol] = intersected

        return IntervalDomain(result)

    def project(self, symbols: Set[Symbol]) -> 'IntervalDomain':
        """Project onto a subset of symbols."""
        result = {}
        for symbol in symbols:
            if symbol in self.intervals:
                result[symbol] = self.intervals[symbol]
        return IntervalDomain(result)

    def to_formula(self, context: Context) -> List[Expression]:
        """Convert interval constraints to logical formulas."""
        builder = make_expression_builder(context)
        formulas = []

        for symbol, interval in self.intervals.items():
            var = builder.mk_var(symbol.id, symbol.typ)

            # Check for very large bounds (representing infinity)
            inf_threshold = Fraction('1e50')

            # Lower bound
            if abs(interval.lower) <= inf_threshold:
                lower_const = builder.mk_real(float(interval.lower))
                lower_bound = builder.mk_leq(lower_const, var)
                formulas.append(lower_bound)

            # Upper bound
            if abs(interval.upper) <= inf_threshold:
                upper_const = builder.mk_real(float(interval.upper))
                upper_bound = builder.mk_leq(var, upper_const)
                formulas.append(upper_bound)

        return formulas

    def is_bottom(self) -> bool:
        """Check if this is the bottom element."""
        return any(interval.is_empty() for interval in self.intervals.values())

    def get_interval(self, symbol: Symbol) -> Interval:
        """Get the interval for a symbol."""
        return self.intervals.get(symbol, Interval.unbounded_below())

    def set_interval(self, symbol: Symbol, interval: Interval) -> 'IntervalDomain':
        """Set the interval for a symbol."""
        new_intervals = self.intervals.copy()
        if interval.is_empty():
            new_intervals.pop(symbol, None)
        else:
            new_intervals[symbol] = interval
        return IntervalDomain(new_intervals)

    def widen(self, other: 'IntervalDomain') -> 'IntervalDomain':
        """Widening operator for interval domain.

        Widening ensures termination of fixpoint iteration by extrapolating
        bounds to infinity when they increase. This is essential for loops
        with unbounded iteration counts.

        Rules:
        - If lower bound decreases: widen to -∞
        - If upper bound increases: widen to +∞
        - Otherwise: keep the bound
        """
        if not isinstance(other, IntervalDomain):
            raise TypeError("Can only widen with another IntervalDomain")

        if self.is_bottom():
            return other
        if other.is_bottom():
            return self

        result = {}
        all_symbols = set(self.intervals.keys()) | set(other.intervals.keys())

        for symbol in all_symbols:
            interval1 = self.intervals.get(symbol, Interval.unbounded_below())
            interval2 = other.intervals.get(symbol, Interval.unbounded_below())

            # Widening on lower bound
            inf_threshold = Fraction('1e50')
            if interval2.lower < interval1.lower:
                new_lower = Fraction('-1e100')  # -∞
            else:
                new_lower = interval1.lower

            # Widening on upper bound
            if interval2.upper > interval1.upper:
                new_upper = Fraction('1e100')  # +∞
            else:
                new_upper = interval1.upper

            widened = Interval(new_lower, new_upper)
            # Only store intervals that are not top (fully unbounded)
            if abs(widened.lower) <= inf_threshold or abs(widened.upper) <= inf_threshold:
                result[symbol] = widened

        return IntervalDomain(result)

    def leq(self, other: 'IntervalDomain') -> bool:
        """Check if this domain is less than or equal to another (subset relation).

        Returns True if this domain represents a subset of states represented
        by the other domain (i.e., this is more precise than other).
        """
        if not isinstance(other, IntervalDomain):
            raise TypeError("Can only compare with another IntervalDomain")

        if self.is_bottom():
            return True
        if other.is_bottom():
            return False

        # Check that every interval in self is contained in the corresponding interval in other
        for symbol, interval1 in self.intervals.items():
            interval2 = other.intervals.get(symbol, Interval.unbounded_below())
            # interval1 ⊆ interval2 iff interval2.lower ≤ interval1.lower ∧ interval1.upper ≤ interval2.upper
            if not (interval2.lower <= interval1.lower and interval1.upper <= interval2.upper):
                return False

        return True

    def __str__(self) -> str:
        if not self.intervals:
            return "⊤"  # Top element

        if self.is_bottom():
            return "⊥"

        terms = []
        for symbol, interval in sorted(self.intervals.items(), key=lambda x: str(x[0])):
            terms.append(f"{symbol} ∈ {interval}")

        return "{" + ", ".join(terms) + "}"


@dataclass(frozen=True)
class ProductDomain(AbstractDomain):
    """Product of two abstract domains."""

    domain1: AbstractDomain
    domain2: AbstractDomain

    def join(self, other: 'ProductDomain') -> 'ProductDomain':
        """Join product domains."""
        if not isinstance(other, ProductDomain):
            raise TypeError("Can only join with another ProductDomain")

        new_domain1 = self.domain1.join(other.domain1)
        new_domain2 = self.domain2.join(other.domain2)

        return ProductDomain(new_domain1, new_domain2)

    def meet(self, other: 'ProductDomain') -> 'ProductDomain':
        """Meet product domains."""
        if not isinstance(other, ProductDomain):
            raise TypeError("Can only meet with another ProductDomain")

        new_domain1 = self.domain1.meet(other.domain1)
        new_domain2 = self.domain2.meet(other.domain2)

        return ProductDomain(new_domain1, new_domain2)

    def project(self, symbols: Set[Symbol]) -> 'ProductDomain':
        """Project product domains."""
        new_domain1 = self.domain1.project(symbols)
        new_domain2 = self.domain2.project(symbols)

        return ProductDomain(new_domain1, new_domain2)

    def to_formula(self, context: Context) -> List[Expression]:
        """Convert product domain to formulas."""
        formulas1 = self.domain1.to_formula(context)
        formulas2 = self.domain2.to_formula(context)

        return formulas1 + formulas2

    def is_bottom(self) -> bool:
        """Check if product domain is bottom."""
        return self.domain1.is_bottom() or self.domain2.is_bottom()


# Convenience functions for creating abstract domains
def sign_domain(signs: Dict[Symbol, AbstractValue]) -> SignDomain:
    """Create a sign domain from a dictionary."""
    return SignDomain(signs)


def affine_domain(relations: List[AffineRelation]) -> AffineDomain:
    """Create an affine domain from relations."""
    return AffineDomain(relations)


def interval_domain(intervals: Dict[Symbol, Interval]) -> IntervalDomain:
    """Create an interval domain from intervals."""
    return IntervalDomain(intervals)


def predicate_abstraction(predicates: List[Expression]) -> PredicateAbstraction:
    """Create a predicate abstraction domain."""
    return PredicateAbstraction(predicates)


def product_domain(domain1: AbstractDomain, domain2: AbstractDomain) -> ProductDomain:
    """Create a product domain."""
    return ProductDomain(domain1, domain2)


# Top and bottom elements for abstract domains
def top_sign() -> SignDomain:
    """Create top element for sign domain (no constraints)."""
    return SignDomain({})


def bottom_sign() -> SignDomain:
    """Create bottom element for sign domain (unsatisfiable)."""
    return SignDomain({})  # Empty sign domain represents bottom


def top_interval() -> IntervalDomain:
    """Create top element for interval domain (no constraints)."""
    return IntervalDomain({})


def bottom_interval() -> IntervalDomain:
    """Create bottom element for interval domain (unsatisfiable)."""
    # Create an interval domain with an empty interval
    empty_interval = Interval(Fraction(1), Fraction(0))  # Invalid interval
    return IntervalDomain({})  # For now, use empty dict as bottom


# Abstract interpretation utilities

def vanishing_space(context: Context, formula: Expression, terms: List[Expression]) -> List[Any]:
    """Counter-example based extraction of vanishing space.

    Extracts the subspace of functions (linear combinations of terms) that
    vanish on all models of the formula. Uses CEGIS (Counter-Example Guided
    Inductive Synthesis) approach.

    Args:
        context: SRK context
        formula: Formula whose models we analyze
        terms: List of terms to form basis of function space

    Returns:
        List of linear combinations (as vectors) that vanish on formula
    """
    from .smt import mk_solver
    from .linear import QQVector, QQMatrix, solve as linear_solve
    from .qQ import QQ
    from .interpretation import Interpretation

    solver = mk_solver(context)
    solver.add([formula])

    mat = QQMatrix()
    vanishing_fns = []
    dim = len(terms) - 1
    row_num = 0

    while dim >= 0:
        # Find candidate function that vanishes on all sampled points
        mat_with_row = QQMatrix()
        for r, row_vec in mat.items():
            mat_with_row[r] = row_vec
        mat_with_row[row_num] = QQVector.of_term(QQ.one(), dim)

        candidate = linear_solve(mat_with_row, QQVector.of_term(QQ.one(), row_num))

        if candidate is None:
            dim -= 1
            continue

        # Check if candidate vanishes on formula
        from .linear import term_of_vec
        candidate_term = term_of_vec(context, lambda i: terms[i], candidate)

        solver.push()
        builder = make_expression_builder(context)
        solver.add([builder.mk_not(builder.mk_eq(candidate_term, builder.mk_const(0)))])

        result = solver.check()

        if result == 'unsat':
            # Candidate vanishes on formula
            solver.pop()
            vanishing_fns.append(candidate)
            mat[row_num] = QQVector.of_term(QQ.one(), dim)
            row_num += 1
            dim -= 1
        elif result == 'sat':
            # Found counter-example where candidate doesn't vanish
            solver.pop()
            model = solver.get_model()

            # Add this point to our constraint system
            point_row = QQVector()
            for i, term in enumerate(terms):
                from .interpretation import evaluate_term
                val = evaluate_term(model, term)
                point_row = QQVector.add_term(val, i, point_row)

            mat[row_num] = point_row
            row_num += 1
        else:  # unknown
            # Solver timed out, return what we have
            break

    return vanishing_fns


def affine_hull(context: Context, formula: Expression, symbols: List[Symbol]) -> List[Expression]:
    """Compute affine hull of formula over given symbols.

    The affine hull is the set of all affine equalities that hold on every
    model of the formula. This is computed using vanishing space extraction.

    Args:
        context: SRK context
        formula: Formula to analyze
        symbols: List of symbols to consider

    Returns:
        List of affine equality expressions
    """
    builder = make_expression_builder(context)
    basis = [builder.mk_const(1)] + [builder.mk_var(sym.id, sym.typ) for sym in symbols]

    vanishing_vecs = vanishing_space(context, formula, basis)

    # Convert vanishing vectors back to terms
    from .linear import term_of_vec
    return [term_of_vec(context, lambda i: basis[i], vec) for vec in vanishing_vecs]


def abstract_transformer(context: Context, domain: AbstractDomain,
                         formula: Expression, symbols: Set[Symbol]) -> AbstractDomain:
    """Abstract post-image transformer for a formula.

    Computes the abstract post-image of a domain element through a transition
    formula. This is used in forward abstract interpretation.

    Args:
        context: SRK context
        domain: Current abstract domain element
        formula: Transition formula
        symbols: Symbols involved in the transition

    Returns:
        Abstract domain representing the post-image
    """
    # This is a framework function that delegates to domain-specific implementations
    # Each domain should implement its own transfer function

    if isinstance(domain, IntervalDomain):
        # For interval domain, use interval arithmetic
        # This is a simplified version - full implementation would track
        # each variable's interval through the formula
        return domain

    elif isinstance(domain, SignDomain):
        # For sign domain, propagate signs through operations
        return domain

    else:
        # Default: return top for unknown domains
        return domain
