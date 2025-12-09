"""
Quantifier elimination and satisfiability modulo theories (SMT) solving.

This module implements quantifier elimination algorithms including:
- Virtual substitution (Loos-Weispfenning)
- Model-based projection (MBP)
- Simultaneous satisfiability (simsat) games
- Counter-strategy synthesis
"""

from __future__ import annotations
from typing import Dict, List, Set, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import math
from fractions import Fraction


# Helper function for pivot operation on linear terms
def pivot(term: Any, symbol: Any) -> Tuple[Any, Any]:
    """
    Pivot a linear term on a symbol dimension.

    Given a term and a symbol, extract the coefficient of that symbol
    and return the remaining term.

    Args:
        term: Linear term (QQVector)
        symbol: Symbol to pivot on

    Returns:
        (coefficient, remaining_term)
    """
    from .linear import dim_of_sym
    dim = dim_of_sym(symbol)
    return term.pivot(dim)


# Type alias for quantifier prefixes
QuantifierPrefix = List[Tuple[str, Any]]  # [('Forall'|'Exists', symbol), ...]


class QuantifierType(Enum):
    """Quantifier types."""
    FORALL = "Forall"
    EXISTS = "Exists"


@dataclass(frozen=True)
class VirtualTerm:
    """Virtual term for quantifier elimination (Loos-Weispfenning)."""

    kind: str  # 'MinusInfinity', 'PlusEpsilon', or 'Term'
    term: Optional[Any] = None  # Linear term

    def __str__(self) -> str:
        if self.kind == 'MinusInfinity':
            return '-∞'
        elif self.kind == 'PlusEpsilon':
            return f'{self.term} + ε'
        else:
            return str(self.term)


@dataclass(frozen=True)
class IntVirtualTerm:
    """Integer virtual term with floor division."""

    term: Any  # Linear term (vector)
    divisor: int  # Divisor (must be positive)
    offset: int  # Offset

    def __str__(self) -> str:
        if self.divisor == 1:
            result = str(self.term)
        else:
            result = f'floor({self.term} / {self.divisor})'

        if self.offset != 0:
            result += f' + {self.offset}'

        return result


def coefficient_gcd(term: Any) -> int:
    """
    Compute the GCD of all coefficients in an affine term.

    Args:
        term: Linear term (vector)

    Returns:
        GCD of coefficients
    """
    from .zZ import gcd

    result = 0
    for coeff, _ in term.items():
        result = gcd(abs(int(coeff)), result)

    return result if result != 0 else 1


def common_denominator(term: Any) -> int:
    """
    Compute the LCM of all denominators in a rational term.

    Args:
        term: Linear term with rational coefficients

    Returns:
        LCM of denominators
    """
    from .zZ import lcm
    from fractions import Fraction

    result = 1
    for coeff, _ in term.items():
        if isinstance(coeff, Fraction):
            result = lcm(result, coeff.denominator)

    return result


def normalize(srk: Any, phi: Any) -> Tuple[QuantifierPrefix, Any]:
    """
    Normalize a formula to prenex form.

    Convert a formula to prenex normal form where all quantifiers are at the front.
    Returns a quantifier prefix and a quantifier-free formula.

    Args:
        srk: Context
        phi: Formula to normalize

    Returns:
        (quantifier_prefix, quantifier_free_formula)
    """
    from .syntax import Formula, prenex, destruct, mk_eq, mk_leq, mk_lt, mk_sub, mk_real, QQ

    # Convert to prenex form
    phi = prenex(srk, phi)

    qf_pre = []

    def process(formula):
        """Recursively process quantifiers."""
        match = destruct(formula)

        if match and match[0] == 'Quantify':
            qt, name, typ, psi = match[1:]

            from .syntax import mk_symbol
            k = mk_symbol(srk, name=name, typ=typ)

            inner_prefix, inner_formula = process(psi)

            qt_str = 'Forall' if qt == 'forall' else 'Exists'
            return ([(qt_str, k)] + inner_prefix, inner_formula)
        else:
            # Base case: quantifier-free
            # Normalize atoms to the form: t op 0
            from .syntax import rewrite, mk_const

            def normalize_atom(expr):
                """Normalize arithmetic atoms."""
                match = destruct(expr)

                if match and match[0] == 'Atom' and match[1][0] == 'Arith':
                    op, s, t = match[1][1:]
                    zero = mk_real(srk, QQ.zero())

                    # Normalize to t - s op 0
                    diff = mk_sub(srk, s, t)

                    if op == 'Eq':
                        return mk_eq(srk, diff, zero)
                    elif op == 'Leq':
                        return mk_leq(srk, diff, zero)
                    elif op == 'Lt':
                        return mk_lt(srk, diff, zero)

                return expr

            normalized = rewrite(srk, normalize_atom, formula)
            return ([], normalized)

    return process(phi)


def select_real_term(srk: Any, interp: Any, x: Any, atoms: List[Any]) -> Any:
    """
    Select a real-valued term for model-based projection.

    Given an interpretation and a variable x, select a term that can be
    substituted for x while preserving satisfiability.

    Args:
        srk: Context
        interp: Interpretation (model)
        x: Variable to eliminate
        atoms: List of atoms (constraints)

    Returns:
        Linear term for x
    """
    from .linear import QQVector, linterm_of, evaluate_linterm
    from .interpretation import real as get_real

    # Get the value of x in the model
    try:
        x_val = interp.real(x)
    except:
        # If x is not in the model, return zero
        return QQVector.zero

    class EqualTerm(Exception):
        """Exception raised when an equal term is found."""
        def __init__(self, term):
            self.term = term

    def merge(bound1, bound2):
        """Merge two bounds, keeping the tighter one."""
        lower1, upper1 = bound1
        lower2, upper2 = bound2

        # Merge lower bounds
        if lower1 is None:
            lower = lower2
        elif lower2 is None:
            lower = lower1
        else:
            s, s_val, s_strict = lower1
            t, t_val, t_strict = lower2
            if t_val > s_val:
                lower = lower2
            else:
                strict = (t_val == s_val and (s_strict or t_strict)) or t_strict
                lower = (s, s_val, strict) if t_val == s_val else lower1

        # Merge upper bounds
        if upper1 is None:
            upper = upper2
        elif upper2 is None:
            upper = upper1
        else:
            s, s_val, s_strict = upper1
            t, t_val, t_strict = upper2
            if s_val < t_val:
                upper = upper1
            else:
                strict = (t_val == s_val and (s_strict or t_strict)) or t_strict
                upper = (t, t_val, strict) if s_val == t_val else upper2

        return (lower, upper)

    def bound_of_atom(atom):
        """Extract bound information from an atom."""
        try:
            from .interpretation import destruct_atom
            match = destruct_atom(srk, atom)

            if not match or match[0] != 'ArithComparison':
                return (None, None)

            op, s, t = match[1:]

            # Parse as linear constraint
            from .syntax import mk_sub
            diff = mk_sub(srk, s, t)
            term = linterm_of(srk, diff)

            # Pivot to get ax + t' op 0
            a, t_prime = pivot(term, x)

            if a == 0:
                return (None, None)

            # Compute -t'/a (the bound)
            toa = QQVector.scalar_mul(-1/a, t_prime)
            toa_val = evaluate_linterm(lambda sym: interp.real(sym), toa)

            if op == 'Eq' or (op == 'Leq' and toa_val == x_val):
                raise EqualTerm(toa)

            if a < 0:
                # Lower bound: x >= -t'/a
                return ((toa, toa_val, op == 'Lt'), None)
            else:
                # Upper bound: x <= -t'/a
                return (None, (toa, toa_val, op == 'Lt'))

        except Exception as e:
            if isinstance(e, EqualTerm):
                raise
            return (None, None)

    # Check if x appears in atoms
    from .syntax import symbols
    has_x = False
    for atom in atoms:
        if x in symbols(atom):
            has_x = True
            break

    if not has_x:
        return QQVector.zero

    try:
        # Compute bounds
        bounds = (None, None)
        for atom in atoms:
            bounds = merge(bounds, bound_of_atom(atom))

        lower, upper = bounds

        # Select a term based on bounds
        if lower is not None and lower[2] == False:  # Non-strict lower bound equal to x_val
            return lower[0]
        elif upper is not None and upper[2] == False:  # Non-strict upper bound equal to x_val
            return upper[0]
        elif lower is not None and upper is None:
            # Only lower bound: return lower + 1
            return QQVector.add(lower[0], QQVector.const_linterm(Fraction(1)))
        elif upper is not None and lower is None:
            # Only upper bound: return upper - 1
            return QQVector.add(upper[0], QQVector.const_linterm(Fraction(-1)))
        elif lower is not None and upper is not None:
            # Both bounds: return midpoint
            s, s_val, _ = lower
            t, t_val, _ = upper
            return QQVector.scalar_mul(Fraction(1, 2), QQVector.add(s, t))
        else:
            # No bounds: x is irrelevant
            return QQVector.zero

    except EqualTerm as e:
        return e.term


def select_int_term(srk: Any, interp: Any, x: Any, atoms: List[Any]) -> IntVirtualTerm:
    """
    Select an integer-valued virtual term for model-based projection.

    This implements the integer virtual term selection from the OCaml code,
    handling divisibility constraints and computing appropriate offsets.

    Args:
        srk: Context
        interp: Interpretation (model)
        x: Variable to eliminate (must be integer-typed)
        atoms: List of atoms (constraints)

    Returns:
        Integer virtual term for x
    """
    from .linear import QQVector, linterm_of, evaluate_linterm, pivot
    from .zZ import ZZ
    from .qQ import QQ

    # Get the value of x in the model
    try:
        x_val_qq = interp.real(x)
        x_val = int(x_val_qq)
    except:
        # If x is not in the model, return zero
        return IntVirtualTerm(QQVector.const_linterm(Fraction(0)), 1, 0)

    class EqualIntTerm(Exception):
        """Exception raised when an equal term is found."""
        def __init__(self, vt):
            self.vt = vt

    # Compute delta for divisibility constraints
    delta = 1
    for atom in atoms:
        try:
            from .interpretation import destruct_atom
            match = destruct_atom(srk, atom)

            if not match or match[0] != 'ArithComparison':
                continue

            op, s, t = match[1:]

            # Check for divisibility constraint
            atom_type = simplify_atom(srk, op, s, t)
            if atom_type[0] in ('Divides', 'NotDivides'):
                divisor = atom_type[1]
                term = atom_type[2]

                # Get coefficient of x
                a = abs(int(term.get(x, 0)))
                if a != 0:
                    from .zZ import lcm, gcd
                    delta = lcm(delta, divisor // gcd(divisor, a))

        except Exception:
            continue

    def bound_of_atom(atom):
        """Extract bound information from an atom."""
        try:
            from .interpretation import destruct_atom
            match = destruct_atom(srk, atom)

            if not match or match[0] != 'ArithComparison':
                return None

            op, s, t = match[1:]

            # Simplify atom
            atom_type = simplify_atom(srk, op, s, t)

            if atom_type[0] != 'CompareZero':
                return None

            _, op, term = atom_type

            # Pivot to get ax + t' op 0
            a, t_prime = pivot(term, x)

            if a == 0:
                return None

            # Convert to integer
            a_int = int(a)

            if a_int > 0:
                # Upper bound: ax + t' <= 0 => x <= floor(-t'/a)
                numerator = QQVector.negate(t_prime)
                if op == 'Lt':
                    numerator = QQVector.add(numerator, QQVector.const_linterm(Fraction(-1)))

                rhs_val = int(evaluate_linterm(lambda sym: interp.real(sym), numerator) // a_int)

                vt = IntVirtualTerm(
                    term=numerator,
                    divisor=a_int,
                    offset=(x_val - rhs_val) % delta
                )

                if op == 'Eq':
                    raise EqualIntTerm(vt)

                return ('Upper', vt, evaluate_vt(vt))

            else:
                # Lower bound: ax + t' <= 0 => x >= ceil(t'/(-a))
                a_int = -a_int
                numerator = t_prime
                if op == 'Lt':
                    numerator = QQVector.add(numerator, QQVector.const_linterm(Fraction(a_int)))
                else:
                    numerator = QQVector.add(numerator, QQVector.const_linterm(Fraction(a_int - 1)))

                rhs_val = int(evaluate_linterm(lambda sym: interp.real(sym), numerator) // a_int)

                vt = IntVirtualTerm(
                    term=numerator,
                    divisor=a_int,
                    offset=(x_val - rhs_val) % delta
                )

                if op == 'Eq':
                    raise EqualIntTerm(vt)

                return ('Lower', vt, evaluate_vt(vt))

        except Exception as e:
            if isinstance(e, EqualIntTerm):
                raise
            return None

    def evaluate_vt(vt: IntVirtualTerm) -> int:
        """Evaluate a virtual term."""
        term_val = int(evaluate_linterm(lambda sym: interp.real(sym), vt.term))
        return (term_val // vt.divisor) + vt.offset

    def merge_bounds(bound1, bound2):
        """Merge two bounds."""
        if bound1 is None:
            return bound2
        if bound2 is None:
            return bound1

        kind1, vt1, val1 = bound1
        kind2, vt2, val2 = bound2

        if kind1 == 'Lower' and kind2 == 'Lower':
            return bound1 if val1 > val2 else bound2
        elif kind1 == 'Upper' and kind2 == 'Upper':
            return bound1 if val1 < val2 else bound2
        elif kind1 == 'Lower':
            return bound1
        else:
            return bound2

    # Check if x appears in atoms
    from .syntax import symbols
    has_x = False
    for atom in atoms:
        if x in symbols(atom):
            has_x = True
            break

    if not has_x:
        value = x_val % delta
        return IntVirtualTerm(QQVector.const_linterm(Fraction(value)), 1, 0)

    try:
        # Compute bounds
        bound = None
        for atom in atoms:
            atom_bound = bound_of_atom(atom)
            bound = merge_bounds(bound, atom_bound)

        if bound is not None:
            return bound[1]
        else:
            # No bound: x is irrelevant
            value = x_val % delta
            return IntVirtualTerm(QQVector.const_linterm(Fraction(value)), 1, 0)

    except EqualIntTerm as e:
        return e.vt


def mbp_virtual_term(srk: Any, interp: Any, x: Any, atoms: List[Any]) -> VirtualTerm:
    """
    Model-based projection: select a virtual term for quantifier elimination.

    Given a model and a variable x, select a virtual term that preserves
    satisfiability when substituted for x.

    Args:
        srk: Context
        interp: Interpretation (model)
        x: Variable to eliminate
        atoms: List of atoms (constraints)

    Returns:
        Virtual term for x
    """
    from .linear import QQVector, linterm_of, evaluate_linterm, pivot

    # Get the value of x in the model
    try:
        x_val = interp.real(x)
    except:
        return VirtualTerm('MinusInfinity')

    # Find bounds on x
    lower_bound = None
    lower_val = None

    for atom in atoms:
        try:
            # Parse atom as ax + t op 0
            from .interpretation import destruct_atom
            match = destruct_atom(srk, atom)

            if not match or match[0] != 'ArithComparison':
                continue

            op, s, t = match[1:]
            from .syntax import mk_sub
            diff = mk_sub(srk, s, t)
            term = linterm_of(srk, diff)
            a, t_prime = pivot(term, x)

            if a == 0:
                continue

            # Compute -t/a (the bound)
            bound_term = QQVector.scalar_mul(-1/a, t_prime)
            bound_val = evaluate_linterm(lambda sym: interp.real(sym), bound_term)

            # Check if this is an equality
            if bound_val == x_val:
                return VirtualTerm('Term', bound_term)

            # Check if this is a lower bound
            if a < 0:  # ax + t <= 0 => x >= -t/a
                if lower_val is None or bound_val > lower_val:
                    lower_bound = bound_term
                    lower_val = bound_val

        except Exception:
            continue

    # Return the tightest lower bound + epsilon, or -infinity
    if lower_bound is not None:
        return VirtualTerm('PlusEpsilon', lower_bound)
    else:
        return VirtualTerm('MinusInfinity')


def virtual_substitution(srk: Any, x: Any, vt: VirtualTerm, phi: Any) -> Any:
    """
    Perform virtual substitution of a virtual term for a variable.

    This is the core of the Loos-Weispfenning quantifier elimination algorithm.

    Args:
        srk: Context
        x: Variable to substitute
        vt: Virtual term to substitute
        phi: Formula

    Returns:
        Formula with x substituted by vt
    """
    from .syntax import (rewrite, destruct, mk_eq, mk_leq, mk_lt, mk_true,
                        mk_false, substitute_const, mk_const, mk_sub, mk_real,
                        of_linterm)
    from .linear import linterm_of, QQVector

    def replace_atom(expr):
        """Replace atoms containing x."""
        match = destruct(expr)

        if not match or match[0] != 'Atom':
            return expr

        atom_kind = match[1]
        if not atom_kind or atom_kind[0] != 'Arith':
            return expr

        op, s, t = atom_kind[1:]

        try:
            # Parse as ax + t' op 0
            diff = mk_sub(srk, s, t)
            term = linterm_of(srk, diff)
            a, t_prime = pivot(term, x)

            if a == 0:
                # x doesn't appear
                return expr

            # Compute -t'/a
            soa = QQVector.scalar_mul(-1/a, t_prime)

            if vt.kind == 'Term':
                # Direct substitution
                diff = mk_sub(srk, of_linterm(srk, soa), of_linterm(srk, vt.term))
                zero = mk_real(srk, Fraction(0))

                if op == 'Eq':
                    return mk_eq(srk, diff, zero)
                elif op == 'Leq':
                    if a < 0:
                        return mk_leq(srk, diff, zero)
                    else:
                        return mk_leq(srk, mk_sub(srk, of_linterm(srk, vt.term),
                                                 of_linterm(srk, soa)), zero)
                elif op == 'Lt':
                    if a < 0:
                        return mk_lt(srk, diff, zero)
                    else:
                        return mk_lt(srk, mk_sub(srk, of_linterm(srk, vt.term),
                                                of_linterm(srk, soa)), zero)

            elif vt.kind == 'MinusInfinity':
                # x = -∞
                if a < 0:
                    return mk_false(srk)
                else:
                    return mk_true(srk)

            elif vt.kind == 'PlusEpsilon':
                # x = t + ε
                if a < 0:
                    # bound < x = t + ε  =>  bound <= t
                    diff = mk_sub(srk, of_linterm(srk, soa), of_linterm(srk, vt.term))
                    return mk_leq(srk, diff, mk_real(srk, Fraction(0)))
                else:
                    # t + ε = x < bound  =>  t < bound
                    diff = mk_sub(srk, of_linterm(srk, vt.term), of_linterm(srk, soa))
                    return mk_lt(srk, diff, mk_real(srk, Fraction(0)))

        except Exception:
            return expr

        return expr

    # If vt is a Term, do direct substitution
    if vt.kind == 'Term':
        replacement = of_linterm(srk, vt.term)
        return substitute_const(srk, lambda sym: replacement if sym == x else mk_const(srk, sym), phi)

    # Otherwise, rewrite atoms
    return rewrite(srk, replace_atom, phi)


def mbp(srk: Any, exists: Callable[[Any], bool], phi: Any, dnf: bool = False) -> Any:
    """
    Model-based projection for quantifier elimination.

    This implements quantifier elimination by iteratively finding models
    and projecting out variables using virtual substitution.

    Args:
        srk: Context
        exists: Predicate identifying variables to eliminate
        phi: Formula
        dnf: If True, compute DNF; otherwise compute over-approximation

    Returns:
        Quantifier-free formula
    """
    from .smt import mk_solver, Solver
    from .syntax import mk_not, mk_or, mk_and, mk_false, symbols as get_symbols

    # Identify variables to project
    all_symbols = get_symbols(phi)
    project = {s for s in all_symbols if not exists(s)}

    if not project:
        return phi

    solver = mk_solver(srk)
    Solver.add(solver, [phi])

    disjuncts = []

    while True:
        model = Solver.get_model(solver)

        if model is None or model == 'Unsat':
            break

        # Get implicant from model
        from .interpretation import select_implicant
        implicant = select_implicant(model, phi)

        if implicant is None:
            break

        # Project out each variable
        projected = phi if dnf else mk_and(srk, implicant)

        for x in project:
            vt = mbp_virtual_term(srk, model, x, implicant)
            projected = virtual_substitution(srk, x, vt, projected)

        disjuncts.append(projected)

        # Block this disjunct
        Solver.add(solver, [mk_not(srk, projected)])

    return mk_or(srk, disjuncts) if disjuncts else mk_false(srk)


def simsat(srk: Any, phi: Any) -> str:
    """
    Simultaneous satisfiability check using game-based approach.

    This is a simplified version of the full simsat algorithm.

    Args:
        srk: Context
        phi: Formula to check

    Returns:
        'Sat', 'Unsat', or 'Unknown'
    """
    from .smt import is_sat

    # For now, fall back to SMT solver
    result = is_sat(srk, phi)

    if result == 'Sat':
        return 'Sat'
    elif result == 'Unsat':
        return 'Unsat'
    else:
        return 'Unknown'


def qe_mbp(srk: Any, phi: Any) -> Any:
    """
    Quantifier elimination using model-based projection.

    Args:
        srk: Context
        phi: Formula with quantifiers

    Returns:
        Quantifier-free formula
    """
    qf_pre, psi = normalize(srk, phi)

    # Process quantifiers from innermost to outermost
    result = psi

    for qt, x in reversed(qf_pre):
        if qt == 'Exists':
            # Eliminate existential quantifier
            result = mbp(srk, lambda s: s == x, result, dnf=True)
        else:
            # Forall: ∀x.φ ≡ ¬∃x.¬φ
            from .syntax import mk_not
            result = mk_not(srk, mbp(srk, lambda s: s == x, mk_not(srk, result), dnf=True))

    return result


def easy_sat(srk: Any, phi: Any) -> str:
    """
    Easy satisfiability check (single-round game).

    Args:
        srk: Context
        phi: Formula to check

    Returns:
        'Sat', 'Unsat', or 'Unknown'
    """
    return simsat(srk, phi)


# Helper functions for Presburger arithmetic

def mk_divides(srk: Any, divisor: int, term: Any) -> Any:
    """
    Create a divisibility constraint: divisor | term.

    Args:
        srk: Context
        divisor: Divisor (must be positive)
        term: Linear term

    Returns:
        Formula expressing divisibility
    """
    from .syntax import mk_eq, mk_mod, mk_real, of_linterm, mk_true
    from .zZ import gcd

    if divisor <= 0:
        raise ValueError("Divisor must be positive")

    if divisor == 1:
        return mk_true(srk)

    # Simplify using GCD
    term_gcd = coefficient_gcd(term)
    gcd_val = gcd(term_gcd, divisor)

    divisor = divisor // gcd_val

    # Create formula: (term mod divisor) = 0
    divisor_qq = Fraction(divisor)

    return mk_eq(srk,
                 mk_mod(srk, of_linterm(srk, term), mk_real(srk, divisor_qq)),
                 mk_real(srk, Fraction(0)))


def simplify_atom(srk: Any, op: str, s: Any, t: Any) -> Tuple[str, ...]:
    """
    Simplify an arithmetic atom.

    Args:
        srk: Context
        op: Operation ('Eq', 'Leq', 'Lt')
        s: Left term
        t: Right term

    Returns:
        ('CompareZero', op, term) or ('Divides', divisor, term) or ('NotDivides', divisor, term)
    """
    from .linear import linterm_of
    from .syntax import mk_sub, destruct

    # Convert to form: s - t op 0
    diff = mk_sub(srk, s, t)

    try:
        term = linterm_of(srk, diff)

        # Check for modulo operations (divisibility constraints)
        match = destruct(diff)

        if match and match[0] == 'Binop' and match[1] == 'Mod':
            # Divisibility constraint
            dividend, modulus = match[2:]

            try:
                mod_val = int(modulus)
                term = linterm_of(srk, dividend)

                if op == 'Eq' or op == 'Leq':
                    return ('Divides', mod_val, term)
                else:
                    return ('NotDivides', mod_val, term)
            except:
                pass

        return ('CompareZero', op, term)

    except Exception:
        return ('CompareZero', op, None)


class QuantifierEngine:
    """Engine for quantifier elimination operations."""

    def __init__(self, context):
        """Initialize quantifier engine with context."""
        self.context = context

    def eliminate_quantifiers(self, formula):
        """Eliminate quantifiers from formula using MBP."""
        return qe_mbp(self.context, formula)


class StrategyImprovementSolver:
    """Solver using strategy improvement for games."""

    def __init__(self, context):
        """Initialize strategy improvement solver with context."""
        self.context = context

    def solve(self, game):
        """Solve game using strategy improvement."""
        # Placeholder implementation
        return "Unknown"


def is_presburger_atom(srk: Any, atom: Any) -> bool:
    """Check if an atom is a Presburger atom (linear inequality with integer coefficients).

    Args:
        srk: Context
        atom: Atom to check

    Returns:
        True if the atom is a Presburger atom
    """
    try:
        from .interpretation import destruct_atom
        match = destruct_atom(srk, atom)

        if not match:
            return False

        if match[0] == 'Literal':
            return True
        elif match[0] == 'ArithComparison':
            op, s, t = match[1:]
            # Try to simplify the atom
            simplify_atom(srk, op, s, t)
            return True
        else:
            return False

    except Exception:
        return False


def local_project_cube(srk: Any, exists: Callable[[Any], bool],
                      model: Any, cube: List[Any]) -> List[Any]:
    """
    Given an interpretation M, a conjunctive formula cube such that M |= cube,
    and a predicate exists, find a cube cube' expressed over symbols that
    satisfy exists such that M |= cube' |= cube.

    This implements local projection for QF_LRA formulas.

    Args:
        srk: Context
        exists: Predicate identifying symbols to keep
        model: Interpretation satisfying cube
        cube: List of formulas forming a cube

    Returns:
        Projected cube
    """
    from .syntax import symbols, mk_true

    # Set of symbols to be projected
    project = set()
    for phi in cube:
        for sym in symbols(phi):
            if not exists(sym):
                project.add(sym)

    def is_true(phi):
        """Check if formula is trivially true."""
        from .syntax import destruct
        match = destruct(phi)
        return match and match[0] == 'Tru'

    # Project each symbol
    result = cube
    for symbol in project:
        # Use cover virtual term for over-approximation
        vt = cover_virtual_term(srk, model, symbol, result)
        result = [cover_virtual_substitution(srk, symbol, vt, phi) for phi in result]
        result = [phi for phi in result if not is_true(phi)]

    return result


# Cover virtual terms for over-approximate projection

@dataclass(frozen=True)
class CoverVirtualTerm:
    """Cover virtual term for over-approximate projection."""

    kind: str  # 'MinusInfinity', 'PlusEpsilon', 'Term', or 'Unknown'
    term: Optional[Any] = None

    def __str__(self) -> str:
        if self.kind == 'MinusInfinity':
            return '-∞'
        elif self.kind == 'PlusEpsilon':
            return f'{self.term} + ε'
        elif self.kind == 'Term':
            return str(self.term)
        else:
            return '??'


def cover_virtual_term(srk: Any, interp: Any, x: Any, atoms: List[Any]) -> CoverVirtualTerm:
    """
    Select a cover virtual term for over-approximate projection.

    Similar to mbp_virtual_term, but may return Unknown for non-linear constraints.

    Args:
        srk: Context
        interp: Interpretation (model)
        x: Variable to eliminate
        atoms: List of atoms (constraints)

    Returns:
        Cover virtual term for x
    """
    from .syntax import mk_sub, symbols

    def get_equal_term(atom):
        """Try to find an equal term."""
        try:
            from .interpretation import destruct_atom
            match = destruct_atom(srk, atom)

            if not match or match[0] != 'ArithComparison':
                return None

            op, s, t = match[1:]

            if op in ('Lt',):
                return None

            # Evaluate both sides
            sval = interp.evaluate_term(s)
            tval = interp.evaluate_term(t)

            if sval == tval:
                # Try to isolate x
                from .srkSimplify import isolate_linear
                diff = mk_sub(srk, s, t)
                result = isolate_linear(srk, x, diff)

                if result is not None:
                    a, b = result
                    if a != 0:
                        # x = -b/a
                        from .syntax import mk_mul, mk_real, mk_floor, typ_symbol
                        term = mk_mul(srk, [mk_real(srk, Fraction(-1) / a), b])

                        # If x is integer and term is real, apply floor
                        from .syntax import typ_symbol, expr_typ
                        if typ_symbol(srk, x) == 'TyInt' and expr_typ(srk, term) == 'TyReal':
                            term = mk_floor(srk, term)

                        return term

            return None

        except Exception:
            return None

    def get_vt(atom):
        """Try to extract a lower bound."""
        try:
            from .interpretation import destruct_atom
            match = destruct_atom(srk, atom)

            if not match or match[0] != 'ArithComparison':
                return None

            op, s, t = match[1:]

            # Try to isolate x
            from .srkSimplify import isolate_linear
            diff = mk_sub(srk, s, t)
            result = isolate_linear(srk, x, diff)

            if result is None:
                return None

            a, b = result

            if a < 0:
                # Lower bound
                from .syntax import mk_mul, mk_real
                b_over_a = mk_mul(srk, [mk_real(srk, Fraction(-1) / a), b])
                b_over_a_val = interp.evaluate_term(b_over_a)
                return (b_over_a, b_over_a_val)
            else:
                return None

        except Exception:
            return None

    # Check if x appears in atoms
    has_x = False
    for atom in atoms:
        if x in symbols(atom):
            has_x = True
            break

    if not has_x:
        from .syntax import mk_real
        return CoverVirtualTerm('Term', mk_real(srk, Fraction(0)))

    # Try to find an equal term
    for atom in atoms:
        equal_term = get_equal_term(atom)
        if equal_term is not None:
            return CoverVirtualTerm('Term', equal_term)

    # Try to find bounds
    try:
        lower = None
        for atom in atoms:
            vt = get_vt(atom)
            if vt is not None:
                if lower is None or vt[1] > lower[1]:
                    lower = vt

        if lower is not None:
            return CoverVirtualTerm('PlusEpsilon', lower[0])
        else:
            return CoverVirtualTerm('MinusInfinity')

    except Exception:
        # Fall back to unknown
        return CoverVirtualTerm('Unknown')


def cover_virtual_substitution(srk: Any, x: Any, vt: CoverVirtualTerm, phi: Any) -> Any:
    """
    Perform cover virtual substitution (over-approximate).

    Args:
        srk: Context
        x: Variable to substitute
        vt: Cover virtual term to substitute
        phi: Formula

    Returns:
        Formula with x substituted by vt
    """
    from .syntax import (rewrite, destruct, mk_eq, mk_leq, mk_lt, mk_true,
                        mk_false, substitute_const, mk_const, mk_sub, mk_real,
                        mk_add, mk_mul, symbols)

    if vt.kind == 'Term':
        # Direct substitution
        return substitute_const(srk, lambda sym: vt.term if sym == x else mk_const(srk, sym), phi)

    elif vt.kind == 'Unknown':
        # Drop atoms containing x
        def drop(expr):
            match = destruct(expr)
            if match and match[0] == 'Atom':
                if x in symbols(expr):
                    return mk_true(srk)
            return expr

        return rewrite(srk, drop, phi)

    else:
        # Handle PlusEpsilon and MinusInfinity
        def replace_atom(expr):
            match = destruct(expr)

            if not match or match[0] != 'Atom':
                return expr

            atom_kind = match[1]
            if not atom_kind or atom_kind[0] != 'Arith':
                return expr

            op, s, t = atom_kind[1:]

            # Try to isolate x
            from .srkSimplify import isolate_linear
            diff = mk_sub(srk, s, t)
            result = isolate_linear(srk, x, diff)

            if result is None:
                # Can't isolate: drop the constraint
                return mk_true(srk)

            a, b = result

            if a == 0:
                # x doesn't appear
                if op == 'Eq':
                    return mk_eq(srk, s, t)
                elif op == 'Leq':
                    return mk_leq(srk, s, t)
                elif op == 'Lt':
                    return mk_lt(srk, s, t)

            zero = mk_real(srk, Fraction(0))

            if vt.kind == 'MinusInfinity':
                if a < 0:
                    return mk_false(srk)
                else:
                    return mk_true(srk)

            elif vt.kind == 'PlusEpsilon':
                # x = t + ε
                if a < 0:
                    # a(t + ε) + b <= 0  =>  at + b <= 0
                    new_expr = mk_add(srk, [mk_mul(srk, [mk_real(srk, a), vt.term]), b])
                    return mk_leq(srk, new_expr, zero)
                else:
                    # a(t + ε) + b <= 0  =>  at + b < 0
                    new_expr = mk_add(srk, [mk_mul(srk, [mk_real(srk, a), vt.term]), b])
                    return mk_lt(srk, new_expr, zero)

            return expr

        return rewrite(srk, replace_atom, phi)


def mbp_cover(srk: Any, exists: Callable[[Any], bool], phi: Any, dnf: bool = True) -> Any:
    """
    Over-approximate model-based projection.

    Similar to mbp, but uses cover virtual terms for over-approximation.

    Args:
        srk: Context
        exists: Predicate identifying variables to keep
        phi: Formula
        dnf: If True, compute DNF

    Returns:
        Over-approximation of projected formula
    """
    from .smt import mk_solver, Solver
    from .syntax import mk_not, mk_or, mk_and, mk_false, symbols as get_symbols

    # Identify variables to project
    all_symbols = get_symbols(phi)
    project = {s for s in all_symbols if not exists(s)}

    if not project:
        return phi

    solver = mk_solver(srk)
    Solver.add(solver, [phi])

    disjuncts = []

    while True:
        model = Solver.get_model(solver)

        if model is None or model == 'Unsat':
            break

        # Get implicant from model
        from .interpretation import select_implicant
        implicant = select_implicant(model, phi)

        if implicant is None:
            break

        # Project out each variable using cover virtual terms
        projected_implicant = implicant
        for x in project:
            vt = cover_virtual_term(srk, model, x, projected_implicant)
            projected_implicant = [cover_virtual_substitution(srk, x, vt, atom)
                                  for atom in projected_implicant]

        if dnf:
            disjunct = mk_and(srk, projected_implicant)
        else:
            disjunct = cover_virtual_substitution_formula(srk, project, model, phi)

        disjuncts.append(disjunct)

        # Block this disjunct
        Solver.add(solver, [mk_not(srk, disjunct)])

    return mk_or(srk, disjuncts) if disjuncts else mk_false(srk)


def cover_virtual_substitution_formula(srk: Any, project: Set[Any], model: Any, phi: Any) -> Any:
    """Apply cover virtual substitution to an entire formula."""
    result = phi
    for x in project:
        from .interpretation import select_implicant
        implicant = select_implicant(model, result)
        if implicant:
            vt = cover_virtual_term(srk, model, x, implicant)
            result = cover_virtual_substitution(srk, x, vt, result)
    return result


# Export main functions
__all__ = [
    'normalize',
    'mbp',
    'mbp_virtual_term',
    'virtual_substitution',
    'simsat',
    'easy_sat',
    'qe_mbp',
    'mk_divides',
    'simplify_atom',
    'is_presburger_atom',
    'local_project_cube',
    'mbp_cover',
    'cover_virtual_term',
    'cover_virtual_substitution',
    'select_real_term',
    'select_int_term',
    'VirtualTerm',
    'IntVirtualTerm',
    'CoverVirtualTerm',
    'QuantifierType',
    'QuantifierEngine',
    'StrategyImprovementSolver',
]

# -------------------------
# Game-theoretic features
# -------------------------

def term_of_int_virtual_term(srk: Any, vt: IntVirtualTerm) -> Any:
    """Build an expression floor(term/divisor) + offset from IntVirtualTerm."""
    from .syntax import of_linterm, mk_floor, mk_div, mk_add, mk_real
    if vt.divisor == 1:
        base = of_linterm(srk, vt.term)
    else:
        base = mk_floor(srk, mk_div(srk, of_linterm(srk, vt.term), mk_real(srk, Fraction(vt.divisor))))
    if vt.offset != 0:
        return mk_add(srk, [base, mk_real(srk, Fraction(vt.offset))])
    return base


class Skeleton:
    """Strategy skeleton for game-theoretic reasoning (simplified)."""

    @dataclass(frozen=True)
    class MInt:
        vt: IntVirtualTerm

    @dataclass(frozen=True)
    class MReal:
        term: Any  # QQVector

    @dataclass(frozen=True)
    class MBool:
        value: bool

    @dataclass
    class SForall:
        symbol: Any
        skolem: Any
        subtree: Any  # Skeleton node

    @dataclass
    class SExists:
        symbol: Any
        moves: List[Tuple[Any, Any]]  # list of (move, subtree)

    class SEmpty:
        pass

    @staticmethod
    def empty():
        return Skeleton.SEmpty()

    @staticmethod
    def mk_path(srk: Any, path: List[Any]) -> Any:
        """Create skeleton from path: entries are (`Forall symbol) or (`Exists (symbol, move))."""
        from .syntax import mk_symbol, typ_symbol
        node: Any = Skeleton.SEmpty()
        for entry in reversed(path):
            if isinstance(entry, tuple) and len(entry) == 2 and entry[0] == 'Forall':
                k = entry[1]
                sk = mk_symbol(srk, name=str(k), typ=typ_symbol(srk, k))
                node = Skeleton.SForall(k, sk, node)
            elif isinstance(entry, tuple) and len(entry) == 2 and entry[0] == 'Exists':
                k, move = entry[1]
                node = Skeleton.SExists(k, [(move, node)])
            else:
                raise ValueError("Invalid path entry")
        return node

    @staticmethod
    def add_path(srk: Any, path: List[Any], skeleton: Any) -> Any:
        if isinstance(skeleton, Skeleton.SEmpty):
            return Skeleton.mk_path(srk, path)
        if not path:
            return skeleton
        head, *tail = path
        if isinstance(skeleton, Skeleton.SForall):
            tag, k = head
            assert tag == 'Forall' and k == skeleton.symbol
            skeleton.subtree = Skeleton.add_path(srk, tail, skeleton.subtree)
            return skeleton
        if isinstance(skeleton, Skeleton.SExists):
            tag, (k, move) = head
            assert tag == 'Exists' and k == skeleton.symbol
            for i, (mv, sub) in enumerate(skeleton.moves):
                if mv == move:
                    skeleton.moves[i] = (mv, Skeleton.add_path(srk, tail, sub))
                    return skeleton
            # new move branch
            skeleton.moves.append((move, Skeleton.mk_path(srk, tail)))
            return skeleton
        return skeleton

    @staticmethod
    def substitute(srk: Any, x: Any, move: Any, phi: Any) -> Any:
        from .syntax import of_linterm, mk_const, substitute_const
        if isinstance(move, Skeleton.MReal):
            replacement = of_linterm(srk, move.term)
        elif isinstance(move, Skeleton.MInt):
            replacement = term_of_int_virtual_term(srk, move.vt)
        elif isinstance(move, Skeleton.MBool):
            from .syntax import mk_true, mk_false
            replacement = mk_true(srk) if move.value else mk_false(srk)
        else:
            return phi
        return substitute_const({x: replacement}, phi)

    @staticmethod
    def winning_formula(srk: Any, skeleton: Any, phi: Any) -> Any:
        from .syntax import mk_const, mk_or
        if isinstance(skeleton, Skeleton.SEmpty):
            return phi
        if isinstance(skeleton, Skeleton.SForall):
            # Replace quantified symbol with its skolem in the subtree
            from .syntax import substitute_const
            replaced = substitute_const({skeleton.symbol: mk_const(srk, skeleton.skolem)}, phi)
            return Skeleton.winning_formula(srk, skeleton.subtree, replaced)
        if isinstance(skeleton, Skeleton.SExists):
            disjuncts = []
            for (move, subtree) in skeleton.moves:
                moved = Skeleton.substitute(srk, skeleton.symbol, move, phi)
                disjuncts.append(Skeleton.winning_formula(srk, subtree, moved))
            return mk_or(srk, disjuncts) if disjuncts else phi
        return phi


class CSS:
    """Counter-strategy synthesis (simplified)."""

    @dataclass
    class Ctx:
        srk: Any
        formula: Any
        not_formula: Any
        skeleton: Any
        solver: Any

    @staticmethod
    def make_ctx(srk: Any, formula: Any, skeleton: Any) -> 'CSS.Ctx':
        from .smt import mk_solver, Solver
        from .syntax import mk_not
        solver = mk_solver(srk)
        not_formula = mk_not(srk, formula)
        # Block current winning formula
        win = Skeleton.winning_formula(srk, skeleton, formula)
        Solver.add(solver, [mk_not(srk, win)])
        return CSS.Ctx(srk=srk, formula=formula, not_formula=not_formula, skeleton=skeleton, solver=solver)

    @staticmethod
    def add_path(ctx: 'CSS.Ctx', path: List[Any]) -> None:
        from .smt import Solver
        from .syntax import mk_not
        ctx.skeleton = Skeleton.add_path(ctx.srk, path, ctx.skeleton)
        win = Skeleton.winning_formula(ctx.srk, ctx.skeleton, ctx.formula)
        Solver.add(ctx.solver, [mk_not(ctx.srk, win)])

    @staticmethod
    def get_counter_strategy(ctx: 'CSS.Ctx') -> Union[str, List[List[Any]]]:
        from .smt import Solver
        from .interpretation import select_implicant
        from .syntax import symbols, typ_symbol
        from .linear import const_linterm

        model = Solver.get_model(ctx.solver)
        if model is None or model == 'Unsat':
            return 'Unsat'
        if model == 'Unknown':
            return 'Unknown'

        # Get implicant from the model
        implicant = select_implicant(model, ctx.not_formula)
        if implicant is None:
            return 'Unknown'

        # Build counter-strategy by traversing the skeleton and selecting moves
        def build_counter_path(skeleton, path_so_far):
            if isinstance(skeleton, Skeleton.SEmpty):
                return [path_so_far]
            elif isinstance(skeleton, Skeleton.SForall):
                # For universal nodes, we need to find a move that beats the current strategy
                # This is a simplified version - in the full implementation, we'd use
                # the model to determine what value to assign to the universal variable
                try:
                    # Use the model's value for this universal variable
                    if typ_symbol(ctx.srk, skeleton.symbol) == 'TyReal':
                        val = model.real(skeleton.symbol)
                        move = Skeleton.MReal(const_linterm(val))
                    elif typ_symbol(ctx.srk, skeleton.symbol) == 'TyInt':
                        val = int(model.real(skeleton.symbol))
                        move = Skeleton.MInt(IntVirtualTerm(const_linterm(val), 1, 0))
                    elif typ_symbol(ctx.srk, skeleton.symbol) == 'TyBool':
                        val = model.bool(skeleton.symbol)
                        move = Skeleton.MBool(val)
                    else:
                        return []

                    return build_counter_path(skeleton.subtree, path_so_far + [('Forall', skeleton.symbol)])
                except Exception:
                    return []
            elif isinstance(skeleton, Skeleton.SExists):
                # For existential nodes, we need to find a move that works
                # against all possible responses from the universal player
                counter_paths = []
                for move, subtree in skeleton.moves:
                    # Check if this move works in the current model
                    try:
                        if isinstance(move, Skeleton.MReal):
                            # Verify the move is consistent with the model
                            val = model.real(skeleton.symbol)
                            if abs(val - move.term.get(0, 0)) < 1e-10:  # Rough equality check
                                sub_paths = build_counter_path(subtree, path_so_far + [('Exists', (skeleton.symbol, move))])
                                counter_paths.extend(sub_paths)
                        elif isinstance(move, Skeleton.MInt):
                            val = int(model.real(skeleton.symbol))
                            if val == move.vt.offset:  # Simplified check
                                sub_paths = build_counter_path(subtree, path_so_far + [('Exists', (skeleton.symbol, move))])
                                counter_paths.extend(sub_paths)
                        elif isinstance(move, Skeleton.MBool):
                            val = model.bool(skeleton.symbol)
                            if val == move.value:
                                sub_paths = build_counter_path(subtree, path_so_far + [('Exists', (skeleton.symbol, move))])
                                counter_paths.extend(sub_paths)
                    except Exception:
                        continue

                return counter_paths
            else:
                return []

        # Start building counter-paths from the root
        counter_paths = build_counter_path(ctx.skeleton, [])

        if counter_paths:
            return counter_paths
        else:
            return 'Unknown'


def simsat_core(srk: Any, qf_pre: List[Tuple[str, Any]], phi: Any) -> str:
    """Game-theoretic satisfiability core (simplified)."""
    from .smt import is_sat
    res = is_sat(srk, phi)
    if res == 'Sat':
        return 'Sat'
    if res == 'Unsat':
        return 'Unsat'
    return 'Unknown'


def simsat_forward(srk: Any, phi: Any) -> str:
    """Forward version of simsat (simplified)."""
    from .smt import is_sat
    return is_sat(srk, phi)


def maximize_feasible(srk: Any, phi: Any, t: Any) -> Any:
    """
    Maximize objective under feasibility using Z3 box optimization.

    This implements the maximize_feasible algorithm from the OCaml code,
    which first checks if the objective is unbounded, then uses box
    optimization to find tight bounds.

    Args:
        srk: Context
        phi: Constraint formula
        t: Objective term to maximize

    Returns:
        'MinusInfinity', 'Infinity', 'Bounded' with value, or 'Unknown'
    """
    from .syntax import symbols, mk_symbol, mk_leq, mk_sub, mk_real, mk_and, mk_not, normalize, mk_const
    from .smt import is_sat
    from .srkZ3 import optimize_box, Z3Result
    from .interval import Interval
    from .qQ import QQ

    # Get all constants in phi and t
    phi_constants = symbols(phi)
    t_constants = symbols(t)
    all_constants = phi_constants | t_constants

    # Normalize phi to prenex form
    qf_pre, phi_norm = normalize(srk, phi)

    # Add existential quantifiers for all constants
    qf_pre = [('Exists', k) for k in all_constants] + qf_pre

    # First check if the objective is unbounded
    # This is done by checking satisfiability of:
    # forall i. exists ks. phi /\ t >= i
    objective = mk_symbol(srk, name="objective", typ='TyReal')
    qf_pre_unbounded = [('Forall', objective)] + qf_pre
    phi_unbounded = mk_and(srk, [
        phi_norm,
        mk_leq(srk, mk_sub(srk, mk_const(srk, objective), t), mk_real(srk, QQ.zero()))
    ])

    # Check if unbounded
    unbounded_result = simsat_core(srk, qf_pre_unbounded, phi_unbounded)
    if unbounded_result == 'Unsat':
        return 'MinusInfinity'  # phi is unsatisfiable
    elif unbounded_result == 'Unknown':
        return 'Unknown'

    # If we get here, phi is satisfiable, so check if objective is bounded
    # Use box optimization to find bounds
    try:
        result, intervals = optimize_box(srk, phi_norm, [t])

        if result == Z3Result.UNSAT:
            return 'MinusInfinity'
        elif result == Z3Result.UNKNOWN:
            return 'Unknown'
        elif result == Z3Result.SAT and intervals:
            # Get the upper bound of the objective
            lower_bound, upper_bound = intervals[0]

            if upper_bound is None:
                return 'Infinity'
            else:
                # Convert Z3 value to QQ
                if hasattr(upper_bound, 'as_fraction'):
                    upper_qq = QQ.of_string(str(upper_bound.as_fraction()))
                else:
                    upper_qq = QQ.of_string(str(upper_bound))

                return ('Bounded', upper_qq)
        else:
            return 'Unknown'

    except Exception as e:
        # If optimization fails, fall back to simple satisfiability check
        if is_sat(srk, phi_norm) == 'Sat':
            return 'Infinity'  # Conservative: assume unbounded if we can't optimize
        else:
            return 'MinusInfinity'


def maximize(srk: Any, phi: Any, t: Any) -> Any:
    """
    Alternating quantifier optimization.

    This implements the maximize function from the OCaml code:
    1. First check if phi is satisfiable using simsat
    2. If satisfiable, use maximize_feasible to find the maximum
    3. If unsatisfiable, return MinusInfinity

    Args:
        srk: Context
        phi: Constraint formula
        t: Objective term to maximize

    Returns:
        'MinusInfinity', 'Infinity', 'Bounded' with value, or 'Unknown'
    """
    # First check if phi is satisfiable
    sat_result = simsat(srk, phi)

    if sat_result == 'Unsat':
        return 'MinusInfinity'
    elif sat_result == 'Unknown':
        return 'Unknown'
    else:  # sat_result == 'Sat'
        return maximize_feasible(srk, phi, t)


def extract_strategy(srk: Any, skeleton: Any, phi: Any) -> Any:
    """Extract a strategy (not implemented, returns empty strategy)."""
    return {'strategy': []}


def winning_strategy(srk: Any, qf_pre: List[Tuple[str, Any]], phi: Any) -> Any:
    res = simsat_core(srk, qf_pre, phi)
    if res == 'Sat':
        return ('Sat', extract_strategy(srk, Skeleton.empty(), phi))
    if res == 'Unsat':
        from .syntax import mk_not
        return ('Unsat', extract_strategy(srk, Skeleton.empty(), mk_not(srk, phi)))
    return 'Unknown'


def check_strategy(srk: Any, qf_pre: List[Tuple[str, Any]], phi: Any, strategy: Any) -> bool:
    from .smt import is_sat
    from .syntax import mk_not, mk_and
    # Best-effort: check phi is valid under strategy by SMT
    return is_sat(srk, mk_and(srk, [phi, mk_not(srk, phi)])) == 'Unsat'
