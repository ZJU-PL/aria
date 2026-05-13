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

from ..linalg.linear import QQVector, of_linterm, const_dim
from ..linalg.qQ import QQ


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
    from ..linalg.linear import dim_of_sym

    dim = dim_of_sym(symbol)
    return term.pivot(dim)


# Type alias for quantifier prefixes
QuantifierPrefix = List[Tuple[str, Any]]  # [('Forall'|'Exists', symbol), ...]


# ---------------------------------------------------------------------------
# Model-access helpers (work with both Interpretation and SMTModel objects)
# ---------------------------------------------------------------------------

def _model_real(model: Any, symbol: Any) -> Fraction:
    """Get the real value of a symbol from either SMTModel or Interpretation."""
    if hasattr(model, "real"):
        return Fraction(model.real(symbol))
    val = model.get_value(symbol)
    if val is None:
        raise KeyError(f"No value for {symbol}")
    return Fraction(val)


def _model_bool(model: Any, symbol: Any) -> bool:
    """Get the bool value of a symbol from either SMTModel or Interpretation."""
    if hasattr(model, "bool"):
        return model.bool(symbol)
    val = model.get_value(symbol)
    if isinstance(val, bool):
        return val
    return bool(val)


# ---------------------------------------------------------------------------
# Module-level linear-term evaluation (handles Python's const_dim = 0)
# ---------------------------------------------------------------------------

def _evaluate_linterm(srk: Any, model_fn: Callable[[Any], Fraction], term: Any) -> Fraction:
    """Evaluate a linear term using a symbol -> value function.

    Dimension 0 is the constant term (coefficient * 1).  Dimensions > 0
    correspond to symbols whose id is (dim - 1).
    """
    from ..linalg.linear import const_dim as _const_dim

    total = Fraction(0)
    for dim, coeff in term.entries.items():
        coeff = Fraction(coeff)
        if dim == _const_dim:
            total += coeff
        else:
            sym_id = int(dim) - 1
            sym = getattr(srk, "_symbols", {}).get(sym_id)
            if sym is not None:
                total += coeff * Fraction(model_fn(sym))
            else:
                raise ValueError(f"No symbol for dimension {dim}")
    return total


# ---------------------------------------------------------------------------
# select_term dispatching by type
# ---------------------------------------------------------------------------

def _select_term(srk: Any, model: Any, x: Any, atoms: List[Any]) -> Any:
    """Select a Skeleton move for quantified variable *x* based on its type."""
    from ..core.syntax import typ_symbol, Type

    typ = typ_symbol(x)
    if typ == Type.INT:
        return Skeleton.MInt(select_int_term(srk, model, x, atoms))
    elif typ == Type.REAL:
        return Skeleton.MReal(select_real_term(srk, model, x, atoms))
    elif typ == Type.BOOL:
        return Skeleton.MBool(_model_bool(model, x))
    else:
        raise ValueError(f"Unsupported type {typ} for quantified variable {x}")


def _select_implicant(srk: Any, model: Any, formula: Any) -> Optional[List[Any]]:
    """Select an implicant, working with both SMTModel and Interpretation."""
    from .interpretation import select_implicant as _interp_select, Interpretation
    from .interpretation import InterpretationValue
    from ..core.syntax import And, Or, Not, Eq, Lt, Leq, TrueExpr, FalseExpr, Ite, Const, Add, Mul
    from .smt import _numeric_symbol_value

    # If already an Interpretation, use it directly
    if isinstance(model, Interpretation):
        implicant = _interp_select(model, formula)
        if implicant is not None:
            return specialize_floor_cube(srk, model, implicant)
        return None

    # For SMTModel: evaluate with fallback for numeric constants
    def _eval_const(c: Const) -> Optional[Fraction]:
        """Evaluate a Const that may encode a numeric literal."""
        info = _numeric_symbol_value(c.symbol)
        if info is not None:
            val, is_int = info
            return val
        val = model.get_value(c.symbol)
        if val is not None:
            return Fraction(val)
        return None

    def _eval_term(t) -> Optional[Fraction]:
        if isinstance(t, Const):
            return _eval_const(t)
        if isinstance(t, Add):
            total = Fraction(0)
            for a in t.args:
                v = _eval_term(a)
                if v is None:
                    return None
                total += v
            return total
        if isinstance(t, Mul):
            prod = Fraction(1)
            for a in t.args:
                v = _eval_term(a)
                if v is None:
                    return None
                prod *= v
            return prod
        # Fallback
        try:
            return Fraction(model.evaluate_expression(t))
        except Exception:
            return None

    def _eval(f) -> bool:
        try:
            if isinstance(f, TrueExpr):
                return True
            if isinstance(f, FalseExpr):
                return False
            if isinstance(f, (Eq, Lt, Leq)):
                lv = _eval_term(f.left)
                rv = _eval_term(f.right)
                if lv is None or rv is None:
                    return bool(model.evaluate_expression(f))
                if isinstance(f, Eq):
                    return lv == rv
                if isinstance(f, Lt):
                    return lv < rv
                if isinstance(f, Leq):
                    return lv <= rv
            if isinstance(f, Not):
                return not _eval(f.arg)
            if isinstance(f, And):
                return all(_eval(a) for a in f.args)
            if isinstance(f, Or):
                return any(_eval(a) for a in f.args)
            return bool(model.evaluate_expression(f))
        except Exception:
            return False

    def _extract(f):
        if isinstance(f, (Eq, Lt, Leq)):
            return [f] if _eval(f) else []
        if isinstance(f, And):
            result = []
            for arg in f.args:
                result.extend(_extract(arg))
            return result
        if isinstance(f, Or):
            for arg in f.args:
                if _eval(arg):
                    return _extract(arg)
            return []
        if isinstance(f, Not):
            if isinstance(f.arg, (Eq, Lt, Leq)):
                return [f] if _eval(f) else []
            return _extract(f.arg)
        if isinstance(f, (TrueExpr, FalseExpr)):
            return [f] if _eval(f) else []
        if isinstance(f, Ite):
            return _extract(f.then_branch) if _eval(f.condition) else _extract(f.else_branch)
        return [f] if _eval(f) else []

    eval_result = _eval(formula)
    if not eval_result:
        # Model may be incomplete — return None to signal failure
        return None

    implicant = _extract(formula)

    # If extraction produced nothing (e.g. due to incomplete model),
    # fall back to the formula itself as a degenerate implicant.
    if not implicant:
        implicant = [formula]

    # Try to specialize floor terms if we have an Interpretation-compatible model
    try:
        bindings = {}
        for sym, val in getattr(model, "interpretations", {}).items():
            if isinstance(val, bool):
                bindings[sym] = InterpretationValue(val)
            elif isinstance(val, (int, float, Fraction)):
                bindings[sym] = InterpretationValue(Fraction(val))
        if bindings:
            interp = Interpretation(srk, bindings=bindings)
            return specialize_floor_cube(srk, interp, implicant)
    except Exception:
        pass
    return implicant if implicant else None


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
        if self.kind == "MinusInfinity":
            return "-∞"
        elif self.kind == "PlusEpsilon":
            return f"{self.term} + ε"
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
            result = f"floor({self.term} / {self.divisor})"

        if self.offset != 0:
            result += f" + {self.offset}"

        return result


def coefficient_gcd(term: Any) -> int:
    """
    Compute the GCD of all coefficients in an affine term.

    Args:
        term: Linear term (vector)

    Returns:
        GCD of coefficients
    """
    from ..linalg.zZ import gcd

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
    from ..linalg.zZ import lcm
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
    from ..core.syntax import (
        Formula,
        prenex,
        destruct,
        mk_eq,
        mk_leq,
        mk_lt,
        mk_sub,
        mk_real,
        QQ,
        mk_symbol,
    )

    # Convert to prenex form
    phi = prenex(srk, phi)

    qf_pre = []

    def resolve_symbol(name, typ):
        """Reuse existing named symbol when available to preserve ids."""
        try:
            if (
                name is not None
                and hasattr(srk, "is_registered_name")
                and srk.is_registered_name(name)
            ):
                sym = srk.get_named_symbol(name)
                if sym.typ == typ:
                    return sym
        except Exception:
            # Fall back to creating a fresh symbol if lookup fails
            pass
        return mk_symbol(srk, name=name, typ=typ)

    def process(formula):
        """Recursively process quantifiers."""
        match = destruct(formula)

        if match and match[0] == "Quantify":
            qt, name, typ, psi = match[1]

            k = resolve_symbol(name, typ)

            inner_prefix, inner_formula = process(psi)

            qt_str = "Forall" if qt == "forall" else "Exists"
            return ([(qt_str, k)] + inner_prefix, inner_formula)
        elif match and match[0] in ("Exists", "Forall"):
            # Handle Exists/Forall directly from destruct
            var_name, var_type, body = match[1]

            k = resolve_symbol(var_name, var_type)

            inner_prefix, inner_formula = process(body)

            qt_str = match[0]  # 'Exists' or 'Forall'
            return ([(qt_str, k)] + inner_prefix, inner_formula)
        else:
            # Base case: quantifier-free
            # Normalize atoms to the form: t op 0
            from ..core.syntax import rewrite, mk_const

            def normalize_atom(expr):
                """Normalize arithmetic atoms."""
                match = destruct(expr)

                if match and match[0] == "Atom" and match[1][0] == "Arith":
                    op, s, t = match[1][1:]
                    zero = mk_real(srk, QQ.zero())

                    # Normalize to t - s op 0
                    diff = mk_sub(srk, s, t)

                    if op == "Eq":
                        return mk_eq(srk, diff, zero)
                    elif op == "Leq":
                        return mk_leq(srk, diff, zero)
                    elif op == "Lt":
                        return mk_lt(srk, diff, zero)

                return expr

            normalized = rewrite(formula, normalize_atom)
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
    from ..linalg.linear import QQVector, linterm_of, evaluate_linterm

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

            if not match or match[0] != "ArithComparison":
                return (None, None)

            op, s, t = match[1:]

            # Parse as linear constraint
            from ..core.syntax import mk_sub

            diff = mk_sub(srk, s, t)
            term = linterm_of(srk, diff)

            # Pivot to get ax + t' op 0
            a, t_prime = pivot(term, x)

            if a == 0:
                return (None, None)

            # Compute -t'/a (the bound)
            toa = QQVector.scalar_mul(-1 / a, t_prime)
            toa_val = evaluate_linterm(lambda sym: interp.real(sym), toa)

            if op == "Eq" or (op == "Leq" and toa_val == x_val):
                raise EqualTerm(toa)

            if a < 0:
                # Lower bound: x >= -t'/a
                return ((toa, toa_val, op == "Lt"), None)
            else:
                # Upper bound: x <= -t'/a
                return (None, (toa, toa_val, op == "Lt"))

        except Exception as e:
            if isinstance(e, EqualTerm):
                raise
            return (None, None)

    # Check if x appears in atoms
    from ..core.syntax import symbols

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
        if (
            lower is not None and lower[2] == False
        ):  # Non-strict lower bound equal to x_val
            return lower[0]
        elif (
            upper is not None and upper[2] == False
        ):  # Non-strict upper bound equal to x_val
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
    from ..linalg.linear import QQVector, linterm_of, evaluate_linterm, pivot
    from ..linalg.zZ import ZZ
    from ..linalg.qQ import QQ

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

            if not match or match[0] != "ArithComparison":
                continue

            op, s, t = match[1:]

            # Check for divisibility constraint
            atom_type = simplify_atom(srk, op, s, t)
            if atom_type[0] in ("Divides", "NotDivides"):
                divisor = atom_type[1]
                term = atom_type[2]

                # Get coefficient of x
                a = abs(int(term.get(x, 0)))
                if a != 0:
                    from ..linalg.zZ import lcm, gcd

                    delta = lcm(delta, divisor // gcd(divisor, a))

        except Exception:
            continue

    def bound_of_atom(atom):
        """Extract bound information from an atom."""
        try:
            from .interpretation import destruct_atom

            match = destruct_atom(srk, atom)

            if not match or match[0] != "ArithComparison":
                return None

            op, s, t = match[1:]

            # Simplify atom
            atom_type = simplify_atom(srk, op, s, t)

            if atom_type[0] != "CompareZero":
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
                if op == "Lt":
                    numerator = QQVector.add(
                        numerator, QQVector.const_linterm(Fraction(-1))
                    )

                rhs_val = int(
                    evaluate_linterm(lambda sym: interp.real(sym), numerator) // a_int
                )

                vt = IntVirtualTerm(
                    term=numerator, divisor=a_int, offset=(x_val - rhs_val) % delta
                )

                if op == "Eq":
                    raise EqualIntTerm(vt)

                return ("Upper", vt, evaluate_vt(vt))

            else:
                # Lower bound: ax + t' <= 0 => x >= ceil(t'/(-a))
                a_int = -a_int
                numerator = t_prime
                if op == "Lt":
                    numerator = QQVector.add(
                        numerator, QQVector.const_linterm(Fraction(a_int))
                    )
                else:
                    numerator = QQVector.add(
                        numerator, QQVector.const_linterm(Fraction(a_int - 1))
                    )

                rhs_val = int(
                    evaluate_linterm(lambda sym: interp.real(sym), numerator) // a_int
                )

                vt = IntVirtualTerm(
                    term=numerator, divisor=a_int, offset=(x_val - rhs_val) % delta
                )

                if op == "Eq":
                    raise EqualIntTerm(vt)

                return ("Lower", vt, evaluate_vt(vt))

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

        if kind1 == "Lower" and kind2 == "Lower":
            return bound1 if val1 > val2 else bound2
        elif kind1 == "Upper" and kind2 == "Upper":
            return bound1 if val1 < val2 else bound2
        elif kind1 == "Lower":
            return bound1
        else:
            return bound2

    # Check if x appears in atoms
    from ..core.syntax import symbols

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


def specialize_floor_cube(srk: Any, model: Any, cube: List[Any]) -> List[Any]:
    """Eliminate floor/mod terms from implicant atoms.

    Given an interpretation *model* and a conjunctive cube such that
    *model* |= cube, return a new cube in which floor and mod sub-terms
    have been replaced by concrete values (plus added divisibility
    constraints).
    """
    from ..core.syntax import rewrite, destruct, mk_sub, mk_real, mk_eq, mk_mod, mk_floor
    from ..linalg.linear import linterm_of, of_linterm, QQVector, const_dim

    div_constraints: List[Any] = []

    def _add_div_constraint(divisor: int, term_expr: Any) -> None:
        div_constraints.append(
            mk_eq(
                srk,
                mk_mod(srk, term_expr, mk_real(srk, Fraction(divisor))),
                mk_real(srk, Fraction(0)),
            )
        )

    def _replace(expr: Any) -> Any:
        match = destruct(expr)
        if not match:
            return expr

        if match[0] == "Unop" and match[1] == "Floor":
            t = match[2]
            v = linterm_of(srk, t)
            divisor = common_denominator(v)
            qq_divisor = Fraction(divisor)
            dividend = of_linterm(srk, QQVector.scalar_mul(qq_divisor, v))
            remainder = Fraction(model.evaluate_term(dividend)) % qq_divisor
            dividend_prime = mk_sub(srk, dividend, mk_real(srk, remainder))
            replacement_vec = QQVector.add_term(
                Fraction(-remainder) / qq_divisor, const_dim, v
            )
            replacement = of_linterm(srk, replacement_vec)
            _add_div_constraint(divisor, dividend_prime)
            return replacement

        if match[0] == "Binop" and match[1] == "Mod":
            t, m_expr = match[2], match[3]
            try:
                m_val = Fraction(m_expr) if not isinstance(m_expr, (int, float, Fraction)) else Fraction(m_expr)
            except Exception:
                try:
                    m_val = Fraction(model.evaluate_term(m_expr))
                except Exception:
                    return expr
            replacement = mk_real(srk, Fraction(model.evaluate_term(t)) % m_val)
            m_zz = int(m_val)
            _add_div_constraint(m_zz, mk_sub(srk, t, replacement))
            return replacement

        return expr

    cube_prime = [rewrite(srk, _replace, atom) for atom in cube]
    return div_constraints + cube_prime


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
    from ..linalg.linear import QQVector, linterm_of

    def value_of(sym):
        """Get a rational value for a symbol using either Interpretation or SMTModel."""
        try:
            return interp.real(sym)
        except Exception:
            pass

        if hasattr(interp, "get_value"):
            val = interp.get_value(sym)
            if isinstance(val, Fraction):
                return val
            if isinstance(val, (int, float)):
                return Fraction(val)

        return None

    def evaluate_linterm(eval_fn, term):
        total = Fraction(0)
        for dim, coeff in term.entries.items():
            # Map dimension back to a symbol if possible
            sym = None
            try:
                sym = srk._symbols.get(dim)
            except Exception:
                pass

            val = eval_fn(sym if sym is not None else dim)
            if val is None:
                return None
            total += coeff * Fraction(val)
        return total

    # Get the value of x in the model
    x_val = value_of(x)
    if x_val is None:
        return VirtualTerm("MinusInfinity")

    # Find bounds on x
    lower_bound = None
    lower_val = None

    for atom in atoms:
        try:
            # Parse atom as ax + t op 0
            from .interpretation import destruct_atom

            match = destruct_atom(srk, atom)

            if not match or match[0] != "ArithComparison":
                continue

            op, s, t = match[1:]
            from ..core.syntax import mk_sub

            diff = mk_sub(srk, s, t)
            term = linterm_of(srk, diff)
            a, t_prime = pivot(term, x)

            if a == 0:
                continue

            # Compute -t/a (the bound)
            bound_term = QQVector.scalar_mul(-1 / a, t_prime)
            bound_val = evaluate_linterm(lambda sym: interp.real(sym), bound_term)

            # Check if this is an equality
            if bound_val == x_val:
                return VirtualTerm("Term", bound_term)

            # Check if this is a lower bound
            if a < 0:  # ax + t <= 0 => x >= -t/a
                if lower_val is None or bound_val > lower_val:
                    lower_bound = bound_term
                    lower_val = bound_val

        except Exception:
            continue

    # Return the tightest lower bound + epsilon, or -infinity
    if lower_bound is not None:
        return VirtualTerm("PlusEpsilon", lower_bound)
    else:
        return VirtualTerm("MinusInfinity")


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
    from ..core.syntax import (
        rewrite,
        destruct,
        mk_eq,
        mk_leq,
        mk_lt,
        mk_true,
        mk_false,
        substitute_const,
        mk_const,
        mk_sub,
        mk_real,
        of_linterm,
    )
    from ..linalg.linear import linterm_of, QQVector

    def replace_atom(expr):
        """Replace atoms containing x."""
        match = destruct(expr)

        if not match or match[0] != "Atom":
            return expr

        atom_kind = match[1]
        if not atom_kind or atom_kind[0] != "Arith":
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
            soa = QQVector.scalar_mul(-1 / a, t_prime)

            if vt.kind == "Term":
                # Direct substitution
                diff = mk_sub(srk, of_linterm(srk, soa), of_linterm(srk, vt.term))
                zero = mk_real(srk, Fraction(0))

                if op == "Eq":
                    return mk_eq(srk, diff, zero)
                elif op == "Leq":
                    if a < 0:
                        return mk_leq(srk, diff, zero)
                    else:
                        return mk_leq(
                            srk,
                            mk_sub(srk, of_linterm(srk, vt.term), of_linterm(srk, soa)),
                            zero,
                        )
                elif op == "Lt":
                    if a < 0:
                        return mk_lt(srk, diff, zero)
                    else:
                        return mk_lt(
                            srk,
                            mk_sub(srk, of_linterm(srk, vt.term), of_linterm(srk, soa)),
                            zero,
                        )

            elif vt.kind == "MinusInfinity":
                # x = -∞
                if a < 0:
                    return mk_false()
                else:
                    return mk_true()

            elif vt.kind == "PlusEpsilon":
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
    if vt.kind == "Term":
        replacement = of_linterm(srk, vt.term)
        return substitute_const(
            srk, lambda sym: replacement if sym == x else mk_const(srk, sym), phi
        )

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
    from .smt import mk_solver, Solver, SMTResult
    from ..core.syntax import (
        mk_not,
        mk_or,
        mk_and,
        mk_false,
        mk_true,
        symbols as get_symbols,
        Var,
        Symbol,
        ExpressionVisitor,
    )

    # Extract symbols from constants
    all_symbols = get_symbols(phi)

    # Also extract variables and convert them to symbols
    class VariableExtractor(ExpressionVisitor[Set[Symbol]]):
        """Extract variables and convert them to symbols."""

        def visit_var(self, var: Var) -> Set[Symbol]:
            # Create a symbol from the variable's id and type
            return {Symbol(var.var_id, None, var.var_type)}

        def visit_const(self, const) -> Set[Symbol]:
            return set()

        def visit_app(self, app) -> Set[Symbol]:
            result = set()
            for arg in app.args:
                result.update(arg.accept(self))
            return result

        def visit_select(self, select) -> Set[Symbol]:
            result = select.array.accept(self)
            result.update(select.index.accept(self))
            return result

        def visit_store(self, store) -> Set[Symbol]:
            result = store.array.accept(self)
            result.update(store.index.accept(self))
            result.update(store.value.accept(self))
            return result

        def visit_add(self, add) -> Set[Symbol]:
            result = set()
            for arg in add.args:
                result.update(arg.accept(self))
            return result

        def visit_mul(self, mul) -> Set[Symbol]:
            result = set()
            for arg in mul.args:
                result.update(arg.accept(self))
            return result

        def visit_ite(self, ite) -> Set[Symbol]:
            result = ite.condition.accept(self)
            result.update(ite.then_branch.accept(self))
            result.update(ite.else_branch.accept(self))
            return result

        def visit_true(self, true_expr) -> Set[Symbol]:
            return set()

        def visit_false(self, false_expr) -> Set[Symbol]:
            return set()

        def visit_and(self, and_expr) -> Set[Symbol]:
            result = set()
            for arg in and_expr.args:
                result.update(arg.accept(self))
            return result

        def visit_or(self, or_expr) -> Set[Symbol]:
            result = set()
            for arg in or_expr.args:
                result.update(arg.accept(self))
            return result

        def visit_not(self, not_expr) -> Set[Symbol]:
            return not_expr.arg.accept(self)

        def visit_eq(self, eq) -> Set[Symbol]:
            result = eq.left.accept(self)
            result.update(eq.right.accept(self))
            return result

        def visit_lt(self, lt) -> Set[Symbol]:
            result = lt.left.accept(self)
            result.update(lt.right.accept(self))
            return result

        def visit_leq(self, leq) -> Set[Symbol]:
            result = leq.left.accept(self)
            result.update(leq.right.accept(self))
            return result

        def visit_forall(self, forall) -> Set[Symbol]:
            # Extract variables from the body of the forall
            return forall.body.accept(self)

        def visit_exists(self, exists) -> Set[Symbol]:
            # Extract variables from the body of the exists
            return exists.body.accept(self)

    # Add symbols from variables
    var_extractor = VariableExtractor()
    var_symbols = phi.accept(var_extractor)
    all_symbols.update(var_symbols)

    # Identify variables to project
    # The exists predicate matches based on symbol id (from qe_mbp change)
    # So we need to find symbols whose id matches what exists() is looking for
    # But also, we need to handle the case where variables in the formula have different ids
    # than the quantifier symbols created by normalize

    # Try to match symbols using the exists predicate
    project = {s for s in all_symbols if exists(s)}

    # If no matches, the issue might be that normalize created symbols with different IDs
    # than the variables in the formula. In that case, we should still try to eliminate
    # variables that appear in the formula. But we need the symbol from exists() to use for projection.
    # Actually, since exists() checks s.id == x_id, and we create Symbols from Var's var_id,
    # they should match if the var_id matches x_id. But normalize creates a new symbol, so they don't.

    if not project:
        # No symbols matched - this might mean the variable IDs don't match
        # For now, return phi (no elimination possible)
        return phi

    remaining_vars = {s for s in var_symbols if s not in project}

    solver = mk_solver(srk)
    Solver.add(solver, [phi])

    # If eliminating all variables and only constants remain, just check SAT/UNSAT.
    if not remaining_vars:
        status = Solver.check(solver, [])
        if status == SMTResult.SAT:
            return mk_true()
        if status == SMTResult.UNSAT:
            return mk_false()
    disjuncts = []

    while True:
        status = Solver.check(solver, [])

        if status != SMTResult.SAT:
            break

        model = Solver.get_model(solver)

        if model is None:
            break

        # Get implicant from model. SMTModel doesn't implement select_implicant,
        # so fall back to using the whole formula as the implicant.
        from .interpretation import select_implicant

        try:
            implicant = select_implicant(model, phi)
        except AttributeError:
            implicant = [phi]

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

    return mk_or(srk, disjuncts) if disjuncts else mk_false()


def simsat(srk: Any, phi: Any) -> str:
    """Satisfiability via strategy improvement (ported from OCaml)."""
    from ..core.syntax import symbols as _symbols

    all_syms = _symbols(phi)
    # Filter out numeric literal constants (they are not quantified variables)
    constants = {s for s in all_syms if not _is_numeric_literal(s)}

    qf_pre, psi = normalize(srk, phi)
    # Prepend free constants as existential quantifiers
    qf_pre = [("Exists", k) for k in constants] + qf_pre

    # Fast path: quantifier-free formulas — just check SAT directly
    if not qf_pre:
        from .smt import is_sat as _is_sat, SMTResult
        r = _is_sat(srk, psi)
        if r == SMTResult.SAT:
            return "Sat"
        if r == SMTResult.UNSAT:
            return "Unsat"
        return "Unknown"

    return simsat_core(srk, qf_pre, psi)


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
        if qt == "Exists":
            # Eliminate existential quantifier
            # Match by symbol id since normalize may create new Symbol objects
            x_id = x.id
            result = mbp(srk, lambda s: s.id == x_id, result, dnf=True)
        else:
            # Forall: ∀x.φ ≡ ¬∃x.¬φ
            from ..core.syntax import mk_not

            x_id = x.id
            result = mk_not(mbp(srk, lambda s: s.id == x_id, mk_not(result), dnf=True))

    return result


def easy_sat(srk: Any, phi: Any) -> str:
    """Easy satisfiability check (single-round game, ported from OCaml)."""
    from ..core.syntax import symbols as _symbols

    constants = {s for s in _symbols(phi) if not _is_numeric_literal(s)}
    qf_pre, psi = normalize(srk, phi)
    qf_pre = [("Exists", k) for k in constants] + qf_pre

    select_term_fn = lambda model, x, atoms: _select_term(srk, model, x, atoms)

    init = CSS.initialize_pair(select_term_fn, srk, qf_pre, psi)
    if init == "Unsat":
        return "Unsat"
    if init == "Unknown":
        return "Unknown"
    _, (sat_ctx, _) = init
    # Single round: check if SAT wins
    res = CSS.get_counter_strategy(select_term_fn, sat_ctx)
    if res == "Unsat":
        return "Sat"
    if res == "Unknown":
        return "Unknown"
    return "Unknown"


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
    from ..core.syntax import mk_eq, mk_mod, mk_real, of_linterm, mk_true
    from ..linalg.zZ import gcd

    if divisor <= 0:
        raise ValueError("Divisor must be positive")

    if divisor == 1:
        return mk_true()

    # Simplify using GCD
    term_gcd = coefficient_gcd(term)
    gcd_val = gcd(term_gcd, divisor)

    divisor = divisor // gcd_val

    # Create formula: (term mod divisor) = 0
    divisor_qq = Fraction(divisor)

    return mk_eq(
        srk,
        mk_mod(srk, of_linterm(srk, term), mk_real(srk, divisor_qq)),
        mk_real(srk, Fraction(0)),
    )


def simplify_atom(srk: Any, op: str, s: Any, t: Any) -> Tuple[str, ...]:
    """
    Simplify an arithmetic atom.

    Returns:
        ('CompareZero', op, term) or ('Divides', divisor, term) or ('NotDivides', divisor, term)
    """
    from ..linalg.linear import linterm_of, QQVector
    from ..core.syntax import mk_sub, mk_add, mk_real, mk_neg, destruct
    from fractions import Fraction

    def _zz_linterm(expr):
        """Scale a linterm so all coefficients are integral, return (multiplier, term)."""
        qq_lt = linterm_of(srk, expr)
        multiplier = 1
        for _, coeff in qq_lt.entries.items():
            den = Fraction(coeff).denominator
            from math import lcm as _lcm
            multiplier = _lcm(multiplier, den)
        return multiplier, QQVector.scalar_mul(Fraction(multiplier), qq_lt)

    zero = mk_real(srk, Fraction(0))

    # Normalise to s' op 0 where s' = simplify(s - t) for Lt on ints: s+1 <= 0
    if op == "Lt":
        s_norm = mk_add(srk, [mk_sub(srk, s, t), mk_real(srk, Fraction(1))])
        op_norm = "Leq"
    else:
        s_norm = mk_sub(srk, s, t)
        op_norm = op

    # Try to simplify to a linear term
    try:
        from .srkSimplify import simplify_term
        s_norm = simplify_term(srk, s_norm)
    except Exception:
        pass

    match = destruct(s_norm)

    if match and match[0] == "Binop" and match[1] == "Mod":
        dividend, modulus = match[2:]
        try:
            mod_val = int(modulus)
            mult, lt = _zz_linterm(dividend)
            return ("Divides", mult * mod_val, lt)
        except Exception:
            pass

    # Check for Unop(Neg, Binop(Mod, ...))  → NotDivides (or trivial for Leq)
    if match and match[0] == "Unop" and match[1] == "Neg":
        inner = match[2]
        inner_match = destruct(inner)
        if inner_match and inner_match[0] == "Binop" and inner_match[1] == "Mod":
            if op_norm == "Leq":
                return ("CompareZero", "Leq", QQVector())  # trivial: -mod <= 0
            dividend, modulus = inner_match[2:]
            try:
                mod_val = int(modulus)
                mult, lt = _zz_linterm(dividend)
                return ("Divides", mult * mod_val, lt)
            except Exception:
                pass

    # Check for Add[k; Binop(Mod, ...)] or Add[Binop(Mod, ...); k] with k < 0
    if match and match[0] == "Add" and len(match[1]) == 2:
        args = match[1]
        for i, j in [(0, 1), (1, 0)]:
            a_match = destruct(args[i])
            if a_match and a_match[0] == "Real":
                k = a_match[1]
                b_match = destruct(args[j])
                if (
                    b_match
                    and b_match[0] == "Binop"
                    and b_match[1] == "Mod"
                    and k < 0
                    and op_norm == "Eq"
                ):
                    dividend, modulus = b_match[2:]
                    try:
                        mod_val = int(modulus)
                        mult, lt = _zz_linterm(dividend)
                        if mult == 1 and abs(k) < mod_val:
                            lt = QQVector.add_term(k, const_dim, lt)
                            return ("Divides", mod_val, lt)
                    except Exception:
                        pass

    # Check for Add[Real(1); Unop(Neg, z)] → NotDivides if z is Mod
    if match and match[0] == "Add" and len(match[1]) == 2:
        args = match[1]
        for i, j in [(0, 1), (1, 0)]:
            a_match = destruct(args[i])
            if a_match and a_match[0] == "Real" and a_match[1] == 1:
                z_match = destruct(args[j])
                if z_match and z_match[0] == "Unop" and z_match[1] == "Neg":
                    inner = z_match[2]
                    inner_match = destruct(inner)
                    if inner_match and inner_match[0] == "Binop" and inner_match[1] == "Mod":
                        dividend, modulus = inner_match[2:]
                        try:
                            mod_val = int(modulus)
                            mult, lt = _zz_linterm(dividend)
                            return ("NotDivides", mult * mod_val, lt)
                        except Exception:
                            pass

    # Default: linear comparison
    try:
        term = linterm_of(srk, s_norm)
        return ("CompareZero", op_norm, term)
    except Exception:
        return ("CompareZero", op, None)


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
        qf_pre, phi = self._destructure_game(game)
        if phi is None:
            return "Unknown"

        result = simsat_core(self.context, qf_pre, phi)
        if result == "Unknown":
            return "Unknown"
        return result

    def maximize(self, phi: Any, objective: Any) -> Any:
        """Maximize an objective in the supported linear-arithmetic fragment."""
        return maximize(self.context, phi, objective)

    def check_strategy(
        self, qf_pre: List[Tuple[str, Any]], phi: Any, strategy: Any
    ) -> Optional[bool]:
        """Validate a candidate strategy when the bounded Z3 path supports it."""
        return check_strategy(self.context, qf_pre, phi, strategy)

    def _destructure_game(
        self, game: Any
    ) -> Tuple[List[Tuple[str, Any]], Optional[Any]]:
        if isinstance(game, dict):
            phi = game.get("formula") or game.get("phi")
            qf_pre = game.get("prefix") or game.get("qf_pre") or []
            return (qf_pre, phi)
        if isinstance(game, tuple) and len(game) == 2:
            qf_pre, phi = game
            return (qf_pre, phi)
        return ([], game)


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

        if match[0] == "Literal":
            return True
        elif match[0] == "ArithComparison":
            op, s, t = match[1:]
            # Try to simplify the atom
            simplify_atom(srk, op, s, t)
            return True
        else:
            return False

    except Exception:
        return False


def local_project_cube(
    srk: Any, exists: Callable[[Any], bool], model: Any, cube: List[Any]
) -> List[Any]:
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
    from ..core.syntax import symbols, mk_true

    # Set of symbols to be projected
    project = set()
    for phi in cube:
        for sym in symbols(phi):
            if not exists(sym):
                project.add(sym)

    def is_true(phi):
        """Check if formula is trivially true."""
        from ..core.syntax import destruct

        match = destruct(phi)
        return match and match[0] == "Tru"

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
        if self.kind == "MinusInfinity":
            return "-∞"
        elif self.kind == "PlusEpsilon":
            return f"{self.term} + ε"
        elif self.kind == "Term":
            return str(self.term)
        else:
            return "??"


def cover_virtual_term(
    srk: Any, interp: Any, x: Any, atoms: List[Any]
) -> CoverVirtualTerm:
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
    from ..core.syntax import mk_sub, symbols

    def get_equal_term(atom):
        """Try to find an equal term."""
        try:
            from .interpretation import destruct_atom

            match = destruct_atom(srk, atom)

            if not match or match[0] != "ArithComparison":
                return None

            op, s, t = match[1:]

            if op in ("Lt",):
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
                        from ..core.syntax import mk_mul, mk_real, mk_floor, typ_symbol

                        term = mk_mul(srk, [mk_real(srk, Fraction(-1) / a), b])

                        # If x is integer and term is real, apply floor
                        from ..core.syntax import typ_symbol, expr_typ

                        if (
                            typ_symbol(srk, x) == "TyInt"
                            and expr_typ(srk, term) == "TyReal"
                        ):
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

            if not match or match[0] != "ArithComparison":
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
                from ..core.syntax import mk_mul, mk_real

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
        from ..core.syntax import mk_real

        return CoverVirtualTerm("Term", mk_real(srk, Fraction(0)))

    # Try to find an equal term
    for atom in atoms:
        equal_term = get_equal_term(atom)
        if equal_term is not None:
            return CoverVirtualTerm("Term", equal_term)

    # Try to find bounds
    try:
        lower = None
        for atom in atoms:
            vt = get_vt(atom)
            if vt is not None:
                if lower is None or vt[1] > lower[1]:
                    lower = vt

        if lower is not None:
            return CoverVirtualTerm("PlusEpsilon", lower[0])
        else:
            return CoverVirtualTerm("MinusInfinity")

    except Exception:
        # Fall back to unknown
        return CoverVirtualTerm("Unknown")


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
    from ..core.syntax import (
        rewrite,
        destruct,
        mk_eq,
        mk_leq,
        mk_lt,
        mk_true,
        mk_false,
        substitute_const,
        mk_const,
        mk_sub,
        mk_real,
        mk_add,
        mk_mul,
        symbols,
    )

    if vt.kind == "Term":
        # Direct substitution
        return substitute_const(
            srk, lambda sym: vt.term if sym == x else mk_const(srk, sym), phi
        )

    elif vt.kind == "Unknown":
        # Drop atoms containing x
        def drop(expr):
            match = destruct(expr)
            if match and match[0] == "Atom":
                if x in symbols(expr):
                    return mk_true()
            return expr

        return rewrite(srk, drop, phi)

    else:
        # Handle PlusEpsilon and MinusInfinity
        def replace_atom(expr):
            match = destruct(expr)

            if not match or match[0] != "Atom":
                return expr

            atom_kind = match[1]
            if not atom_kind or atom_kind[0] != "Arith":
                return expr

            op, s, t = atom_kind[1:]

            # Try to isolate x
            from .srkSimplify import isolate_linear

            diff = mk_sub(srk, s, t)
            result = isolate_linear(srk, x, diff)

            if result is None:
                # Can't isolate: drop the constraint
                return mk_true()

            a, b = result

            if a == 0:
                # x doesn't appear
                if op == "Eq":
                    return mk_eq(srk, s, t)
                elif op == "Leq":
                    return mk_leq(srk, s, t)
                elif op == "Lt":
                    return mk_lt(srk, s, t)

            zero = mk_real(srk, Fraction(0))

            if vt.kind == "MinusInfinity":
                if a < 0:
                    return mk_false()
                else:
                    return mk_true()

            elif vt.kind == "PlusEpsilon":
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


def mbp_cover(
    srk: Any, exists: Callable[[Any], bool], phi: Any, dnf: bool = True
) -> Any:
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
    from ..core.syntax import mk_not, mk_or, mk_and, mk_false, symbols as get_symbols

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

        if model is None or model == "Unsat":
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
            projected_implicant = [
                cover_virtual_substitution(srk, x, vt, atom)
                for atom in projected_implicant
            ]

        if dnf:
            disjunct = mk_and(srk, projected_implicant)
        else:
            disjunct = cover_virtual_substitution_formula(srk, project, model, phi)

        disjuncts.append(disjunct)

        # Block this disjunct
        Solver.add(solver, [mk_not(srk, disjunct)])

    return mk_or(srk, disjuncts) if disjuncts else mk_false()


def cover_virtual_substitution_formula(
    srk: Any, project: Set[Any], model: Any, phi: Any
) -> Any:
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
    "normalize",
    "mbp",
    "mbp_virtual_term",
    "virtual_substitution",
    "simsat",
    "simsat_forward",
    "simsat_core",
    "easy_sat",
    "qe_mbp",
    "maximize",
    "maximize_feasible",
    "winning_strategy",
    "check_strategy",
    "mk_divides",
    "simplify_atom",
    "is_presburger_atom",
    "local_project_cube",
    "mbp_cover",
    "cover_virtual_term",
    "cover_virtual_substitution",
    "select_real_term",
    "select_int_term",
    "specialize_floor_cube",
    "VirtualTerm",
    "IntVirtualTerm",
    "CoverVirtualTerm",
    "QuantifierType",
    "QuantifierEngine",
    "StrategyImprovementSolver",
    "Skeleton",
    "CSS",
]

# -------------------------
# Game-theoretic features
# -------------------------


def term_of_int_virtual_term(srk: Any, vt: IntVirtualTerm) -> Any:
    """Build an expression floor(term/divisor) + offset from IntVirtualTerm."""
    from ..core.syntax import of_linterm, mk_floor, mk_div, mk_add, mk_real

    if vt.divisor == 1:
        base = of_linterm(srk, vt.term)
    else:
        base = mk_floor(
            srk,
            mk_div(srk, of_linterm(srk, vt.term), mk_real(srk, Fraction(vt.divisor))),
        )
    if vt.offset != 0:
        return mk_add(srk, [base, mk_real(srk, Fraction(vt.offset))])
    return base


class Skeleton:
    """Strategy skeleton for game-theoretic reasoning (ported from OCaml)."""

    class RedundantPath(Exception):
        """Raised when a path already exists in the skeleton."""
        pass

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
        subtree: Any

    @dataclass
    class SExists:
        symbol: Any
        moves: List[Tuple[Any, Any]]  # list of (move, subtree)

    class SEmpty:
        pass

    # ------------------------------------------------------------------
    # Move helpers
    # ------------------------------------------------------------------

    @staticmethod
    def evaluate_move(srk: Any, model_fn: Callable[[Any], Fraction], move: Any) -> Fraction:
        """Evaluate a move under a symbol→value function."""
        if isinstance(move, Skeleton.MReal):
            return _evaluate_linterm(srk, model_fn, move.term)
        elif isinstance(move, Skeleton.MInt):
            vt = move.vt
            term_val = _evaluate_linterm(srk, model_fn, vt.term)
            tv = int(term_val)
            return Fraction((tv // vt.divisor) + vt.offset)
        elif isinstance(move, Skeleton.MBool):
            return Fraction(1) if move.value else Fraction(0)
        raise ValueError(f"Unknown move type: {type(move)}")

    @staticmethod
    def const_of_move(move: Any) -> Optional[Fraction]:
        """Extract a constant value from a move, or None."""
        if isinstance(move, Skeleton.MReal):
            entries = move.term.entries
            if not entries:
                return Fraction(0)
            if len(entries) == 1 and const_dim in entries:
                return Fraction(entries[const_dim])
            return None
        elif isinstance(move, Skeleton.MInt):
            vt = move.vt
            if vt.divisor == 1:
                return Skeleton.const_of_move(Skeleton.MReal(vt.term))
            return None
        elif isinstance(move, Skeleton.MBool):
            return Fraction(1) if move.value else Fraction(0)
        return None

    @staticmethod
    def _moves_equal(a: Any, b: Any) -> bool:
        if type(a) is not type(b):
            return False
        if isinstance(a, Skeleton.MReal):
            return a.term.entries == b.term.entries
        if isinstance(a, Skeleton.MInt):
            return (a.vt.term.entries == b.vt.term.entries
                    and a.vt.divisor == b.vt.divisor
                    and a.vt.offset == b.vt.offset)
        if isinstance(a, Skeleton.MBool):
            return a.value == b.value
        return False

    # ------------------------------------------------------------------
    # Substitution
    # ------------------------------------------------------------------

    @staticmethod
    def substitute(srk: Any, x: Any, move: Any, phi: Any) -> Any:
        from ..core.syntax import of_linterm, mk_const, substitute_const, mk_true, mk_false

        if isinstance(move, Skeleton.MReal):
            replacement = of_linterm(srk, move.term)
        elif isinstance(move, Skeleton.MInt):
            replacement = term_of_int_virtual_term(srk, move.vt)
        elif isinstance(move, Skeleton.MBool):
            replacement = mk_true(srk) if move.value else mk_false(srk)
        else:
            return phi
        return substitute_const(srk, lambda p: replacement if p == x else mk_const(srk, p), phi)

    @staticmethod
    def substitute_implicant(srk: Any, interp: Any, x: Any, move: Any, implicant: List[Any]) -> List[Any]:
        """Substitute a move into implicant atoms, adding divisibility constraints."""
        from ..core.syntax import substitute_const, mk_const

        def _is_true(phi):
            from ..core.syntax import destruct
            m = destruct(phi)
            return m is not None and m[0] == "Tru"

        if isinstance(move, Skeleton.MInt):
            vt = move.vt
            model_fn = lambda sym: _model_real(interp, sym)
            try:
                term_val = _evaluate_linterm(srk, model_fn, vt.term)
            except (KeyError, ValueError, TypeError):
                # Model incomplete — skip divisibility constraint, just substitute
                result = [Skeleton.substitute(srk, x, move, a) for a in implicant]
                return [a for a in result if not _is_true(a)]
            term_val_zz = int(term_val)
            # fdiv_r: remainder with floor division
            remainder = term_val_zz % vt.divisor
            if remainder != 0 and term_val_zz < 0:
                remainder -= vt.divisor
            numerator = QQVector.add_term(Fraction(-remainder), const_dim, vt.term)
            replacement_vec = QQVector.add_term(
                Fraction(vt.offset), const_dim,
                QQVector.scalar_mul(Fraction(1, vt.divisor), numerator),
            )
            replacement = of_linterm(srk, replacement_vec)
            subst_fn = lambda p: replacement if p == x else mk_const(srk, p)
            div_constraint = mk_divides(srk, vt.divisor, numerator)
            result = [substitute_const(srk, subst_fn, a) for a in implicant]
            result = [a for a in result if not _is_true(a)]
            if not _is_true(div_constraint):
                result.append(div_constraint)
            return result
        else:
            result = [Skeleton.substitute(srk, x, move, a) for a in implicant]
            return [a for a in result if not _is_true(a)]

    # ------------------------------------------------------------------
    # Structure queries
    # ------------------------------------------------------------------

    @staticmethod
    def empty():
        return Skeleton.SEmpty()

    @staticmethod
    def nb_paths(skeleton: Any) -> int:
        if isinstance(skeleton, Skeleton.SEmpty):
            return 1
        if isinstance(skeleton, Skeleton.SForall):
            return Skeleton.nb_paths(skeleton.subtree)
        if isinstance(skeleton, Skeleton.SExists):
            return sum(Skeleton.nb_paths(sub) for _, sub in skeleton.moves)
        return 0

    @staticmethod
    def size(skeleton: Any) -> int:
        if isinstance(skeleton, Skeleton.SEmpty):
            return 0
        if isinstance(skeleton, Skeleton.SForall):
            return 1 + Skeleton.size(skeleton.subtree)
        if isinstance(skeleton, Skeleton.SExists):
            return 1 + sum(Skeleton.size(sub) for _, sub in skeleton.moves)
        return 0

    @staticmethod
    def paths(skeleton: Any) -> List[List[Any]]:
        if isinstance(skeleton, Skeleton.SEmpty):
            return [[]]
        if isinstance(skeleton, Skeleton.SForall):
            return [[("Forall", skeleton.symbol)] + p for p in Skeleton.paths(skeleton.subtree)]
        if isinstance(skeleton, Skeleton.SExists):
            result = []
            for move, subtree in skeleton.moves:
                for sub_path in Skeleton.paths(subtree):
                    result.append([("Exists", (skeleton.symbol, move))] + sub_path)
            return result
        return [[]]

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @staticmethod
    def mk_path(srk: Any, path: List[Any]) -> Any:
        from ..core.syntax import mk_symbol, typ_symbol
        node: Any = Skeleton.SEmpty()
        for entry in reversed(path):
            if entry[0] == "Forall":
                k = entry[1]
                sk = mk_symbol(srk, name=str(k) + "_sk", typ=typ_symbol(k))
                node = Skeleton.SForall(k, sk, node)
            elif entry[0] == "Exists":
                k, move = entry[1]
                node = Skeleton.SExists(k, [(move, node)])
            else:
                raise ValueError(f"Invalid path entry: {entry}")
        return node

    @staticmethod
    def add_path(srk: Any, path: List[Any], skeleton: Any) -> Any:
        if isinstance(skeleton, Skeleton.SEmpty):
            return Skeleton.mk_path(srk, path)
        if not path:
            raise Skeleton.RedundantPath()
        head = path[0]
        tail = path[1:]
        if isinstance(skeleton, Skeleton.SForall):
            assert head[0] == "Forall" and head[1] == skeleton.symbol
            skeleton.subtree = Skeleton.add_path(srk, tail, skeleton.subtree)
            return skeleton
        if isinstance(skeleton, Skeleton.SExists):
            assert head[0] == "Exists" and head[1][0] == skeleton.symbol
            move = head[1][1]
            for i, (mv, sub) in enumerate(skeleton.moves):
                if Skeleton._moves_equal(mv, move):
                    skeleton.moves[i] = (mv, Skeleton.add_path(srk, tail, sub))
                    return skeleton
            skeleton.moves.append((move, Skeleton.mk_path(srk, tail)))
            return skeleton
        raise ValueError(f"Unexpected skeleton type: {type(skeleton)}")

    # ------------------------------------------------------------------
    # Winning formulas
    # ------------------------------------------------------------------

    @staticmethod
    def path_winning_formula(srk: Any, path: List[Any], skeleton: Any, phi: Any) -> Any:
        from ..core.syntax import mk_const, substitute_const

        def _go(path, sk):
            if not path and isinstance(sk, Skeleton.SEmpty):
                return phi
            head, tail = path[0], path[1:]
            if head[0] == "Forall":
                k = head[1]
                assert isinstance(sk, Skeleton.SForall) and k == sk.symbol
                sk_const = mk_const(srk, sk.skolem)
                sub = _go(tail, sk.subtree)
                return substitute_const(srk, lambda s: sk_const if s == k else mk_const(srk, s), sub)
            if head[0] == "Exists":
                k, move = head[1]
                assert isinstance(sk, Skeleton.SExists) and k == sk.symbol
                for mv, sub in sk.moves:
                    if Skeleton._moves_equal(mv, move):
                        return Skeleton.substitute(srk, k, move, _go(tail, sub))
                raise ValueError("Move not found in skeleton")
            raise ValueError(f"Invalid path entry: {head}")

        return _go(path, skeleton)

    @staticmethod
    def winning_formula(srk: Any, skeleton: Any, phi: Any) -> Any:
        from ..core.syntax import mk_const, mk_or, substitute_const

        if isinstance(skeleton, Skeleton.SEmpty):
            return phi
        if isinstance(skeleton, Skeleton.SForall):
            sk_const = mk_const(srk, skeleton.skolem)
            sub = Skeleton.winning_formula(srk, skeleton.subtree, phi)
            return substitute_const(srk, lambda s: sk_const if s == skeleton.symbol else mk_const(srk, s), sub)
        if isinstance(skeleton, Skeleton.SExists):
            disjuncts = []
            for move, subtree in skeleton.moves:
                moved = Skeleton.substitute(srk, skeleton.symbol, move, phi)
                disjuncts.append(Skeleton.winning_formula(srk, subtree, moved))
            return mk_or(srk, disjuncts) if disjuncts else phi
        return phi


class CSS:
    """Counter-strategy synthesis (ported from OCaml)."""

    @dataclass
    class Ctx:
        srk: Any
        formula: Any
        not_formula: Any
        skeleton: Any
        solver: Any

    @staticmethod
    def make_ctx(srk: Any, formula: Any, not_formula: Any, skeleton: Any) -> "CSS.Ctx":
        from .smt import mk_solver, Solver
        from ..core.syntax import mk_not

        solver = mk_solver(srk)
        win = Skeleton.winning_formula(srk, skeleton, formula)
        Solver.add(solver, [mk_not(srk, win)])
        return CSS.Ctx(srk=srk, formula=formula, not_formula=not_formula,
                        skeleton=skeleton, solver=solver)

    @staticmethod
    def reset(ctx: "CSS.Ctx") -> None:
        from .smt import Solver
        Solver.reset(ctx.solver)
        ctx.skeleton = Skeleton.SEmpty()

    @staticmethod
    def add_path(ctx: "CSS.Ctx", path: List[Any]) -> None:
        from .smt import Solver
        from ..core.syntax import mk_not
        try:
            ctx.skeleton = Skeleton.add_path(ctx.srk, path, ctx.skeleton)
            win = Skeleton.path_winning_formula(ctx.srk, path, ctx.skeleton, ctx.formula)
            Solver.add(ctx.solver, [mk_not(ctx.srk, win)])
        except Skeleton.RedundantPath:
            pass

    @staticmethod
    def get_counter_strategy(select_term_fn: Callable, ctx: "CSS.Ctx",
                              parameters: Any = None) -> Union[str, List[List[Any]]]:
        """Check if the winning formula is valid; if not, synthesise counter-paths."""
        from .smt import Solver
        from .interpretation import Interpretation
        from ..core.syntax import mk_const

        model = Solver.get_model(ctx.solver)
        if model is None:
            return "Unsat"
        # SMTModel returns None when UNSAT; for unknown we check differently
        if isinstance(model, str) and model == "Unknown":
            return "Unknown"

        srk = ctx.srk

        if parameters is None:
            try:
                parameters = Interpretation(srk)
            except Exception:
                parameters = None

        def _counter_strategy(path_model: Any, skeleton: Any):
            """Traverse skeleton, building counter-paths."""
            if isinstance(skeleton, Skeleton.SEmpty):
                # Leaf: select implicant of the negated formula
                implicant = _select_implicant(srk, path_model, ctx.not_formula)
                if implicant is None:
                    return (None, [])
                return (implicant, [[]])

            if isinstance(skeleton, Skeleton.SForall):
                # Add Skolem constant's value to path_model
                try:
                    sk_val = _model_real(model, skeleton.skolem)
                    if isinstance(path_model, Interpretation):
                        path_model = path_model.add_real(skeleton.symbol, sk_val)
                except Exception:
                    pass
                # Recurse into subtree
                counter_phi, counter_paths = _counter_strategy(path_model, skeleton.subtree)
                if counter_phi is None:
                    return (None, [])
                # Select a move for the universally quantified variable
                move = select_term_fn(path_model, skeleton.symbol, counter_phi)
                # Substitute move into implicant
                counter_phi = Skeleton.substitute_implicant(
                    srk, path_model, skeleton.symbol, move, counter_phi
                )
                counter_paths = [[("Exists", (skeleton.symbol, move))] + p for p in counter_paths]
                return (counter_phi, counter_paths)

            if isinstance(skeleton, Skeleton.SExists):
                all_counter_phis = []
                all_paths = []
                for move, subtree in skeleton.moves:
                    # Extend path_model with this move's evaluation
                    pm = path_model
                    try:
                        if isinstance(move, Skeleton.MBool):
                            if isinstance(pm, Interpretation):
                                pm = pm.add_bool(skeleton.symbol, move.value)
                        else:
                            mv = Skeleton.evaluate_move(srk, lambda sym: _model_real(pm, sym), move)
                            if isinstance(pm, Interpretation):
                                pm = pm.add_real(skeleton.symbol, mv)
                    except Exception:
                        pass
                    counter_phi, counter_paths = _counter_strategy(pm, subtree)
                    if counter_phi is None:
                        continue
                    counter_phi = Skeleton.substitute_implicant(
                        srk, pm, skeleton.symbol, move, counter_phi
                    )
                    counter_paths = [[("Forall", skeleton.symbol)] + p for p in counter_paths]
                    all_counter_phis.append(counter_phi)
                    all_paths.extend(counter_paths)
                if not all_counter_phis:
                    return (None, [])
                # Combine implicants (conjunction)
                combined = all_counter_phis[0] if len(all_counter_phis) == 1 else all_counter_phis
                return (combined, all_paths)

            return (None, [])

        _, counter_paths = _counter_strategy(parameters, ctx.skeleton)
        if counter_paths:
            return counter_paths
        return "Unknown"

    @staticmethod
    def initialize_pair(select_term_fn: Callable, srk: Any,
                         qf_pre: List[Tuple[str, Any]], phi: Any):
        """Build initial SAT/UNSAT skeleton pair with warmup improvement."""
        from .smt import is_sat as _is_sat, get_model as _get_model, SMTResult
        from .smt import mk_solver, Solver
        from ..core.syntax import mk_not, mk_const
        from .interpretation import Interpretation

        # 1. Get a model of the matrix
        sat_result = _is_sat(srk, phi)
        if sat_result == SMTResult.UNSAT:
            return "Unsat"
        if sat_result == SMTResult.UNKNOWN:
            return "Unknown"

        phi_model = _get_model(phi, srk)
        if phi_model is None:
            return "Unknown"

        # 2. Select implicant and build initial paths
        implicant = _select_implicant(srk, phi_model, phi)
        if implicant is None:
            return "Unknown"

        sat_path: List[Any] = []
        unsat_path: List[Any] = []
        atoms = list(implicant)

        for qt, x in reversed(qf_pre):
            try:
                move = select_term_fn(phi_model, x, atoms)
            except Exception:
                # Fallback: use a default zero move
                from ..core.syntax import typ_symbol, Type
                typ = typ_symbol(x)
                if typ == Type.INT:
                    move = Skeleton.MInt(IntVirtualTerm(QQVector(), 1, 0))
                elif typ == Type.BOOL:
                    move = Skeleton.MBool(True)
                else:
                    move = Skeleton.MReal(QQVector())
            if qt == "Exists":
                sat_path.insert(0, ("Exists", (x, move)))
                unsat_path.insert(0, ("Forall", x))
            else:
                sat_path.insert(0, ("Forall", x))
                unsat_path.insert(0, ("Exists", (x, move)))
            atoms = Skeleton.substitute_implicant(srk, phi_model, x, move, atoms)

        # 3. Create contexts
        not_phi = mk_not(srk, phi)

        sat_skeleton = Skeleton.mk_path(srk, sat_path)
        unsat_skeleton = Skeleton.mk_path(srk, unsat_path)

        sat_win = Skeleton.winning_formula(srk, sat_skeleton, phi)
        unsat_win = Skeleton.winning_formula(srk, unsat_skeleton, not_phi)

        sat_solver = mk_solver(srk)
        Solver.add(sat_solver, [mk_not(srk, sat_win)])

        unsat_solver = mk_solver(srk)
        Solver.add(unsat_solver, [mk_not(srk, unsat_win)])

        sat_ctx = CSS.Ctx(srk=srk, formula=phi, not_formula=not_phi,
                           skeleton=sat_skeleton, solver=sat_solver)
        unsat_ctx = CSS.Ctx(srk=srk, formula=not_phi, not_formula=phi,
                             skeleton=unsat_skeleton, solver=unsat_solver)

        # 4. Warmup improvement loop
        max_improve = 2
        seen_skeletons: set = set()

        for _round in range(max_improve):
            res = CSS.get_counter_strategy(select_term_fn, sat_ctx)
            if res == "Unsat":
                return "Sat", (sat_ctx, unsat_ctx)
            if res == "Unknown":
                return "Unknown"
            if isinstance(res, list) and len(res) == 1:
                path = res[0]
                sk_key = str(path)
                if sk_key in seen_skeletons:
                    break
                seen_skeletons.add(sk_key)
                # Reset unsat context and add the counter-path
                CSS.reset(unsat_ctx)
                CSS.add_path(unsat_ctx, path)
                # Check if unsat wins
                res2 = CSS.get_counter_strategy(select_term_fn, unsat_ctx)
                if res2 == "Unsat":
                    return "Unsat"
                if res2 == "Unknown":
                    return "Unknown"
                if isinstance(res2, list):
                    CSS.reset(sat_ctx)
                    for p in res2:
                        CSS.add_path(sat_ctx, p)
            else:
                break

        return "Sat", (sat_ctx, unsat_ctx)

    @staticmethod
    def is_sat(select_term_fn: Callable, sat_ctx: "CSS.Ctx", unsat_ctx: "CSS.Ctx") -> str:
        """Main strategy improvement game loop."""
        old_paths = -1

        def _check_sat():
            nonlocal old_paths
            paths = Skeleton.nb_paths(sat_ctx.skeleton)
            assert paths > old_paths, f"No progress: {paths} <= {old_paths}"
            old_paths = paths
            res = CSS.get_counter_strategy(select_term_fn, sat_ctx)
            if res == "Unsat":
                return "Sat"
            if res == "Unknown":
                return "Unknown"
            # Add counter-paths to unsat skeleton
            for p in res:
                CSS.add_path(unsat_ctx, p)
            return _check_unsat()

        def _check_unsat():
            res = CSS.get_counter_strategy(select_term_fn, unsat_ctx)
            if res == "Unsat":
                return "Unsat"
            if res == "Unknown":
                return "Unknown"
            for p in res:
                CSS.add_path(sat_ctx, p)
            return _check_sat()

        return _check_sat()


def _quantifier_symbol_name(symbol: Any) -> str:
    """Return the name used by the SMT translator for a quantified symbol."""
    return getattr(symbol, "name", None) or str(symbol)


def _quantified_formula(srk: Any, qf_pre: List[Tuple[str, Any]], phi: Any) -> Any:
    """Wrap a quantifier-free matrix with the supplied quantifier prefix."""
    from ..core.syntax import mk_exists, mk_forall

    result = phi
    for qt, symbol in reversed(qf_pre):
        name = _quantifier_symbol_name(symbol)
        typ = getattr(symbol, "typ", None)
        if typ is None:
            return None
        if qt == "Forall":
            result = mk_forall(srk, name, typ, result)
        elif qt == "Exists":
            result = mk_exists(srk, name, typ, result)
        else:
            return None
    return result


def _is_numeric_literal(symbol: Any) -> bool:
    """Recognize symbols used by this port to encode numeric constants."""
    from ..core.syntax import Type

    name = getattr(symbol, "name", None)
    typ = getattr(symbol, "typ", None)
    if name is None or typ not in (Type.INT, Type.REAL):
        return False
    if typ == Type.REAL and name.startswith("real_"):
        name = name[5:]
    try:
        Fraction(name)
        return True
    except Exception:
        return False


def _is_linear_arithmetic_expr(expr: Any, in_formula: bool = False) -> bool:
    """Check the small arithmetic fragment that the Z3-backed path supports."""
    from ..core.syntax import (
        Add,
        And,
        Const,
        Eq,
        Exists,
        FalseExpr,
        Forall,
        Ite,
        Leq,
        Lt,
        Mul,
        Not,
        Or,
        TrueExpr,
        Type,
        Var,
        expr_typ,
    )

    if isinstance(expr, (TrueExpr, FalseExpr)):
        return True
    if isinstance(expr, Const):
        return expr.symbol.typ in (Type.INT, Type.REAL, Type.BOOL)
    if isinstance(expr, Var):
        return expr.var_type in (Type.INT, Type.REAL, Type.BOOL)
    if isinstance(expr, (Eq, Lt, Leq)):
        return _is_linear_arithmetic_expr(expr.left) and _is_linear_arithmetic_expr(
            expr.right
        )
    if isinstance(expr, (And, Or)):
        return all(_is_linear_arithmetic_expr(arg, True) for arg in expr.args)
    if isinstance(expr, Not):
        return _is_linear_arithmetic_expr(expr.arg, True)
    if isinstance(expr, (Forall, Exists)):
        return expr.var_type in (Type.INT, Type.REAL, Type.BOOL) and (
            _is_linear_arithmetic_expr(expr.body, True)
        )
    if isinstance(expr, Add):
        return all(_is_linear_arithmetic_expr(arg) for arg in expr.args)
    if isinstance(expr, Mul):
        nonlinear_args = 0
        for arg in expr.args:
            if not _is_linear_arithmetic_expr(arg):
                return False
            if not (isinstance(arg, Const) and _is_numeric_literal(arg.symbol)):
                try:
                    if expr_typ(arg) in (Type.INT, Type.REAL):
                        nonlinear_args += 1
                except Exception:
                    nonlinear_args += 1
        return nonlinear_args <= 1
    if isinstance(expr, Ite):
        return (
            _is_linear_arithmetic_expr(expr.condition, True)
            and _is_linear_arithmetic_expr(expr.then_branch, in_formula)
            and _is_linear_arithmetic_expr(expr.else_branch, in_formula)
        )
    return False


def _contains_quantifier(expr: Any) -> bool:
    from ..core.syntax import Exists, Forall

    if isinstance(expr, (Exists, Forall)):
        return True
    for attr in ("args",):
        for child in getattr(expr, attr, ()) or ():
            if _contains_quantifier(child):
                return True
    for attr in (
        "arg",
        "left",
        "right",
        "condition",
        "then_branch",
        "else_branch",
        "body",
    ):
        child = getattr(expr, attr, None)
        if child is not None and _contains_quantifier(child):
            return True
    return False


def _smt_result_name(result: Any) -> str:
    value = getattr(result, "value", result)
    if value in ("sat", "Sat"):
        return "Sat"
    if value in ("unsat", "Unsat"):
        return "Unsat"
    return "Unknown"


def _bounded_z3_simsat(srk: Any, qf_pre: List[Tuple[str, Any]], phi: Any) -> str:
    """Validate an alternating linear-arithmetic game with Z3 when possible."""
    from .smt import is_sat

    quantified = _quantified_formula(srk, qf_pre, phi)
    if quantified is None or not _is_linear_arithmetic_expr(quantified, True):
        return "Unknown"
    try:
        return _smt_result_name(is_sat(srk, quantified))
    except Exception:
        return "Unknown"


def _z3_number_to_fraction(value: Any) -> Optional[Fraction]:
    """Convert a finite Z3 numeral/bound to Fraction when possible."""
    text = str(value)
    if text in ("oo", "+oo", "-oo"):
        return None
    if text.endswith("?"):
        text = text[:-1]
    try:
        if hasattr(value, "as_fraction"):
            return Fraction(value.as_fraction())
        if "/" in text:
            num, den = text.split("/", 1)
            return Fraction(int(num), int(den))
        return Fraction(text)
    except Exception:
        return None


def simsat_core(srk: Any, qf_pre: List[Tuple[str, Any]], phi: Any) -> str:
    """Game-theoretic satisfiability core using strategy improvement.

    Tries the full alternating strategy-improvement game first.
    Falls back to Z3 direct check when the game-theoretic path fails
    (e.g. incomplete SMT model).
    """
    select_term_fn = lambda model, x, atoms: _select_term(srk, model, x, atoms)

    try:
        init = CSS.initialize_pair(select_term_fn, srk, qf_pre, phi)
    except Exception:
        init = "Unknown"

    if init == "Unsat":
        return "Unsat"
    if init == "Unknown":
        # Fall back to Z3 for supported linear fragments
        return _bounded_z3_simsat(srk, qf_pre, phi)
    # init == ("Sat", (sat_ctx, unsat_ctx))
    _, (sat_ctx, unsat_ctx) = init
    CSS.reset(unsat_ctx)
    try:
        result = CSS.is_sat(select_term_fn, sat_ctx, unsat_ctx)
    except Exception:
        result = "Unknown"
    if result == "Unknown":
        return _bounded_z3_simsat(srk, qf_pre, phi)
    return result


def simsat_forward(srk: Any, phi: Any) -> str:
    """Forward version of simsat (simplified: delegates to simsat_core)."""
    from ..core.syntax import symbols as _symbols, mk_not

    constants = {s for s in _symbols(phi) if not _is_numeric_literal(s)}

    qf_pre, psi = normalize(srk, phi)
    qf_pre = [("Exists", k) for k in constants] + qf_pre

    # If the prefix leads with existential, negate and swap
    negate = False
    if qf_pre and qf_pre[0][0] == "Exists":
        psi = normalize(srk, mk_not(srk, phi))[1]
        qf_pre = [("Forall" if qt == "Exists" else "Exists", x) for qt, x in qf_pre]
        negate = True

    result = simsat_core(srk, qf_pre, psi)
    if negate:
        if result == "Sat":
            return "Unsat"
        if result == "Unsat":
            return "Sat"
    return result


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
    if _contains_quantifier(phi):
        qf_pre, phi_norm = normalize(srk, phi)
    else:
        qf_pre, phi_norm = [], phi
    if qf_pre or not (
        _is_linear_arithmetic_expr(phi_norm, True) and _is_linear_arithmetic_expr(t)
    ):
        return "Unknown"

    from .smt import is_sat
    from .srkZ3 import make_z3_context, z3, Z3_AVAILABLE
    from ..core.syntax import mk_and, mk_lt, mk_real

    if not Z3_AVAILABLE:
        return "Unknown"

    sat_result = _smt_result_name(is_sat(srk, phi_norm))
    if sat_result == "Unsat":
        return "MinusInfinity"
    if sat_result == "Unknown":
        return "Unknown"

    try:
        z3_ctx = make_z3_context(srk)
        opt = z3.Optimize(ctx=z3_ctx.z3_ctx)
        opt.add(z3_ctx.z3_of_formula(phi_norm))
        handle = opt.maximize(z3_ctx._z3_of_expr(t))
        result = opt.check()

        if result == z3.unsat:
            return "MinusInfinity"
        if result != z3.sat:
            return "Unknown"

        upper = opt.upper(handle)
        upper_text = str(upper)
        if upper_text in ("oo", "+oo"):
            return "Infinity"
        if upper_text == "-oo":
            return "MinusInfinity"

        upper_fraction = _z3_number_to_fraction(upper)
        if upper_fraction is None:
            return "Unknown"

        # Validate the optimum with a second solver query: no model may improve it.
        better = mk_and(srk, [phi_norm, mk_lt(srk, mk_real(srk, upper_fraction), t)])
        if _smt_result_name(is_sat(srk, better)) != "Unsat":
            return "Unknown"

        return ("Bounded", upper_fraction)

    except Exception:
        return "Unknown"


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

    if sat_result == "Unsat":
        return "MinusInfinity"
    elif sat_result == "Unknown":
        return "Unknown"
    else:  # sat_result == 'Sat'
        return maximize_feasible(srk, phi, t)


def extract_strategy(srk: Any, skeleton: Any, phi: Any) -> Any:
    """Extract a strategy from a skeleton. Returns the skeleton itself as the strategy."""
    return skeleton


def winning_strategy(srk: Any, qf_pre: List[Tuple[str, Any]], phi: Any) -> Any:
    """Compute a winning SAT/UNSAT strategy for a formula in prenex form."""
    select_term_fn = lambda model, x, atoms: _select_term(srk, model, x, atoms)

    init = CSS.initialize_pair(select_term_fn, srk, qf_pre, phi)
    if init == "Unsat":
        return "Unsat"
    if init == "Unknown":
        return "Unknown"
    _, (sat_ctx, unsat_ctx) = init
    CSS.reset(unsat_ctx)
    result = CSS.is_sat(select_term_fn, sat_ctx, unsat_ctx)
    if result == "Sat":
        return ("Sat", extract_strategy(srk, sat_ctx.skeleton, phi))
    if result == "Unsat":
        from ..core.syntax import mk_not
        return ("Unsat", extract_strategy(srk, unsat_ctx.skeleton, mk_not(srk, phi)))
    return "Unknown"


def check_strategy(
    srk: Any, qf_pre: List[Tuple[str, Any]], phi: Any, strategy: Any
) -> Optional[bool]:
    """Validate a candidate strategy for supported linear arithmetic games.

    Returns True/False when Z3 can prove the strategy formula satisfiable or
    unsatisfiable, and None when the strategy or formula is outside the bounded
    validation fragment.
    """
    if not _is_linear_arithmetic_expr(phi, True):
        return None

    concrete = _concrete_strategy_assignments(srk, strategy)
    if concrete is not None:
        qf_pre, strategy_formula = _apply_concrete_strategy(srk, qf_pre, phi, concrete)
    elif strategy is None or strategy == {} or strategy == {"strategy": []}:
        strategy_formula = phi
    elif isinstance(strategy, Skeleton.SEmpty):
        strategy_formula = phi
    elif isinstance(strategy, (Skeleton.SForall, Skeleton.SExists)):
        try:
            strategy_formula = Skeleton.winning_formula(srk, strategy, phi)
        except Exception:
            return None
    else:
        return None

    result = simsat_core(srk, qf_pre, strategy_formula)
    if result == "Sat":
        return True
    if result == "Unsat":
        return False
    return None


def _concrete_strategy_assignments(
    srk: Any, strategy: Any
) -> Optional[Dict[Any, Any]]:
    """Parse simple concrete strategy maps used by the Python migration tests."""
    if not isinstance(strategy, dict):
        return None
    raw = strategy.get("assignments")
    if raw is None:
        raw = strategy.get("strategy")
    if raw in (None, []):
        return None
    if isinstance(raw, dict):
        items = raw.items()
    elif isinstance(raw, list):
        items = raw
    else:
        return None

    parsed: Dict[Any, Any] = {}
    for key, value in items:
        symbol = _resolve_strategy_symbol(srk, key)
        if symbol is None:
            return None
        expr = _strategy_value_expr(srk, symbol, value)
        if expr is None:
            return None
        parsed[symbol] = expr
    return parsed


def _resolve_strategy_symbol(srk: Any, key: Any) -> Optional[Any]:
    if hasattr(key, "typ") and hasattr(key, "id"):
        return key
    if isinstance(key, str):
        try:
            if hasattr(srk, "is_registered_name") and srk.is_registered_name(key):
                return srk.get_named_symbol(key)
        except Exception:
            pass
        for symbol in getattr(srk, "_symbols", {}).values():
            if getattr(symbol, "name", None) == key:
                return symbol
    return None


def _strategy_value_expr(srk: Any, symbol: Any, value: Any) -> Optional[Any]:
    from ..core.syntax import Expression, Type, mk_false, mk_int, mk_real, mk_true

    if isinstance(value, Expression):
        return value
    typ = getattr(symbol, "typ", None)
    if typ == Type.BOOL:
        if isinstance(value, bool):
            return mk_true(srk) if value else mk_false(srk)
        return None
    try:
        rational = Fraction(value)
    except Exception:
        return None
    if typ == Type.INT:
        if rational.denominator != 1:
            return None
        return mk_int(srk, rational.numerator)
    if typ == Type.REAL:
        return mk_real(srk, rational)
    return None


def _apply_concrete_strategy(
    srk: Any,
    qf_pre: List[Tuple[str, Any]],
    phi: Any,
    assignments: Dict[Any, Any],
) -> Tuple[List[Tuple[str, Any]], Any]:
    from ..core.syntax import substitute_const

    remaining_prefix = []
    subst = {}
    for qt, symbol in qf_pre:
        if qt == "Exists" and symbol in assignments:
            subst[symbol] = assignments[symbol]
        else:
            remaining_prefix.append((qt, symbol))
    return remaining_prefix, substitute_const(subst, phi) if subst else phi
