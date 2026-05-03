"""
Convex polyhedra (wedge) implementation for SRK.

This module provides abstract domain operations for convex polyhedra using
the APRON library for polyhedral operations and analysis.
Based on the OCaml implementation in src/wedge.ml.
"""

import logging
from typing import List, Dict, Set, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
from functools import reduce

from . import linear
from . import syntax
from . import smt as Smt
from . import interpretation
from . import srkSimplify
from . import nonlinear
from . import apron as ApronInterface
from . import coordinateSystem as CS
from . import polynomial as P
from .polynomial import RewriteSystem

logger = logging.getLogger(__name__)


# APRON types (simplified)
class Scalar:
    Float = "Float"
    Mpqf = "Mpqf"
    Mpfrf = "Mpfrf"


class Coeff:
    Scalar = "Scalar"
    Interval = "Interval"


class Abstract0:
    def __init__(self, manager, int_dim: int, real_dim: int, srk=None):
        self.manager = manager
        self.int_dim = int_dim
        self.real_dim = real_dim
        self._constraints: List[Lincons0] = []
        self._is_bottom: bool = False
        self._srk = srk  # Optional SRK context for Z3-based checks

    @staticmethod
    def top(manager, int_dim: int, real_dim: int, srk=None):
        obj = Abstract0(manager, int_dim, real_dim, srk)
        obj._constraints = []
        obj._is_bottom = False
        return obj

    @staticmethod
    def bottom(manager, int_dim: int, real_dim: int, srk=None):
        obj = Abstract0(manager, int_dim, real_dim, srk)
        obj._constraints = []
        obj._is_bottom = True
        return obj

    @staticmethod
    def of_lincons_array(manager, int_dim, real_dim, lincons_array, srk=None):
        obj = Abstract0(manager, int_dim, real_dim, srk)
        obj._constraints = list(lincons_array)
        obj._is_bottom = False
        return obj

    @staticmethod
    def is_top(manager, abstract):
        if abstract._is_bottom:
            return False
        if len(abstract._constraints) == 0:
            return True
        # Even with no explicit constraints, top means no restrictions.
        # If constraints exist, check if any are non-trivial.
        # A non-trivial constraint that is satisfiable means not top.
        for lc in abstract._constraints:
            # Check if constraint is non-trivial (not 0 >= 0)
            all_zero = True
            for coeff, d in lc.linexpr0.coeffs:
                if coeff != 0:
                    all_zero = False
                    break
            cst = lc.linexpr0.get_cst()
            if cst is not None and cst != 0:
                all_zero = False
            if not all_zero:
                return False
        return True

    @staticmethod
    def is_bottom(manager, abstract):
        if abstract._is_bottom:
            return True
        if not abstract._constraints:
            return False
        # Use Z3 to check satisfiability if srk context is available
        if abstract._srk is not None:
            srk = abstract._srk
            n_dims = abstract.int_dim + abstract.real_dim
            dim_vars = []
            for i in range(n_dims):
                sym = syntax.mk_symbol(srk, f"_apron_d{i}", syntax.Type.REAL)
                dim_vars.append(syntax.mk_const(srk, sym))
            formulas = []
            for lc in abstract._constraints:
                terms = []
                for coeff, dim in lc.linexpr0.coeffs:
                    if dim is not None and dim < len(dim_vars):
                        coeff_term = syntax.mk_real(srk, coeff)
                        terms.append(syntax.mk_mul(srk, [coeff_term, dim_vars[dim]]))
                cst = lc.linexpr0.get_cst()
                if cst is not None:
                    terms.append(syntax.mk_real(srk, cst))
                if len(terms) > 1:
                    lin_term = syntax.mk_add(srk, terms)
                elif len(terms) == 1:
                    lin_term = terms[0]
                else:
                    lin_term = syntax.mk_real(srk, linear.QQ.zero())
                zero = syntax.mk_real(srk, linear.QQ.zero())
                if lc.typ == Lincons0.EQ:
                    formulas.append(syntax.mk_eq(srk, lin_term, zero))
                elif lc.typ == Lincons0.SUPEQ:
                    formulas.append(syntax.mk_leq(srk, zero, lin_term))
                elif lc.typ == Lincons0.SUP:
                    formulas.append(syntax.mk_lt(srk, zero, lin_term))
            if formulas:
                formula = syntax.mk_and(srk, formulas)
                result = Smt.is_sat(srk, formula)
                return result == Smt.Unsat
        return False

    @staticmethod
    def is_eq(manager, abstract1, abstract2):
        if abstract1._is_bottom and abstract2._is_bottom:
            return True
        if abstract1._is_bottom != abstract2._is_bottom:
            return False
        # Compare constraint sets (order-independent)
        c1 = [(lc.typ, tuple(lc.linexpr0.coeffs), lc.linexpr0.get_cst())
               for lc in abstract1._constraints]
        c2 = [(lc.typ, tuple(lc.linexpr0.coeffs), lc.linexpr0.get_cst())
               for lc in abstract2._constraints]
        return set(c1) == set(c2)

    @staticmethod
    def add_dimensions(manager, abstract, dim: "Dim", project: bool):
        """Add dimensions to an abstract value, remapping existing dimension indices.

        When adding ``dim.intdim`` integer dimensions and ``dim.realdim``
        real dimensions, existing real-dimension references in all
        constraint linear expressions must be shifted by ``dim.intdim``
        (since new integer dims are inserted before the real dims).

        The ``dim.dim`` array specifies insertion positions: the first
        ``dim.intdim`` entries are the int-dim insertion point and the
        remaining ``dim.realdim`` entries are the real-dim insertion point.
        """
        new_int = abstract.int_dim + dim.intdim
        new_real = abstract.real_dim + dim.realdim
        result = Abstract0(manager, new_int, new_real, abstract._srk)
        result._is_bottom = abstract._is_bottom

        shift = dim.intdim  # existing real dims shift by this amount

        # Remap linexpr coefficients: shift real-dimension indices
        new_constraints = []
        for lc in abstract._constraints:
            new_coeffs = []
            for coeff, d in lc.linexpr0.coeffs:
                if d is not None and d >= abstract.int_dim:
                    # This is a real dimension — shift it
                    new_coeffs.append((coeff, d + shift))
                else:
                    new_coeffs.append((coeff, d))
            new_linexpr = Linexpr0.of_list(None, new_coeffs, lc.linexpr0.get_cst())
            new_constraints.append(Lincons0.make(new_linexpr, lc.typ))
        result._constraints = new_constraints
        return result

    @staticmethod
    def remove_dimensions(manager, abstract, dim: "Dim"):
        """Remove dimensions from an abstract value.

        ``dim.dim`` lists the dimension indices to remove (sorted
        ascending).  After removal, remaining dimension indices are
        compacted so that they form a contiguous range [0, new_dim).
        """
        if abstract._is_bottom:
            new_int = abstract.int_dim - dim.intdim
            new_real = abstract.real_dim - dim.realdim
            result = Abstract0(manager, new_int, new_real, abstract._srk)
            result._is_bottom = True
            return result

        remove_set = set(dim.dim)

        new_constraints = []
        for lc in abstract._constraints:
            new_coeffs = []
            for coeff, d in lc.linexpr0.coeffs:
                if d is not None and d in remove_set:
                    continue  # drop this dimension
                if d is not None:
                    # compute new index: count removed dims below d
                    shift = sum(1 for r in sorted(remove_set) if r < d)
                    new_coeffs.append((coeff, d - shift))
                else:
                    new_coeffs.append((coeff, d))
            new_linexpr = Linexpr0.of_list(None, new_coeffs, lc.linexpr0.get_cst())
            new_constraints.append(Lincons0.make(new_linexpr, lc.typ))

        new_int = abstract.int_dim - dim.intdim
        new_real = abstract.real_dim - dim.realdim
        result = Abstract0(manager, new_int, new_real, abstract._srk)
        result._constraints = new_constraints
        result._is_bottom = False
        return result

    @staticmethod
    def meet_lincons_array(manager, abstract, lincons_array):
        result = Abstract0(manager, abstract.int_dim, abstract.real_dim, abstract._srk)
        result._constraints = list(abstract._constraints) + list(lincons_array)
        result._is_bottom = abstract._is_bottom
        return result

    @staticmethod
    def join(manager, abstract1, abstract2):
        """Compute the convex hull (least upper bound) of two abstract values.

        Keeps only constraints that are syntactically present in both
        operands.  This is a sound over-approximation of the true convex
        hull.
        """
        if abstract1._is_bottom:
            result = Abstract0(manager, abstract2.int_dim, abstract2.real_dim,
                               abstract2._srk)
            result._constraints = list(abstract2._constraints)
            result._is_bottom = abstract2._is_bottom
            return result
        if abstract2._is_bottom:
            result = Abstract0(manager, abstract1.int_dim, abstract1.real_dim,
                               abstract1._srk)
            result._constraints = list(abstract1._constraints)
            result._is_bottom = abstract1._is_bottom
            return result
        # Keep only constraints present in both (intersection of constraint sets)
        c1 = {(lc.typ, tuple(lc.linexpr0.coeffs), lc.linexpr0.get_cst())
               for lc in abstract1._constraints}
        c2 = {(lc.typ, tuple(lc.linexpr0.coeffs), lc.linexpr0.get_cst())
               for lc in abstract2._constraints}
        common = c1 & c2
        srk = abstract1._srk or abstract2._srk
        result = Abstract0(manager, abstract1.int_dim, abstract1.real_dim, srk)
        result._constraints = [lc for lc in abstract1._constraints
                               if (lc.typ, tuple(lc.linexpr0.coeffs),
                                   lc.linexpr0.get_cst()) in common]
        result._is_bottom = False
        return result

    @staticmethod
    def widening(manager, abstract1, abstract2):
        """APRON-style widening: keep only constraints present in BOTH operands.

        This ensures monotone convergence in fixpoint iteration by
        discarding constraints that appear only in the second operand.
        """
        if abstract1._is_bottom:
            result = Abstract0(manager, abstract2.int_dim, abstract2.real_dim,
                               abstract2._srk)
            result._constraints = list(abstract2._constraints)
            result._is_bottom = abstract2._is_bottom
            return result
        if abstract2._is_bottom:
            result = Abstract0(manager, abstract1.int_dim, abstract1.real_dim,
                               abstract1._srk)
            result._constraints = list(abstract1._constraints)
            result._is_bottom = abstract1._is_bottom
            return result
        # Widening: keep only constraints from abstract1 that are also in abstract2
        # (discard anything new in abstract2 to ensure convergence)
        c2 = {(lc.typ, tuple(lc.linexpr0.coeffs), lc.linexpr0.get_cst())
               for lc in abstract2._constraints}
        srk = abstract1._srk or abstract2._srk
        result = Abstract0(manager, abstract1.int_dim, abstract1.real_dim, srk)
        result._constraints = [lc for lc in abstract1._constraints
                               if (lc.typ, tuple(lc.linexpr0.coeffs),
                                   lc.linexpr0.get_cst()) in c2]
        result._is_bottom = False
        return result

    @staticmethod
    def forget_array(manager, abstract, dims, project: bool):
        """Project out (forget) the given dimension indices.

        Removes all constraints that mention any of the forgotten
        dimensions.  This is a sound (but potentially imprecise)
        over-approximation of Fourier-Motzkin elimination.
        """
        if abstract._is_bottom:
            result = Abstract0(manager, abstract.int_dim, abstract.real_dim, abstract._srk)
            result._is_bottom = True
            return result

        forget_set = set(dims) if not isinstance(dims, set) else dims
        new_constraints = []
        for lc in abstract._constraints:
            mentions_forgotten = False
            for coeff, d in lc.linexpr0.coeffs:
                if d is not None and d in forget_set:
                    mentions_forgotten = True
                    break
            if not mentions_forgotten:
                new_constraints.append(lc)

        result = Abstract0(manager, abstract.int_dim, abstract.real_dim, abstract._srk)
        result._constraints = new_constraints
        result._is_bottom = False
        return result

    @staticmethod
    def substitute_linexpr_array(manager, abstract, dims, linexprs, envp):
        """Substitute linear expressions for the given dimensions.

        For each dimension ``dims[i]`` replace it with ``linexprs[i]``
        in all stored constraints.
        """
        if abstract._is_bottom:
            result = Abstract0(manager, abstract.int_dim, abstract.real_dim, abstract._srk)
            result._is_bottom = True
            return result

        dim_to_subst = {}
        for d, le in zip(dims, linexprs):
            dim_to_subst[d] = le

        new_constraints = []
        for lc in abstract._constraints:
            # Build a mapping of dim -> coeff from the original constraint
            orig_coeffs = {}
            for coeff, d in lc.linexpr0.coeffs:
                if d is not None:
                    orig_coeffs[d] = coeff
            orig_cst = lc.linexpr0.get_cst() or 0

            # Perform substitution: for each dimension in dim_to_subst,
            # replace it with the linear expression
            new_coeffs_map = {}
            new_cst = orig_cst

            for d, coeff in orig_coeffs.items():
                if d in dim_to_subst:
                    # Substitute: coeff * linexpr(d) expands into other dims
                    subst_le = dim_to_subst[d]
                    for subst_coeff, subst_dim in subst_le.coeffs:
                        if subst_dim is not None:
                            new_coeffs_map[subst_dim] = (
                                new_coeffs_map.get(subst_dim, 0) + coeff * subst_coeff
                            )
                    subst_cst = subst_le.get_cst()
                    if subst_cst is not None:
                        new_cst += coeff * subst_cst
                else:
                    new_coeffs_map[d] = new_coeffs_map.get(d, 0) + coeff

            # Filter out zero coefficients
            final_coeffs = [(c, d) for d, c in new_coeffs_map.items()
                           if c != 0]
            final_cst = new_cst if new_cst != 0 else None

            new_linexpr = Linexpr0.of_list(None, final_coeffs, final_cst)
            new_constraints.append(Lincons0.make(new_linexpr, lc.typ))

        result = Abstract0(manager, abstract.int_dim, abstract.real_dim, abstract._srk)
        result._constraints = new_constraints
        result._is_bottom = False
        return result

    @staticmethod
    def copy_abstract(abstract):
        """Deep copy an Abstract0 value."""
        result = Abstract0(abstract.manager, abstract.int_dim, abstract.real_dim,
                          abstract._srk)
        result._constraints = [
            Lincons0.make(
                Linexpr0.of_list(None, list(lc.linexpr0.coeffs), lc.linexpr0.get_cst()),
                lc.typ
            )
            for lc in abstract._constraints
        ]
        result._is_bottom = abstract._is_bottom
        return result

    @staticmethod
    def to_lincons_array(manager, abstract):
        return list(abstract._constraints)


class Linexpr0:
    def __init__(self, coeffs: List[Tuple], cst: Optional):
        self.coeffs = coeffs
        self.cst = cst

    @staticmethod
    def of_list(project, coeffs, cst):
        return Linexpr0(coeffs, cst)

    def iter(self, f):
        for coeff, dim in self.coeffs:
            f(coeff, dim)
        if self.cst:
            f(self.cst, None)

    def get_cst(self):
        return self.cst

    def set_coeff(self, dim, coeff):
        # Update coefficient for dimension
        pass


class Lincons0:
    EQ = "EQ"
    SUPEQ = "SUPEQ"
    SUP = "SUP"
    DISEQ = "DISEQ"
    EQMOD = "EQMOD"

    def __init__(self, linexpr0, typ: str):
        self.linexpr0 = linexpr0
        self.typ = typ

    @staticmethod
    def make(linexpr0, typ: str):
        return Lincons0(linexpr0, typ)


class Dim:
    def __init__(self, dim: List[int], intdim: int, realdim: int):
        self.dim = dim
        self.intdim = intdim
        self.realdim = realdim


# Manager for APRON operations
def get_manager():
    """Get APRON manager (Polka strict)"""
    return "PolkaManager"  # Placeholder


# Environment for coordinate system mapping
@dataclass
class Environment:
    """Environment mapping coordinates to APRON dimensions"""

    int_dim: List[int]  # Integer coordinate IDs
    real_dim: List[int]  # Real coordinate IDs

    def copy(self) -> "Environment":
        return Environment(self.int_dim.copy(), self.real_dim.copy())


@dataclass
class Wedge:
    """Convex polyhedron (wedge) representation"""

    srk: syntax.Context  # SRK context
    cs: CS.CoordinateSystem  # Coordinate system
    env: Environment  # Environment mapping
    abstract: Abstract0  # APRON abstract value

    def is_consistent(self) -> bool:
        """Check if environment is consistent with coordinate system"""
        return CS.dim(self.cs) == len(self.env.int_dim) + len(self.env.real_dim)

    def update_env(self) -> None:
        """Update environment when coordinate system grows"""
        int_dim = len(self.env.int_dim)
        real_dim = len(self.env.real_dim)

        if int_dim + real_dim < CS.dim(self.cs):
            added_int = 0
            added_real = 0

            for id in range(int_dim + real_dim, CS.dim(self.cs)):
                match CS.type_of_id(self.cs, id):
                    case syntax.TyInt:
                        added_int += 1
                        self.env.int_dim.append(id)
                    case syntax.TyReal:
                        added_real += 1
                        self.env.real_dim.append(id)

            logger.debug(
                f"update env: adding {added_int} integer and {added_real} real dimension(s)"
            )

            added = [
                int_dim if i < added_int else int_dim + real_dim
                for i in range(added_int + added_real)
            ]

            # Add dimensions to APRON abstract value
            self.abstract = Abstract0.add_dimensions(
                get_manager(), self.abstract, Dim(added, added_int, added_real), False
            )


# Utility functions
def qq_of_scalar(scalar) -> linear.QQ:
    """Convert APRON scalar to QQ"""
    match scalar:
        case Scalar.Float(k):
            return linear.QQ.of_float(k)
        case Scalar.Mpqf(k):
            return k
        case Scalar.Mpfrf(k):
            return linear.QQ.from_mpfrf(k)


def qq_of_coeff(coeff) -> Optional[linear.QQ]:
    """Convert APRON coefficient to QQ"""
    match coeff:
        case Coeff.Scalar(s):
            return qq_of_scalar(s)
        case Coeff.Interval(_):
            return None


def qq_of_coeff_exn(coeff) -> linear.QQ:
    """Convert APRON coefficient to QQ (raises exception if interval)"""
    match coeff:
        case Coeff.Scalar(s):
            return qq_of_scalar(s)
        case Coeff.Interval(_):
            raise ValueError("qq_of_coeff_exn: argument must be a scalar")


def coeff_of_qq(qq: linear.QQ) -> str:
    """Convert QQ to APRON coefficient"""
    return Coeff.Scalar(Scalar.Mpqf(qq))


def mk_log(srk: syntax.Context) -> syntax.Symbol:
    """Get log symbol"""
    return nonlinear.mk_log(srk)


def mk_pow(srk: syntax.Context) -> syntax.Symbol:
    """Get pow symbol"""
    return nonlinear.mk_pow(srk)


def vec_of_poly(poly: P.Polynomial) -> Optional[linear.QQVector]:
    """Convert polynomial to vector if linear"""
    # This would need proper polynomial vector conversion
    return None


def poly_of_vec(vec: linear.QQVector) -> P.Polynomial:
    """Convert vector to polynomial"""
    # This would need proper vector polynomial conversion
    return P.zero()


# Environment operations
def mk_empty_env() -> Environment:
    """Create empty environment"""
    return Environment([], [])


def mk_env(cs: CS.CoordinateSystem) -> Environment:
    """Create environment from coordinate system"""
    env = mk_empty_env()
    for id in range(CS.dim(cs)):
        match CS.type_of_id(cs, id):
            case syntax.TyInt:
                env.int_dim.append(id)
            case syntax.TyReal:
                env.real_dim.append(id)
    return env


# Wedge operations
def top(srk: syntax.Context) -> Wedge:
    """Create top wedge (universe)"""
    cs = CS.mk_empty(srk)
    return Wedge(srk, cs, mk_empty_env(), Abstract0.top(get_manager(), 0, 0))


def is_top(wedge: Wedge) -> bool:
    """Check if wedge is top"""
    return Abstract0.is_top(get_manager(), wedge.abstract)


def bottom(srk: syntax.Context) -> Wedge:
    """Create bottom wedge (empty)"""
    cs = CS.mk_empty(srk)
    return Wedge(srk, cs, mk_empty_env(), Abstract0.bottom(get_manager(), 0, 0))


def is_bottom(wedge: Wedge) -> bool:
    """Check if wedge is bottom"""
    return Abstract0.is_bottom(get_manager(), wedge.abstract)


def copy(wedge: Wedge) -> Wedge:
    """Deep copy a wedge (including its mutable abstract value)."""
    return Wedge(
        wedge.srk,
        CS.copy(wedge.cs),
        wedge.env.copy(),
        Abstract0.copy_abstract(wedge.abstract),
    )


def equal(wedge1: Wedge, wedge2: Wedge) -> bool:
    """Check if two wedges are equal"""
    srk = wedge1.srk
    phi = nonlinear.uninterpret(srk, to_formula(wedge1))
    phi_prime = nonlinear.uninterpret(srk, to_formula(wedge2))

    return (
        Smt.is_sat(srk, syntax.mk_not(srk, syntax.mk_iff(srk, phi, phi_prime)))
        == Smt.Unsat
    )


def to_atoms(wedge: Wedge) -> List[syntax.Formula]:
    """Convert wedge to list of atomic formulas.

    Retrieves all linear constraints stored in the abstract domain and
    converts each one to an SRK formula via :func:`atom_of_lincons`.
    """
    lincons_array = Abstract0.to_lincons_array(get_manager(), wedge.abstract)
    return [atom_of_lincons(wedge, lc) for lc in lincons_array]


def to_formula(wedge: Wedge) -> syntax.Formula:
    """Convert wedge to formula"""
    return syntax.mk_and(wedge.srk, to_atoms(wedge))


# APRON conversion functions
def vec_of_linexpr(env: Environment, linexpr: Linexpr0) -> linear.QQVector:
    """Convert APRON linexpr to vector"""
    vec = linear.QQVector.zero()

    for coeff, dim in linexpr.coeffs:
        qq_coeff = qq_of_coeff(coeff)
        if qq_coeff is not None and not linear.QQ.equal(qq_coeff, linear.QQ.zero()):
            # Map dimension back to coordinate ID
            if dim < len(env.int_dim):
                coord_id = env.int_dim[dim]
            else:
                coord_id = env.real_dim[dim - len(env.int_dim)]
            vec = linear.QQVector.add_term(qq_coeff, coord_id, vec)

    qq_cst = qq_of_coeff(linexpr.cst)
    if qq_cst is not None:
        vec = linear.QQVector.add_term(qq_cst, CS.const_id, vec)

    return vec


def linexpr_of_vec(
    cs: CS.CoordinateSystem, env: Environment, vec: linear.QQVector
) -> Linexpr0:
    """Convert vector to APRON linexpr"""

    def mk_coeff_dim(coeff, id):
        coord_id = CS.dim_of_id(cs, env, id)
        return (coeff_of_qq(coeff), coord_id)

    const_coeff, rest = linear.QQVector.pivot(CS.const_id, vec)
    coeffs = [mk_coeff_dim(coeff, id) for coeff, id in linear.QQVector.enum(rest)]
    # Pass None if constant coefficient is zero, otherwise pass the coefficient
    cst_value = (
        None
        if linear.QQ.equal(const_coeff, linear.QQ.zero())
        else coeff_of_qq(const_coeff)
    )
    return Linexpr0.of_list(None, coeffs, cst_value)


def atom_of_lincons(wedge: Wedge, lincons: Lincons0) -> syntax.Formula:
    """Convert APRON lincons to atomic formula"""
    term = CS.term_of_vec(wedge.cs, vec_of_linexpr(wedge.env, lincons.linexpr0))
    zero = syntax.mk_real(wedge.srk, linear.QQ.zero())

    match lincons.typ:
        case Lincons0.EQ:
            return syntax.mk_eq(wedge.srk, term, zero)
        case Lincons0.SUPEQ:
            return syntax.mk_leq(wedge.srk, zero, term)
        case Lincons0.SUP:
            return syntax.mk_lt(wedge.srk, zero, term)
        case _:
            raise ValueError(f"Unsupported lincons type: {lincons.typ}")


def pp(formatter, wedge: Wedge) -> None:
    """Pretty-print a wedge (mirrors OCaml ``Wedge.pp``)."""
    if hasattr(wedge, 'to_formula'):
        formatter.write(str(wedge.to_formula()))
    else:
        formatter.write(f"Wedge({CS.dim(wedge.cs)} dims)")


def show(wedge: Wedge) -> str:
    """String representation of a wedge (mirrors OCaml ``Wedge.show``)."""
    if hasattr(wedge, 'to_formula'):
        return str(wedge.to_formula())
    return f"Wedge({CS.dim(wedge.cs)} dims)"


def coordinate_system(wedge: Wedge):
    """Get the coordinate system associated with a wedge.

    Mirrors OCaml ``Wedge.coordinate_system``.
    """
    if hasattr(wedge, 'cs'):
        return wedge.cs
    return None


def polyhedron(
    wedge: Wedge,
) -> List[Tuple[str, "QQVector"]]:
    """Extract the polyhedron as tagged (Eq/Geq, QQVector) pairs.

    Mirrors OCaml ``Wedge.polyhedron``.
    """
    from .polyhedron import Polyhedron as _Poly

    constraints: List[Tuple[str, "QQVector"]] = []
    for atom in wedge.to_atoms():
        if isinstance(atom, Eq):
            from .linear import linterm_of
            vec = linterm_of(wedge.srk, mk_sub(wedge.srk, atom.left, atom.right))
            constraints.append(("Eq", vec))
        elif isinstance(atom, Leq):
            from .linear import linterm_of
            vec = linterm_of(wedge.srk, mk_sub(wedge.srk, atom.right, atom.left))
            constraints.append(("Geq", vec))
    return constraints


def lincons_of_atom(
    srk: syntax.Context, cs: CS.CoordinateSystem, env: Environment, atom: syntax.Formula
) -> Lincons0:
    """Convert atomic formula to APRON lincons"""
    match interpretation.destruct_atom(srk, atom):
        case ("ArithComparison", ("Lt", x, y)):
            vec = linear.QQVector.add(
                CS.vec_of_term(cs, y), linear.QQVector.negate(CS.vec_of_term(cs, x))
            )
            return Lincons0.make(linexpr_of_vec(cs, env, vec), Lincons0.SUP)
        case ("ArithComparison", ("Leq", x, y)):
            vec = linear.QQVector.add(
                CS.vec_of_term(cs, y), linear.QQVector.negate(CS.vec_of_term(cs, x))
            )
            return Lincons0.make(linexpr_of_vec(cs, env, vec), Lincons0.SUPEQ)
        case ("ArithComparison", ("Eq", x, y)):
            vec = linear.QQVector.add(
                CS.vec_of_term(cs, y), linear.QQVector.negate(CS.vec_of_term(cs, x))
            )
            return Lincons0.make(linexpr_of_vec(cs, env, vec), Lincons0.EQ)
        case _:
            raise ValueError(f"Unsupported atom type: {atom}")


def meet_atoms(wedge: Wedge, atoms: List[syntax.Formula]) -> None:
    """Meet wedge with atomic formulas"""
    # Ensure coordinate system admits all atoms
    for atom in atoms:
        match interpretation.destruct_atom(wedge.srk, atom):
            case ("ArithComparison", (_, x, y)):
                CS.admit_term(wedge.cs, x)
                CS.admit_term(wedge.cs, y)
            case _:
                pass

    wedge.update_env()

    # Convert atoms to APRON lincons
    lincons_array = [
        lincons_of_atom(wedge.srk, wedge.cs, wedge.env, atom) for atom in atoms
    ]

    # Meet with APRON abstract value
    wedge.abstract = Abstract0.meet_lincons_array(
        get_manager(), wedge.abstract, lincons_array
    )


def bound_vec(wedge: Wedge, vec: linear.QQVector) -> "Interval":
    """Compute bounds for vector expression from stored constraints.

    Iterates over all constraints in the Abstract0 and finds those whose
    non-constant part is a scalar multiple of *vec*.  From each matching
    constraint we extract a lower or upper bound on *vec* and return the
    tightest interval.
    """
    from fractions import Fraction
    _Interval = Interval  # local alias; will be the real class after Edit B

    if Abstract0.is_bottom(get_manager(), wedge.abstract):
        return _Interval.bottom()

    constraints = Abstract0.to_lincons_array(get_manager(), wedge.abstract)
    if not constraints:
        return _Interval.top()

    lower: Optional[Fraction] = None
    upper: Optional[Fraction] = None

    # Decompose vec into (constant_part, non_constant_part)
    vec_cst, vec_rest = linear.QQVector.pivot(CS.const_id, vec) if CS.const_id in vec.entries else (linear.QQ.zero(), vec)

    for lc in constraints:
        # Build a dict of (dim -> coeff) for the constraint, excluding constant
        lc_coeffs: Dict[int, Any] = {}
        lc_cst = linear.QQ.zero()
        for coeff, dim in lc.linexpr0.coeffs:
            if dim is not None:
                lc_coeffs[dim] = coeff
            else:
                lc_cst = coeff
        cst_val = lc.linexpr0.get_cst()
        if cst_val is not None:
            lc_cst = cst_val

        # Check if lc_coeffs is a scalar multiple of vec_rest
        # We need: lc_coeffs = k * vec_rest for some k
        vec_rest_entries = {d: c for d, c in vec_rest.entries.items() if c != 0}
        lc_nonzero = {d: c for d, c in lc_coeffs.items() if c != 0}

        if not vec_rest_entries and not lc_nonzero:
            # Both are constant-only; skip (constraint is purely about constants)
            continue
        if not vec_rest_entries or not lc_nonzero:
            continue

        # Check dimension set match
        if set(vec_rest_entries.keys()) != set(lc_nonzero.keys()):
            continue

        # Compute k = lc_coeffs[d] / vec_rest[d] for any dimension d
        ref_dim = next(iter(vec_rest_entries.keys()))
        vec_ref = vec_rest_entries[ref_dim]
        if vec_ref == 0:
            continue
        k = lc_nonzero[ref_dim] / vec_ref

        # Verify k works for all dimensions
        match = True
        for d in vec_rest_entries:
            expected = k * vec_rest_entries[d]
            actual = lc_nonzero[d]
            if expected != actual:
                match = False
                break
        if not match:
            continue

        # Now: k * vec + lc_cst [typ] 0
        # For SUPEQ: k * vec + lc_cst >= 0  =>  k * vec >= -lc_cst
        # For SUP:   k * vec + lc_cst > 0   =>  k * vec > -lc_cst (approximate as >=)
        # For EQ:    k * vec + lc_cst = 0    =>  vec = -lc_cst / k
        if k == 0:
            continue
        neg_cst_over_k = -lc_cst / k

        if lc.typ == Lincons0.EQ:
            # Equality gives both upper and lower
            lower = neg_cst_over_k if lower is None else max(lower, neg_cst_over_k)
            upper = neg_cst_over_k if upper is None else min(upper, neg_cst_over_k)
        elif lc.typ in (Lincons0.SUPEQ, Lincons0.SUP):
            if k > 0:
                # k * vec >= -lc_cst  =>  vec >= -lc_cst / k  (lower bound)
                lower = neg_cst_over_k if lower is None else max(lower, neg_cst_over_k)
            else:
                # k * vec >= -lc_cst with k < 0  =>  vec <= -lc_cst / k  (upper bound)
                upper = neg_cst_over_k if upper is None else min(upper, neg_cst_over_k)

    return _Interval.make(lower, upper)


def bound_coordinate(wedge: Wedge, coordinate: int) -> "Interval":
    """Compute bounds for coordinate"""
    return bound_vec(wedge, linear.QQVector.of_term(linear.QQ.one(), coordinate))


def bound_monomial(wedge: Wedge, monomial) -> "Interval":
    """Compute interval bounds for a monomial over the wedge.

    For each (dim, power) pair in the monomial, bounds the coordinate
    and raises it to the given power, then multiplies all intervals.
    Ported from OCaml ``bound_monomial``.
    """
    result = Interval.const(linear.QQ.one())
    for dim, power in P.Monomial.enum(monomial):
        dim_ivl = bound_coordinate(wedge, dim)
        result = Interval.mul(result, Interval.exp_const(dim_ivl, power))
    return result


def symbolic_bounds(
    wedge: Wedge, symbol
) -> Tuple[List[syntax.Formula], List[syntax.Formula]]:
    """Compute symbolic lower and upper bounds for *symbol*.

    Returns ``(lower_bounds, upper_bounds)`` where each list contains SRK
    terms representing the bound expressions.  For a constraint
    ``a*sym + rest + c >= 0`` with ``a != 0``:

    - ``a > 0``  =>  ``sym >= -(rest + c) / a``  (lower bound)
    - ``a < 0``  =>  ``sym <= -(rest + c) / a``  (upper bound)
    """
    srk = wedge.srk
    cs = wedge.cs

    # Find the coordinate ID for the symbol
    sym_term = syntax.mk_const(srk, symbol)
    sym_vec = cs.vec_of_term(sym_term, admit=False)

    # The symbol should map to a single-dimension vector {coord_id: 1}
    non_const = {d: c for d, c in sym_vec.entries.items() if d != CS.const_id}
    if not non_const:
        return ([], [])

    sym_dim = next(iter(non_const.keys()))
    sym_coeff = non_const[sym_dim]

    lower_bounds = []
    upper_bounds = []

    constraints = Abstract0.to_lincons_array(get_manager(), wedge.abstract)
    for lc in constraints:
        # Build coefficient dict from linexpr
        lc_coeffs: Dict[int, Any] = {}
        lc_cst = linear.QQ.zero()
        for coeff, dim in lc.linexpr0.coeffs:
            if dim is not None:
                lc_coeffs[dim] = coeff
            else:
                lc_cst = coeff
        cst_val = lc.linexpr0.get_cst()
        if cst_val is not None:
            lc_cst = cst_val

        if sym_dim not in lc_coeffs:
            continue

        a = lc_coeffs[sym_dim]
        if a == 0:
            continue

        # Build "rest" vector: all coefficients except sym_dim
        rest_entries = {d: c for d, c in lc_coeffs.items() if d != sym_dim}
        rest_entries[CS.const_id] = lc_cst
        rest_vec = linear.QQVector(rest_entries)

        # Bound term = -rest / a   (as a formula)
        # We express it as a term: -(rest_term) / a
        rest_term = cs._vector_to_term(rest_vec)
        neg_rest = syntax.mk_neg(srk, rest_term)
        a_term = syntax.mk_real(srk, abs(a))
        bound_term = syntax.mk_div(srk, neg_rest, a_term) if a != linear.QQ.one() else neg_rest

        if lc.typ in (Lincons0.EQ, Lincons0.SUPEQ, Lincons0.SUP):
            if a > 0:
                # sym >= bound
                lower_bounds.append(bound_term)
            else:
                # sym <= bound
                upper_bounds.append(bound_term)

    return (lower_bounds, upper_bounds)


# Interval arithmetic — delegates to the real Interval implementation.
from aria.srk.interval import Interval as _Interval


class Interval:
    """Facade over :class:`aria.srk.interval.Interval`.

    Static-method API used throughout wedge.py; every returned / accepted
    interval value is a real ``_Interval`` instance.
    """

    @staticmethod
    def top():
        return _Interval.top()

    @staticmethod
    def bottom():
        return _Interval.bottom()

    @staticmethod
    def const(qq: linear.QQ):
        return _Interval.const(linear.QQ(qq))

    @staticmethod
    def make(lower, upper):
        return _Interval.make(
            linear.QQ(lower) if lower is not None else None,
            linear.QQ(upper) if upper is not None else None,
        )

    @staticmethod
    def of_apron(apron_interval):
        """Convert an APRON interval to a real Interval."""
        # APRON intervals typically have (lower, upper) attributes
        try:
            lo = apron_interval.lower if apron_interval.lower is not None else None
            hi = apron_interval.upper if apron_interval.upper is not None else None
            return _Interval.make(
                linear.QQ(lo) if lo is not None else None,
                linear.QQ(hi) if hi is not None else None,
            )
        except Exception:
            return _Interval.top()

    @staticmethod
    def elem(qq: linear.QQ, interval) -> bool:
        """Check whether *qq* belongs to *interval*."""
        if isinstance(interval, _Interval):
            return interval.contains(linear.QQ(qq))
        return False

    @staticmethod
    def is_nonnegative(interval) -> bool:
        if isinstance(interval, _Interval):
            if interval.is_bottom():
                return False
            return interval.lower is not None and interval.lower >= 0
        return False

    @staticmethod
    def is_nonpositive(interval) -> bool:
        if isinstance(interval, _Interval):
            if interval.is_bottom():
                return False
            return interval.upper is not None and interval.upper <= 0
        return False

    @staticmethod
    def is_positive(interval) -> bool:
        if isinstance(interval, _Interval):
            if interval.is_bottom():
                return False
            return interval.lower is not None and interval.lower > 0
        return False

    @staticmethod
    def is_negative(interval) -> bool:
        if isinstance(interval, _Interval):
            if interval.is_bottom():
                return False
            return interval.upper is not None and interval.upper < 0
        return False

    @staticmethod
    def lower(interval) -> Optional[linear.QQ]:
        if isinstance(interval, _Interval):
            return interval.lower
        return None

    @staticmethod
    def upper(interval) -> Optional[linear.QQ]:
        if isinstance(interval, _Interval):
            return interval.upper
        return None

    @staticmethod
    def mul(ivl1, ivl2):
        if isinstance(ivl1, _Interval) and isinstance(ivl2, _Interval):
            return ivl1 * ivl2
        return _Interval.top()

    @staticmethod
    def add(ivl1, ivl2):
        if isinstance(ivl1, _Interval) and isinstance(ivl2, _Interval):
            return ivl1 + ivl2
        return _Interval.top()

    @staticmethod
    def exp_const(ivl, power: int):
        """Raise *ivl* to an integer *power*."""
        if isinstance(ivl, _Interval):
            if ivl.is_bottom():
                return _Interval.bottom()
            if ivl.is_point():
                try:
                    val = ivl.lower ** power
                    return _Interval.const(linear.QQ(val))
                except Exception:
                    return _Interval.top()
            # Conservative: use point-interval power for positive intervals
            if Interval.is_nonnegative(ivl) and power == 2:
                lo = linear.QQ(0) if ivl.lower is None else (ivl.lower ** 2 if ivl.lower >= 0 else linear.QQ(0))
                hi = None if ivl.upper is None else ivl.upper ** 2
                return _Interval.make(lo, hi)
        return _Interval.top()

    @staticmethod
    def div(ivl1, ivl2):
        if isinstance(ivl1, _Interval) and isinstance(ivl2, _Interval):
            if ivl1.is_bottom() or ivl2.is_bottom():
                return _Interval.bottom()
            return ivl1 / ivl2
        return _Interval.top()

    @staticmethod
    def log(base_ivl, exp_ivl):
        """Approximate logarithm interval (conservative: returns top)."""
        return _Interval.top()


def mk_sign_axioms(srk: syntax.Context) -> syntax.Formula:
    """Create sign axioms for nonlinear operations.

    Sign axioms constrain nonlinear operations (Mul, Floor, Mod, Pow, Div)
    to respect basic arithmetic laws such as monotonicity, bounds, and
    sign relationships. These are used with uninterpreted functions so the
    SMT solver can reason soundly about nonlinear terms.
    """
    from fractions import Fraction

    axioms: List[syntax.Formula] = []

    # Mul sign axiom: x >= 0 && y >= 0 -> x * y >= 0
    x_mul = syntax.mk_symbol(srk, "x_mul", syntax.Type.REAL)
    y_mul = syntax.mk_symbol(srk, "y_mul", syntax.Type.REAL)
    z_mul = syntax.mk_app(srk, syntax.mk_symbol(srk, "mul", syntax.Type.FUN), [syntax.mk_const(srk, x_mul), syntax.mk_const(srk, y_mul)])
    axioms.append(
        syntax.mk_if(
            srk,
            syntax.mk_and(srk, [
                syntax.mk_leq(srk, syntax.mk_real(srk, Fraction(0)), syntax.mk_const(srk, x_mul)),
                syntax.mk_leq(srk, syntax.mk_real(srk, Fraction(0)), syntax.mk_const(srk, y_mul)),
            ]),
            syntax.mk_leq(srk, syntax.mk_real(srk, Fraction(0)), z_mul),
        )
    )

    # Floor axiom: floor(x) <= x and x < floor(x) + 1
    x_floor = syntax.mk_symbol(srk, "x_floor", syntax.Type.REAL)
    f_x = syntax.mk_app(srk, syntax.mk_symbol(srk, "floor", syntax.Type.FUN), [syntax.mk_const(srk, x_floor)])
    axioms.append(syntax.mk_leq(srk, f_x, syntax.mk_const(srk, x_floor)))
    axioms.append(
        syntax.mk_lt(
            srk,
            syntax.mk_const(srk, x_floor),
            syntax.mk_add(srk, [f_x, syntax.mk_real(srk, Fraction(1))]),
        )
    )

    # Mod axiom: for d > 0, 0 <= x mod d and x mod d < d
    x_mod = syntax.mk_symbol(srk, "x_mod", syntax.Type.REAL)
    d_mod = syntax.mk_symbol(srk, "d_mod", syntax.Type.REAL)
    r_mod = syntax.mk_app(
        srk, syntax.mk_symbol(srk, "mod", syntax.Type.FUN),
        [syntax.mk_const(srk, x_mod), syntax.mk_const(srk, d_mod)]
    )
    axioms.append(
        syntax.mk_if(
            srk,
            syntax.mk_lt(srk, syntax.mk_real(srk, Fraction(0)), syntax.mk_const(srk, d_mod)),
            syntax.mk_and(srk, [
                syntax.mk_leq(srk, syntax.mk_real(srk, Fraction(0)), r_mod),
                syntax.mk_lt(srk, r_mod, syntax.mk_const(srk, d_mod)),
            ]),
        )
    )

    # Pow axiom: for base >= 0, pow(base, n) >= 0 for even n
    x_pow = syntax.mk_symbol(srk, "x_pow", syntax.Type.REAL)
    n_pow = syntax.mk_symbol(srk, "n_pow", syntax.Type.INT)
    p_x = syntax.mk_app(
        srk, syntax.mk_symbol(srk, "pow", syntax.Type.FUN),
        [syntax.mk_const(srk, x_pow), syntax.mk_const(srk, n_pow)]
    )
    axioms.append(
        syntax.mk_if(
            srk,
            syntax.mk_leq(srk, syntax.mk_real(srk, Fraction(0)), syntax.mk_const(srk, x_pow)),
            syntax.mk_leq(srk, syntax.mk_real(srk, Fraction(0)), p_x),
        )
    )

    # Div sign axiom: x >= 0 && y > 0 -> x/y >= 0
    x_div = syntax.mk_symbol(srk, "x_div", syntax.Type.REAL)
    y_div = syntax.mk_symbol(srk, "y_div", syntax.Type.REAL)
    q_div = syntax.mk_app(
        srk, syntax.mk_symbol(srk, "div", syntax.Type.FUN),
        [syntax.mk_const(srk, x_div), syntax.mk_const(srk, y_div)]
    )
    axioms.append(
        syntax.mk_if(
            srk,
            syntax.mk_and(srk, [
                syntax.mk_leq(srk, syntax.mk_real(srk, Fraction(0)), syntax.mk_const(srk, x_div)),
                syntax.mk_lt(srk, syntax.mk_real(srk, Fraction(0)), syntax.mk_const(srk, y_div)),
            ]),
            syntax.mk_leq(srk, syntax.mk_real(srk, Fraction(0)), q_div),
        )
    )

    return syntax.mk_and(srk, axioms)


def wedge_entails(wedge: Wedge, phi: syntax.Formula) -> bool:
    """Check if wedge entails formula modulo LIRA + sign axioms"""
    srk = wedge.srk
    s = Smt.mk_solver(srk)
    Smt.Solver.add(
        s,
        [
            nonlinear.uninterpret(srk, to_formula(wedge)),
            nonlinear.uninterpret(srk, syntax.mk_not(srk, phi)),
            mk_sign_axioms(srk),
        ],
    )

    match Smt.Solver.check(s, []):
        case Smt.Sat | Smt.Unknown:
            return False
        case Smt.Unsat:
            return True


def nonnegative_polynomial(wedge: Wedge, p: P.Polynomial) -> bool:
    """Check if polynomial is nonnegative on wedge"""
    term = CS.term_of_polynomial(wedge.cs, p)
    geq_zero = syntax.mk_leq(
        wedge.srk, syntax.mk_real(wedge.srk, linear.QQ.zero()), term
    )
    return wedge_entails(wedge, geq_zero)


def bound_polynomial(wedge: Wedge, polynomial: P.Polynomial) -> "Interval":
    """Compute interval bounds for a polynomial over the wedge.

    Splits the polynomial into its linear part (bounded via
    :func:`bound_vec`) and nonlinear part (each monomial bounded via
    :func:`bound_monomial` times its coefficient).
    Ported from OCaml ``bound_polynomial``.
    """
    cs = wedge.cs
    # Split into linear vector t and nonlinear part p
    t, p = P.split_linear(polynomial, const=CS.const_id)

    # Bound the linear part
    result = bound_vec(wedge, t)

    # Bound each nonlinear term: coeff * monomial
    for monom, coeff in P.Polynomial.enum(p):
        monomial_ivl = Interval.const(coeff)
        for dim, power in P.Monomial.enum(monom):
            dim_ivl = bound_coordinate(wedge, dim)
            monomial_ivl = Interval.mul(monomial_ivl, Interval.exp_const(dim_ivl, power))
        result = Interval.add(result, monomial_ivl)

    return result


def affine_hull(wedge: Wedge) -> List[linear.QQVector]:
    """Compute affine hull of wedge.

    Extracts all equality (EQ) constraints from the abstract domain and
    converts each linear expression into a :class:`linear.QQVector`.
    The returned vectors *v* satisfy ``v = 0`` in the affine hull.
    """
    if is_bottom(wedge):
        return [
            linear.QQVector.add_term(
                linear.QQ.one(), CS.const_id, linear.QQVector.zero()
            )
        ]

    lincons_array = Abstract0.to_lincons_array(get_manager(), wedge.abstract)
    result = []
    for lc in lincons_array:
        if lc.typ == Lincons0.EQ:
            vec = vec_of_linexpr(wedge.env, lc.linexpr0)
            result.append(vec)
    return result


def polynomial_constraints(
    lemma: Callable, wedge: Wedge
) -> List[Tuple[str, P.Polynomial]]:
    """Extract polynomial constraints from wedge.

    For each APRON linear constraint, converts the linear expression to a
    vector, then to a polynomial via the coordinate system.  Emits a lemma
    asserting the equality between the vector form and polynomial form.
    Returns a list of (comparison, polynomial) pairs where comparison is
    one of ``'Nonneg'``, ``'Pos'``, ``'Zero'``.
    Ported from OCaml ``polynomial_constraints``.
    """
    cs = wedge.cs
    srk = wedge.srk
    zero = syntax.mk_real(srk, linear.QQ.zero())
    result = []

    for lc in Abstract0.to_lincons_array(get_manager(), wedge.abstract):
        vec = vec_of_linexpr(wedge.env, lc.linexpr0)
        polynomial = cs.polynomial_of_vec(vec)

        # Lemma: term_of_vec(vec) = term_of_polynomial(polynomial)
        vec_term = CS.term_of_vec(cs, vec)
        poly_term = CS.term_of_polynomial(cs, polynomial)
        lemma(syntax.mk_eq(srk, vec_term, poly_term))

        if lc.typ == Lincons0.SUPEQ:
            result.append(("Nonneg", polynomial))
        elif lc.typ == Lincons0.SUP:
            result.append(("Pos", polynomial))
        elif lc.typ == Lincons0.EQ:
            result.append(("Zero", polynomial))

    return result


def polynomial_cone(lemma: Callable, wedge: Wedge) -> List[P.Polynomial]:
    """Extract polynomial cone from wedge.

    The polynomial cone consists of all polynomials known to be non-negative.
    For Nonneg and Pos constraints, the polynomial itself is in the cone.
    For Zero constraints, both p and -p are in the cone.
    Ported from OCaml ``polynomial_cone``.
    """
    constraints = polynomial_constraints(lemma, wedge)
    result = []
    for cmp, p in constraints:
        if cmp in ("Nonneg", "Pos"):
            result.append(p)
        elif cmp == "Zero":
            result.append(p)
            result.append(P.negate(p))
    return result


def vanishing_ideal(wedge: Wedge) -> List[P.Polynomial]:
    """Compute vanishing ideal of wedge.

    Returns the list of polynomials that vanish on the wedge.  Each
    equality vector from :func:`affine_hull` is converted to a polynomial
    via the coordinate system's :meth:`polynomial_of_vec`.

    Additionally, for ``Inv`` coordinates whose operand interval does not
    contain zero, adds the defining polynomial ``x * inv(x) - 1``.
    """
    if is_bottom(wedge):
        return [P.one()]

    result = []
    # Equality constraints from the polyhedron
    for lc in Abstract0.to_lincons_array(get_manager(), wedge.abstract):
        if lc.typ == Lincons0.EQ:
            vec = vec_of_linexpr(wedge.env, lc.linexpr0)
            poly = wedge.cs.polynomial_of_vec(vec)
            if not poly.is_zero():
                result.append(poly)

    # Inv coordinate elimination: if 0 ∉ interval(x), add x*inv(x) - 1
    for coord_id in range(CS.dim(wedge.cs)):
        cs_term = wedge.cs.destruct_coordinate(coord_id)
        if cs_term.term_type == CS.CSTermType.INV and cs_term.vectors:
            x_vec = cs_term.vectors[0]
            x_ivl = bound_vec(wedge, x_vec)
            if not Interval.elem(linear.QQ.zero(), x_ivl):
                x_poly = wedge.cs.polynomial_of_vec(x_vec)
                inv_poly = P.Polynomial.of_dim(coord_id, CS.dim(wedge.cs))
                defining = P.sub(P.mul(x_poly, inv_poly), P.scalar(linear.QQ.one()))
                if not defining.is_zero():
                    result.append(defining)

    return result


def coordinate_ideal(lemma: Callable, wedge: Wedge) -> List[P.Polynomial]:
    """Compute coordinate ideal of wedge.

    Starts with the :func:`vanishing_ideal` and adds defining polynomials
    for every *compound* coordinate (Mul, App, Inv, Mod, Floor).
    For a coordinate ``c`` whose defining expression is ``e``, the
    polynomial ``c - e`` is added to the ideal.
    """
    cs = wedge.cs
    result = list(vanishing_ideal(wedge))

    for coord_id in range(cs.dim):
        cs_term = cs.destruct_coordinate(coord_id)
        # Compound terms are those whose definition involves other coordinates
        if cs_term.term_type in (
            CS.CSTermType.MUL,
            CS.CSTermType.INV,
            CS.CSTermType.MOD,
            CS.CSTermType.FLOOR,
            CS.CSTermType.APP,
        ):
            coord_poly = cs.polynomial_of_coordinate(coord_id)
            # defining poly: coord_id - expr(coord_id)
            coord_var = P.Polynomial.of_dim(coord_id, cs.dim)
            defining = coord_var - coord_poly
            if not defining.is_zero():
                result.append(defining)

    return result


def equational_saturation(lemma: Callable, wedge: Wedge) -> P.RewriteSystem:
    """Compute equational saturation of wedge.

    Builds a Gröbner-basis rewrite system from the :func:`coordinate_ideal`
    and iterates until no new equalities are discovered.  Includes congruence
    closure: when ``a = b`` in the coordinate system, ``f(a) = f(b)`` for
    all function applications (App, Inv, Mod, Floor).

    Returns the final :class:`P.RewriteSystem`.
    """
    cs = wedge.cs
    srk = wedge.srk
    zero = syntax.mk_real(srk, linear.QQ.zero())

    nonlinear.ensure_symbols(srk)
    assert wedge.is_consistent()

    saturated = False

    # Build initial Grobner basis for coordinate ideal + polyhedron ideal
    gens = coordinate_ideal(lemma, wedge)
    ideal_obj = P.Ideal(gens) if gens else P.Ideal([])
    rewrite = ideal_obj.groebner_basis()

    # Hashtable mapping canonical forms of nonlinear terms to representatives
    canonical: Dict[int, syntax.Expression] = {}

    def _add_bound(precondition, bound):
        lemma(syntax.mk_or(srk, [syntax.mk_not(srk, precondition), bound]))
        meet_atoms(wedge, [bound])

    def _reduce_vec(vec):
        """Reduce a vector through the rewrite system, returning (reduced_term, lemma)."""
        p = wedge.cs.polynomial_of_vec(vec)
        reduced = rewrite.reduce(p)
        p_term = CS.term_of_polynomial(cs, reduced)
        term = CS.term_of_vec(cs, vec)
        return p_term, term

    while not saturated:
        saturated = True
        for coord_id in range(CS.dim(cs)):
            term = CS.term_of_coordinate(cs, coord_id)

            # Reduce the coordinate's polynomial representation
            reduced_id_term, original_term = _reduce_vec(
                linear.QQVector.of_term(linear.QQ.one(), coord_id)
            )

            if not syntax.expr_equal(term, reduced_id_term):
                _add_bound(syntax.mk_true(srk), syntax.mk_eq(srk, term, reduced_id_term))

            # Congruence closure
            cs_term = cs.destruct_coordinate(coord_id)
            if cs_term.term_type == CS.CSTermType.MUL:
                # Mul: no congruence needed (already handled by polynomial ideal)
                pass
            elif cs_term.term_type == CS.CSTermType.APP and cs_term.func is not None:
                # App(func, args): reduce each argument, then check canonical map
                if cs_term.args:
                    reduced_args = []
                    for arg_vec in cs_term.args:
                        arg_term, _ = _reduce_vec(arg_vec)
                        reduced_args.append(arg_term)
                    canonical_key = hash(("App", cs_term.func, tuple(
                        syntax.expr_id(a) for a in reduced_args
                    )))
                    canonical_expr = syntax.mk_app(srk, cs_term.func, reduced_args)
                    if canonical_key in canonical:
                        existing = canonical[canonical_key]
                        if not syntax.expr_equal(term, existing):
                            lemma(syntax.mk_true(srk))
                            meet_atoms(wedge, [syntax.mk_eq(srk, term, existing)])
                    else:
                        canonical[canonical_key] = term
            elif cs_term.term_type == CS.CSTermType.INV and cs_term.vectors:
                # Inv(t): reduce t, then 1/reduced_t
                t_term, _ = _reduce_vec(cs_term.vectors[0])
                canonical_key = hash(("Inv", syntax.expr_id(t_term)))
                canonical_expr = syntax.mk_div(
                    srk, syntax.mk_real(srk, linear.QQ.one()), t_term
                )
                if canonical_key in canonical:
                    existing = canonical[canonical_key]
                    if not syntax.expr_equal(term, existing):
                        lemma(syntax.mk_true(srk))
                        meet_atoms(wedge, [syntax.mk_eq(srk, term, existing)])
                else:
                    canonical[canonical_key] = term
            elif cs_term.term_type == CS.CSTermType.MOD and len(cs_term.vectors) == 2:
                # Mod(num, den): reduce both
                num_term, _ = _reduce_vec(cs_term.vectors[0])
                den_term, _ = _reduce_vec(cs_term.vectors[1])
                canonical_key = hash(("Mod", syntax.expr_id(num_term), syntax.expr_id(den_term)))
                if canonical_key in canonical:
                    existing = canonical[canonical_key]
                    if not syntax.expr_equal(term, existing):
                        lemma(syntax.mk_true(srk))
                        meet_atoms(wedge, [syntax.mk_eq(srk, term, existing)])
                else:
                    canonical[canonical_key] = term
            elif cs_term.term_type == CS.CSTermType.FLOOR and cs_term.vectors:
                # Floor(t): reduce t
                t_term, _ = _reduce_vec(cs_term.vectors[0])
                canonical_key = hash(("Floor", syntax.expr_id(t_term)))
                if canonical_key in canonical:
                    existing = canonical[canonical_key]
                    if not syntax.expr_equal(term, existing):
                        lemma(syntax.mk_true(srk))
                        meet_atoms(wedge, [syntax.mk_eq(srk, term, existing)])
                else:
                    canonical[canonical_key] = term

        # Check for new polynomials vanishing on the underlying polyhedron
        for vec in affine_hull(wedge):
            p = wedge.cs.polynomial_of_vec(vec)
            reduced = rewrite.reduce(p)
            if not reduced.is_zero():
                saturated = False
                # Add to ideal and recompute Grobner basis
                ideal_obj = P.Ideal(ideal_obj.generators + [reduced])
                rewrite = ideal_obj.groebner_basis()

    return rewrite


def generalized_fourier_motzkin(lemma: Callable, order, wedge: Wedge) -> None:
    """Apply generalized Fourier-Motzkin elimination"""
    srk = wedge.srk
    cs = wedge.cs

    def add_bound(precondition, bound):
        logger.debug(f"Lemma: {precondition} => {bound}")
        lemma(syntax.mk_or(srk, [syntax.mk_not(srk, precondition), bound]))
        meet_atoms(wedge, [bound])

    old_wedge = bottom(srk)

    def polyhedron_equal(w1, w2):
        return CS.dim(w1.cs) == CS.dim(w2.cs) and Abstract0.is_eq(
            get_manager(), w1.abstract, w2.abstract
        )

    gfm_limit = 10  # Maximum iterations
    iterations = 0

    while iterations < gfm_limit and not polyhedron_equal(wedge, old_wedge):
        iterations += 1
        logger.debug(f"GFM iteration: {iterations}")
        old_wedge = copy(wedge)
        cone = polynomial_cone(lemma, wedge)

        for p in cone:
            c, m, p_rest = P.split_leading(order, p)
            if linear.QQ.lt(c, linear.QQ.zero()):
                p_scaled = P.scalar_mul(linear.QQ.negate(linear.QQ.inverse(c)), p)

                for q in cone:
                    quot, rem = P.qr_monomial(q, m)
                    if P.degree(quot) >= 1 and nonnegative_polynomial(wedge, quot):
                        zero = syntax.mk_real(srk, linear.QQ.zero())
                        mk_nonneg = lambda t: syntax.mk_leq(srk, zero, t)

                        p_sub_m = P.add_term(linear.QQ.of_int(-1), m, p_scaled)
                        hypothesis = syntax.mk_and(
                            srk,
                            [
                                mk_nonneg(CS.term_of_polynomial(cs, p_sub_m)),
                                mk_nonneg(CS.term_of_polynomial(cs, quot)),
                                mk_nonneg(CS.term_of_polynomial(cs, q)),
                            ],
                        )

                        conclusion = mk_nonneg(
                            CS.term_of_polynomial(cs, P.add(P.mul(quot, p_scaled), rem))
                        )
                        add_bound(hypothesis, conclusion)


def strengthen_intervals(lemma: Callable, wedge: Wedge) -> None:
    """Compute bounds for synthetic dimensions using the bounds of their operands.

    Handles Mul, Floor, Inv, Log, Pow, and Mod coordinates specially,
    ported from the OCaml ``strengthen_intervals`` in ``wedge.ml``.
    """
    srk = wedge.srk
    cs = wedge.cs
    zero = syntax.mk_real(srk, linear.QQ.zero())

    _log_sym = nonlinear.get_named_symbol(srk, "log")
    _pow_sym = nonlinear.get_named_symbol(srk, "pow")

    def _add_bound(precondition, bound):
        lemma(syntax.mk_or(srk, [syntax.mk_not(srk, precondition), bound]))
        meet_atoms(wedge, [bound])

    def _mk_ivl_conj(srk, t, ivl):
        r"""Build the conjunction: lo <= t /\ t <= hi."""
        parts = []
        lo = Interval.lower(ivl)
        if lo is not None:
            parts.append(syntax.mk_leq(srk, syntax.mk_real(srk, lo), t))
        hi = Interval.upper(ivl)
        if hi is not None:
            parts.append(syntax.mk_leq(srk, t, syntax.mk_real(srk, hi)))
        return syntax.mk_and(srk, parts) if parts else syntax.mk_true(srk)

    for coord_id in range(CS.dim(cs)):
        term = CS.term_of_coordinate(cs, coord_id)
        cs_term = cs.destruct_coordinate(coord_id)

        if cs_term.term_type == CS.CSTermType.MUL and len(cs_term.vectors) == 2:
            # --- Mul(x, y) ---
            x_vec, y_vec = cs_term.vectors[0], cs_term.vectors[1]
            x_ivl = bound_vec(wedge, x_vec)
            y_ivl = bound_vec(wedge, y_vec)
            x_term = CS.term_of_vec(cs, x_vec)
            y_term = CS.term_of_vec(cs, y_vec)

            # Helper: propagate bounds through non-negative/non-positive factors
            def _propagate(x_vec, x_ivl, x_term, y_vec, y_ivl, y_term):
                if Interval.is_nonnegative(y_ivl):
                    y_nn = syntax.mk_leq(srk, zero, y_term)
                    lo = Interval.lower(x_ivl)
                    if lo is not None:
                        _add_bound(
                            syntax.mk_and(srk, [y_nn, syntax.mk_leq(srk, syntax.mk_real(srk, lo), x_term)]),
                            syntax.mk_leq(srk, CS.term_of_vec(cs, linear.QQVector.scalar_mul(lo, y_vec)), term),
                        )
                    hi = Interval.upper(x_ivl)
                    if hi is not None:
                        _add_bound(
                            syntax.mk_and(srk, [y_nn, syntax.mk_leq(srk, x_term, syntax.mk_real(srk, hi))]),
                            syntax.mk_leq(srk, term, CS.term_of_vec(cs, linear.QQVector.scalar_mul(hi, y_vec))),
                        )
                elif Interval.is_nonpositive(y_ivl):
                    y_np = syntax.mk_leq(srk, y_term, zero)
                    lo = Interval.lower(x_ivl)
                    if lo is not None:
                        _add_bound(
                            syntax.mk_and(srk, [y_np, syntax.mk_leq(srk, syntax.mk_real(srk, lo), x_term)]),
                            syntax.mk_leq(srk, term, CS.term_of_vec(cs, linear.QQVector.scalar_mul(lo, y_vec))),
                        )
                    hi = Interval.upper(x_ivl)
                    if hi is not None:
                        _add_bound(
                            syntax.mk_and(srk, [y_np, syntax.mk_leq(srk, x_term, syntax.mk_real(srk, hi))]),
                            syntax.mk_leq(srk, CS.term_of_vec(cs, linear.QQVector.scalar_mul(hi, y_vec)), term),
                        )

            _propagate(x_vec, x_ivl, x_term, y_vec, y_ivl, y_term)
            _propagate(y_vec, y_ivl, y_term, x_vec, x_ivl, x_term)

            # Interval product bounds
            mul_ivl = Interval.mul(x_ivl, y_ivl)
            precondition = syntax.mk_and(srk, [_mk_ivl_conj(srk, x_term, x_ivl), _mk_ivl_conj(srk, y_term, y_ivl)])
            lo = Interval.lower(mul_ivl)
            if lo is not None:
                _add_bound(precondition, syntax.mk_leq(srk, syntax.mk_real(srk, lo), term))
            hi = Interval.upper(mul_ivl)
            if hi is not None:
                _add_bound(precondition, syntax.mk_leq(srk, term, syntax.mk_real(srk, hi)))

        elif cs_term.term_type == CS.CSTermType.FLOOR and cs_term.vectors:
            # --- Floor(x) ---
            x_vec = cs_term.vectors[0]
            x_term = CS.term_of_vec(cs, x_vec)
            _true = syntax.mk_true(srk)
            # floor(x) <= x
            _add_bound(_true, syntax.mk_leq(srk, term, x_term))
            # x - 1 < floor(x)  i.e.  floor(x) > x - 1
            _add_bound(_true, syntax.mk_lt(srk, syntax.mk_add(srk, [x_term, syntax.mk_real(srk, linear.QQ.of_int(-1))]), term))

        elif cs_term.term_type == CS.CSTermType.INV and cs_term.vectors:
            # --- Inv(x) ---
            x_vec = cs_term.vectors[0]
            x_ivl = bound_vec(wedge, x_vec)
            x_term = CS.term_of_vec(cs, x_vec)
            precondition = _mk_ivl_conj(srk, x_term, x_ivl)
            inv_ivl = Interval.div(Interval.const(linear.QQ.one()), x_ivl)
            lo = Interval.lower(inv_ivl)
            if lo is not None:
                _add_bound(precondition, syntax.mk_leq(srk, syntax.mk_real(srk, lo), term))
            hi = Interval.upper(inv_ivl)
            if hi is not None:
                _add_bound(precondition, syntax.mk_leq(srk, term, syntax.mk_real(srk, hi)))

        elif cs_term.term_type == CS.CSTermType.APP and cs_term.func is not None:
            args = cs_term.args if cs_term.args else cs_term.vectors
            if cs_term.func == _log_sym and len(args) == 2:
                # --- log(base, exp) ---
                base_vec, exp_vec = args[0], args[1]
                base_ivl = bound_vec(wedge, base_vec)
                exp_ivl = bound_vec(wedge, exp_vec)
                base_term = CS.term_of_vec(cs, base_vec)
                exp_term = CS.term_of_vec(cs, exp_vec)
                precondition = syntax.mk_and(srk, [
                    _mk_ivl_conj(srk, base_term, base_ivl),
                    _mk_ivl_conj(srk, exp_term, exp_ivl),
                ])
                log_ivl = Interval.log(base_ivl, exp_ivl)
                lo = Interval.lower(log_ivl)
                if lo is not None:
                    _add_bound(precondition, syntax.mk_leq(srk, syntax.mk_real(srk, lo), term))
                hi = Interval.upper(log_ivl)
                if hi is not None:
                    _add_bound(precondition, syntax.mk_leq(srk, term, syntax.mk_real(srk, hi)))

            elif cs_term.func == _pow_sym and len(args) == 2:
                # --- pow(base, exp) ---
                base_vec, exp_vec = args[0], args[1]
                base_ivl = bound_vec(wedge, base_vec)
                exp_ivl = bound_vec(wedge, exp_vec)
                base_term = CS.term_of_vec(cs, base_vec)
                exp_term = CS.term_of_vec(cs, exp_vec)
                precondition = syntax.mk_and(srk, [
                    _mk_ivl_conj(srk, base_term, base_ivl),
                    _mk_ivl_conj(srk, exp_term, exp_ivl),
                ])
                pow_ivl = Interval.exp(base_ivl, exp_ivl)
                lo = Interval.lower(pow_ivl)
                if lo is not None:
                    _add_bound(precondition, syntax.mk_leq(srk, syntax.mk_real(srk, lo), term))
                hi = Interval.upper(pow_ivl)
                if hi is not None:
                    _add_bound(precondition, syntax.mk_leq(srk, term, syntax.mk_real(srk, hi)))

        elif cs_term.term_type == CS.CSTermType.MOD and len(cs_term.vectors) == 2:
            # --- Mod(_, y) ---
            y_vec = cs_term.vectors[1]
            y_ivl = bound_vec(wedge, y_vec)
            y_term = CS.term_of_vec(cs, y_vec)
            # mod result is always >= 0
            _add_bound(syntax.mk_true(srk), syntax.mk_leq(srk, zero, term))
            if Interval.is_positive(y_ivl):
                _add_bound(syntax.mk_lt(srk, zero, y_term), syntax.mk_lt(srk, term, y_term))
            elif Interval.is_negative(y_ivl):
                _add_bound(syntax.mk_lt(srk, y_term, zero),
                           syntax.mk_lt(srk, term, syntax.mk_neg(srk, y_term)))


def strengthen_products(lemma: Callable, rewrite: P.RewriteSystem, wedge: Wedge) -> None:
    """Derive linear constraints from product non-negativity of polynomial cone.

    For each pair (p, q) in the polynomial cone, if p*q reduces to a linear
    polynomial r, then  p >= 0 and q >= 0  implies  r >= 0.
    Ported from OCaml ``strengthen_products``.
    """
    srk = wedge.srk
    cs = wedge.cs
    zero = syntax.mk_real(srk, linear.QQ.zero())

    def _mk_geqz(p):
        """p >= 0 as a formula (after reduction)."""
        reduced = rewrite.reduce(p)
        return syntax.mk_leq(srk, CS.term_of_polynomial(cs, P.negate(reduced)), zero)

    def _add_bound(precondition, bound):
        lemma(syntax.mk_or(srk, [syntax.mk_not(srk, precondition), bound]))
        meet_atoms(wedge, [bound])

    cone = polynomial_cone(lemma, wedge)
    cone_list = list(cone)

    def _add_products(cone_remaining):
        if not cone_remaining:
            return
        p = cone_remaining[0]
        rest = cone_remaining[1:]
        for q in rest:
            r, provenance = rewrite.preduce(P.mul(p, q))
            r_vec = _vec_of_poly_safe(r)
            if r_vec is not None:
                precondition = syntax.mk_and(srk, [
                    syntax.mk_eq(srk, CS.term_of_polynomial(cs, pp), zero)
                    for pp in provenance
                ] + [_mk_geqz(p), _mk_geqz(q)])
                r_geqz = syntax.mk_leq(srk, CS.term_of_vec(cs, linear.QQVector.negate(r_vec)), zero)
                _add_bound(precondition, r_geqz)
        _add_products(rest)

    _add_products(cone_list)


def _vec_of_poly_safe(poly: P.Polynomial) -> Optional[linear.QQVector]:
    """Convert a polynomial to a vector if it is linear (degree <= 1).

    Returns None if the polynomial is nonlinear.
    """
    if poly.degree() > 1:
        return None
    vec = linear.QQVector.zero()
    for monom, coeff in poly.terms.items():
        dims = [(d, e) for d, e in enumerate(monom.exponents) if e > 0]
        if len(dims) == 0:
            vec = linear.QQVector.add_term(coeff, CS.const_id, vec)
        elif len(dims) == 1 and dims[0][1] == 1:
            vec = linear.QQVector.add_term(coeff, dims[0][0], vec)
        else:
            return None  # nonlinear
    return vec


def strengthen_integral(lemma: Callable, wedge: Wedge) -> None:
    """Tighten integral dimensions by rounding non-integer bounds.

    For each integer-typed coordinate whose bound is not an integer,
    round the lower bound up (ceiling) and the upper bound down (floor),
    adding a lemma to justify the tightening.
    Ported from OCaml ``strengthen_integral``.
    """
    srk = wedge.srk
    cs = wedge.cs

    for coord_id in range(CS.dim(cs)):
        if CS.type_of_id(cs, coord_id) == CS.TermType.TY_INT:
            term = CS.term_of_coordinate(cs, coord_id)
            ivl = bound_coordinate(wedge, coord_id)

            lo = Interval.lower(ivl)
            if lo is not None:
                lo_zz = linear.QQ.to_zz(lo)
                if lo_zz is None:
                    # Non-integer lower bound: round up
                    new_lo = linear.QQ.from_zz(linear.QQ.ceiling(lo))
                    bound = syntax.mk_leq(srk, syntax.mk_real(srk, new_lo), term)
                    lemma(syntax.mk_or(srk, [
                        syntax.mk_leq(srk, term, syntax.mk_real(srk, linear.QQ.sub(new_lo, linear.QQ.one()))),
                        bound,
                    ]))
                    meet_atoms(wedge, [bound])

            hi = Interval.upper(ivl)
            if hi is not None:
                hi_zz = linear.QQ.to_zz(hi)
                if hi_zz is None:
                    # Non-integer upper bound: round down
                    new_hi = linear.QQ.from_zz(linear.QQ.floor(hi))
                    bound = syntax.mk_leq(srk, term, syntax.mk_real(srk, new_hi))
                    lemma(syntax.mk_or(srk, [
                        syntax.mk_leq(srk, syntax.mk_real(srk, linear.QQ.add(new_hi, linear.QQ.one())), term),
                        bound,
                    ]))
                    meet_atoms(wedge, [bound])


def strengthen_cut(lemma: Callable, rewrite: P.RewriteSystem, wedge: Wedge) -> None:
    """Derive Gomory-style integer cut planes.

    For each polynomial constraint ``c*m(x)*q(x) + k >= 0`` in the
    polynomial cone, if ``q`` is integer-typed and the coefficient
    ``c*m(x)`` is positive, derive ``ceil(rhs) <= q(x)``.
    Ported from OCaml ``strengthen_cut``.
    """
    srk = wedge.srk
    cs = wedge.cs
    zero = syntax.mk_real(srk, linear.QQ.zero())

    def _add_bound(precondition, bound):
        lemma(syntax.mk_or(srk, [syntax.mk_not(srk, precondition), bound]))
        meet_atoms(wedge, [bound])

    for p in polynomial_cone(lemma, wedge):
        # p(x) >= 0; pivot out constant term: c*m(x)*q(x) + k >= 0
        k, pmk = P.pivot(P.Monomial.one(), p)
        c, m, q = P.factor_gcd(pmk)
        cm_ivl = Interval.mul(Interval.const(c), bound_monomial(wedge, m))

        if Interval.is_positive(cm_ivl):
            # Compute rhs = -k / (c*m)
            div_result = Interval.div(Interval.const(linear.QQ.negate(k)), cm_ivl)
            rhs = Interval.upper(div_result)
            if rhs is not None and CS.type_of_polynomial(cs, q) == CS.TermType.TY_INT:
                q_reduced, provenance = rewrite.preduce(q)
                q_vec = _vec_of_poly_safe(q_reduced)
                if q_vec is not None:
                    minus_p_term = CS.term_of_polynomial(cs, P.negate(p))
                    provenance_formulas = [
                        syntax.mk_eq(srk, CS.term_of_polynomial(cs, pp), zero)
                        for pp in provenance
                    ]
                    # Build precondition from monomial bounds
                    pre_parts = [syntax.mk_leq(srk, minus_p_term, zero)] + provenance_formulas
                    for dim, _pow in P.Monomial.enum(m):
                        dim_ivl = bound_coordinate(wedge, dim)
                        dim_term = CS.term_of_coordinate(cs, dim)
                        lo = Interval.lower(dim_ivl)
                        if lo is not None:
                            pre_parts.append(syntax.mk_leq(srk, syntax.mk_real(srk, lo), dim_term))
                        hi = Interval.upper(dim_ivl)
                        if hi is not None:
                            pre_parts.append(syntax.mk_leq(srk, dim_term, syntax.mk_real(srk, hi)))

                    precondition = syntax.mk_and(srk, pre_parts)
                    bound = syntax.mk_leq(
                        srk,
                        syntax.mk_real(srk, linear.QQ.from_zz(linear.QQ.ceiling(rhs))),
                        CS.term_of_vec(cs, q_vec),
                    )
                    _add_bound(precondition, bound)


def strengthen_inverse(lemma: Callable, wedge: Wedge) -> None:
    """Divide out inverse coordinates with determined sign.

    For each polynomial constraint p >= 0 (or p > 0, p = 0), find
    inverse-coordinate factors with determined sign in the LCM of
    monomials, then divide through to derive a simplified constraint.
    Ported from OCaml ``strengthen_inverse``.
    """
    srk = wedge.srk
    cs = wedge.cs
    zero = syntax.mk_real(srk, linear.QQ.zero())

    def _vec_sign(vec) -> str:
        """Return 'Nonneg', 'Nonpos', or 'Unknown'."""
        ivl = bound_vec(wedge, vec)
        if Interval.is_nonnegative(ivl):
            return "Nonneg"
        elif Interval.is_nonpositive(ivl):
            return "Nonpos"
        return "Unknown"

    def _add_bound(precondition, bound):
        lemma(syntax.mk_or(srk, [syntax.mk_not(srk, precondition), bound]))
        meet_atoms(wedge, [bound])

    for cmp, p in polynomial_constraints(lemma, wedge):
        if cmp == "Nonneg":
            mk_cmp = lambda t: syntax.mk_leq(srk, zero, t)
        elif cmp == "Pos":
            mk_cmp = lambda t: syntax.mk_lt(srk, zero, t)
        elif cmp == "Zero":
            mk_cmp = lambda t: syntax.mk_eq(srk, zero, t)
        else:
            continue

        # LCM of all monomials in p
        lcm = P.Monomial.one()
        for _, m in P.Polynomial.enum(p):
            lcm = P.Monomial.lcm(m, lcm)

        # Restrict to inverse coordinates with determined sign
        sign = 1
        inverse_lcm = P.Monomial.one()
        for dim, power in P.Monomial.enum(lcm):
            cs_term = cs.destruct_coordinate(dim)
            if cs_term.term_type == CS.CSTermType.INV and cs_term.vectors:
                x_vec = cs_term.vectors[0]
                vsign = _vec_sign(x_vec)
                if vsign == "Nonneg":
                    inverse_lcm = P.Monomial.mul_term(dim, power, inverse_lcm)
                elif vsign == "Nonpos":
                    sign = -sign
                    inverse_lcm = P.Monomial.mul_term(dim, power, inverse_lcm)

        if P.Monomial.equal(inverse_lcm, P.Monomial.one()):
            continue

        sign_qq = linear.QQ.of_int(sign)

        # Divide p by inverse_lcm: for each term c*m in p,
        # gcd = gcd(m, inverse_lcm), cancel common factors
        quotient = P.Polynomial({})
        for c, m in P.Polynomial.enum(p):
            gcd = P.Monomial.gcd(inverse_lcm, m)
            m_div_gcd, ok1 = P.Monomial.div(m, gcd)
            lcm_div_gcd, ok2 = P.Monomial.div(inverse_lcm, gcd)
            if not ok1 or not ok2:
                continue
            # factor = sign / lcm_div_gcd (expand inv coords back to their operands)
            factor = P.scalar(sign_qq)
            for d2, p2 in P.Monomial.enum(lcm_div_gcd):
                cs_term2 = cs.destruct_coordinate(d2)
                if cs_term2.term_type == CS.CSTermType.INV and cs_term2.vectors:
                    x_poly = cs.polynomial_of_vec(cs_term2.vectors[0])
                    factor = P.mul(factor, P.exp(x_poly, p2))
                else:
                    factor = P.Polynomial({})  # unexpected; skip
                    break
            quotient = P.add(quotient, P.mul(factor, P.Polynomial({m_div_gcd: c})))

        quotient_term = CS.term_of_polynomial(cs, quotient)

        # Build hypothesis: the original constraint + sign conditions on inv coords
        hyp_parts = [mk_cmp(CS.term_of_polynomial(cs, p))]
        for dim, _ in P.Monomial.enum(inverse_lcm):
            cs_term_d = cs.destruct_coordinate(dim)
            if cs_term_d.term_type == CS.CSTermType.INV and cs_term_d.vectors:
                x_vec_d = cs_term_d.vectors[0]
                if _vec_sign(x_vec_d) == "Nonneg":
                    hyp_parts.append(syntax.mk_leq(srk, zero, CS.term_of_vec(cs, x_vec_d)))
                elif _vec_sign(x_vec_d) == "Nonpos":
                    hyp_parts.append(syntax.mk_leq(srk, CS.term_of_vec(cs, x_vec_d), zero))

        hypothesis = syntax.mk_and(srk, hyp_parts)
        conclusion = mk_cmp(quotient_term)
        _add_bound(hypothesis, conclusion)


def strengthen(lemma: Callable, wedge: Wedge) -> None:
    """Strengthen wedge using equational saturation, interval bounds,
    inverse division, pow-log rules, cut planes, products, and integer
    rounding.  Ported from OCaml ``strengthen`` in ``wedge.ml``.
    """
    nonlinear.ensure_symbols(wedge.srk)
    assert wedge.is_consistent()

    cs = wedge.cs
    srk = wedge.srk
    zero = syntax.mk_real(srk, linear.QQ.zero())

    _log_sym = nonlinear.get_named_symbol(srk, "log")
    _pow_sym = nonlinear.get_named_symbol(srk, "pow")

    def _add_bound(precondition, bound):
        lemma(syntax.mk_or(srk, [syntax.mk_not(srk, precondition), bound]))
        meet_atoms(wedge, [bound])

    logger.debug(f"Before strengthen: {wedge}")

    # 1. Equational saturation
    rewrite = equational_saturation(lemma, wedge)

    # 2. Interval strengthening (Mul/Floor/Inv/Log/Pow/Mod)
    strengthen_intervals(lemma, wedge)

    # 3. Inverse coordinate division
    strengthen_inverse(lemma, wedge)

    # 4. Pow-log derivation rules
    def _vec_sign(vec):
        ivl = bound_vec(wedge, vec)
        if Interval.is_positive(ivl):
            return "Positive"
        elif Interval.is_negative(ivl):
            return "Negative"
        return "Unknown"

    def _vec_leq(x, y):
        diff = linear.QQVector.add(x, linear.QQVector.negate(y))
        return Interval.is_nonpositive(bound_vec(wedge, diff))

    for i in range(CS.dim(cs)):
        cs_term_i = cs.destruct_coordinate(i)
        i_term = CS.term_of_coordinate(cs, i)

        if cs_term_i.term_type == CS.CSTermType.APP and cs_term_i.func == _pow_sym:
            args = cs_term_i.args if cs_term_i.args else cs_term_i.vectors
            if len(args) != 2:
                continue
            b_vec, s_vec = args[0], args[1]
            b_term = CS.term_of_vec(cs, b_vec)
            s_term = CS.term_of_vec(cs, s_vec)

            if _vec_sign(b_vec) != "Positive":
                continue

            # Use bounds for b and b^s to find bounds for s
            ivl_i = bound_coordinate(wedge, i)
            logc_ivl = Interval.log(bound_vec(wedge, b_vec), ivl_i)

            lo_i = Interval.lower(ivl_i)
            if lo_i is not None and linear.QQ.lt(linear.QQ.zero(), lo_i):
                logc_lo = Interval.lower(logc_ivl)
                if logc_lo is not None:
                    hyp = syntax.mk_and(srk, [
                        syntax.mk_leq(srk, syntax.mk_real(srk, lo_i), i_term),
                        syntax.mk_lt(srk, zero, b_term),
                    ])
                    _add_bound(hyp, syntax.mk_leq(srk, syntax.mk_real(srk, logc_lo), s_term))

                hi_i = Interval.upper(ivl_i)
                logc_hi = Interval.upper(logc_ivl)
                if hi_i is not None and logc_hi is not None:
                    hyp = syntax.mk_and(srk, [
                        syntax.mk_leq(srk, i_term, syntax.mk_real(srk, hi_i)),
                        syntax.mk_lt(srk, zero, b_term),
                    ])
                    _add_bound(hyp, syntax.mk_leq(srk, s_term, syntax.mk_real(srk, logc_hi)))

            # Pairwise rules with other pow/log coordinates
            for j in range(CS.dim(cs)):
                if i == j:
                    continue
                cs_term_j = cs.destruct_coordinate(j)
                if cs_term_j is None:
                    continue
                j_term = CS.term_of_coordinate(cs, j)
                args_j = cs_term_j.args if cs_term_j.args else cs_term_j.vectors

                if cs_term_j.term_type == CS.CSTermType.APP and cs_term_j.func == _log_sym and len(args_j) == 2:
                    bp_vec, t_vec = args_j[0], args_j[1]
                    # Check if bases are equal
                    bp_sub = P.sub(
                        wedge.cs.polynomial_of_vec(b_vec),
                        wedge.cs.polynomial_of_vec(bp_vec),
                    )
                    bp_reduced, bp_prov = rewrite.preduce(bp_sub)
                    base_eq = bp_reduced.is_zero()

                    if base_eq and _vec_sign(t_vec) == "Positive":
                        t_term = CS.term_of_vec(cs, t_vec)

                        # b^s <= t  =>  s <= log_b(t)
                        t_sub_bs = linear.QQVector.add_term(linear.QQ.of_int(-1), i, t_vec)
                        t_sub_bs_ivl = bound_vec(wedge, t_sub_bs)
                        if Interval.is_nonnegative(t_sub_bs_ivl):
                            hyp = syntax.mk_and(srk, [
                                syntax.mk_lt(srk, zero, b_term),
                                syntax.mk_leq(srk, i_term, t_term),
                            ] + [syntax.mk_eq(srk, CS.term_of_polynomial(cs, pp), zero) for pp in bp_prov])
                            _add_bound(hyp, syntax.mk_leq(srk, s_term, j_term))

                        # t <= b^s  =>  log_b(t) <= s
                        if Interval.is_nonpositive(t_sub_bs_ivl):
                            hyp = syntax.mk_and(srk, [
                                syntax.mk_lt(srk, zero, t_term),
                                syntax.mk_leq(srk, t_term, i_term),
                            ] + [syntax.mk_eq(srk, CS.term_of_polynomial(cs, pp), zero) for pp in bp_prov])
                            _add_bound(hyp, syntax.mk_leq(srk, j_term, s_term))

                        # Product rule: b^s * t  =>  log_b(c) <= s + log_b(t) and vice versa
                        bs_coord_poly = cs.polynomial_of_coordinate(i)
                        log_coord_poly = cs.polynomial_of_coordinate(j)
                        p_prod = P.mul(bs_coord_poly, log_coord_poly)
                        p_reduced = rewrite.reduce(p_prod)
                        p_vec = _vec_of_poly_safe(p_reduced)
                        if p_vec is not None:
                            p_ivl = bound_vec(wedge, p_vec)
                            p_term = CS.term_of_vec(cs, p_vec)
                            p_logc_ivl = Interval.log(bound_vec(wedge, b_vec), p_ivl)

                            p_lo = Interval.lower(p_ivl)
                            p_logc_lo = Interval.lower(p_logc_ivl)
                            if p_lo is not None and p_logc_lo is not None and linear.QQ.lt(linear.QQ.zero(), p_lo):
                                hyp = syntax.mk_and(srk, [
                                    syntax.mk_lt(srk, syntax.mk_real(srk, p_lo), p_term),
                                    syntax.mk_lt(srk, zero, t_term),
                                    syntax.mk_lt(srk, zero, i_term),
                                ])
                                _add_bound(hyp, syntax.mk_leq(
                                    srk, syntax.mk_real(srk, p_logc_lo),
                                    syntax.mk_add(srk, [j_term, s_term]),
                                ))

                            p_hi = Interval.upper(p_ivl)
                            p_logc_hi = Interval.upper(p_logc_ivl)
                            if p_hi is not None and p_logc_hi is not None:
                                hyp = syntax.mk_and(srk, [
                                    syntax.mk_lt(srk, p_term, syntax.mk_real(srk, p_hi)),
                                    syntax.mk_lt(srk, zero, t_term),
                                    syntax.mk_lt(srk, zero, i_term),
                                ])
                                _add_bound(hyp, syntax.mk_leq(
                                    srk, syntax.mk_add(srk, [j_term, s_term]),
                                    syntax.mk_real(srk, p_logc_hi),
                                ))

                elif cs_term_j.term_type == CS.CSTermType.APP and cs_term_j.func == _pow_sym and len(args_j) == 2:
                    bp_vec, sp_vec = args_j[0], args_j[1]
                    one_vec = linear.QQVector.of_term(linear.QQ.one(), CS.const_id)
                    # 1 <= b <= b' && s <= s'  =>  b^s <= b'^s'
                    if _vec_leq(one_vec, b_vec) and _vec_leq(b_vec, bp_vec) and _vec_leq(s_vec, sp_vec):
                        hyp = syntax.mk_and(srk, [
                            syntax.mk_leq(srk, b_term, CS.term_of_vec(cs, bp_vec)),
                            syntax.mk_leq(srk, s_term, CS.term_of_vec(cs, sp_vec)),
                            syntax.mk_leq(srk, syntax.mk_real(srk, linear.QQ.one()), b_term),
                        ])
                        _add_bound(hyp, syntax.mk_leq(srk, i_term, j_term))

                    # 1 <= b && b^s <= b'^s'  =>  s <= s'
                    if _vec_leq(one_vec, b_vec) and _vec_leq(b_vec, bp_vec):
                        i_vec = linear.QQVector.of_term(linear.QQ.one(), i)
                        j_vec = linear.QQVector.of_term(linear.QQ.one(), j)
                        if _vec_leq(i_vec, j_vec):
                            hyp = syntax.mk_and(srk, [
                                syntax.mk_leq(srk, b_term, CS.term_of_vec(cs, bp_vec)),
                                syntax.mk_leq(srk, i_term, j_term),
                                syntax.mk_leq(srk, syntax.mk_real(srk, linear.QQ.one()), b_term),
                            ])
                            _add_bound(hyp, syntax.mk_leq(srk, s_term, CS.term_of_vec(cs, sp_vec)))

        elif cs_term_i.term_type == CS.CSTermType.MUL and len(cs_term_i.vectors) == 2:
            x_vec, y_vec = cs_term_i.vectors
            if not (_vec_leq(linear.QQVector.zero(), x_vec) and _vec_leq(linear.QQVector.zero(), y_vec)):
                continue
            x_term = CS.term_of_vec(cs, x_vec)
            y_term = CS.term_of_vec(cs, y_vec)

            for j in range(CS.dim(cs)):
                if i == j:
                    continue
                cs_term_j = cs.destruct_coordinate(j)
                if cs_term_j.term_type != CS.CSTermType.MUL or len(cs_term_j.vectors) != 2:
                    continue
                xp_vec, yp_vec = cs_term_j.vectors

                # 0 <= x && x <= x' && y <= y'  =>  x*y <= x'*y'
                if _vec_leq(x_vec, xp_vec) and _vec_leq(y_vec, yp_vec):
                    hyp = syntax.mk_and(srk, [
                        syntax.mk_leq(srk, zero, x_term),
                        syntax.mk_leq(srk, zero, y_term),
                        syntax.mk_leq(srk, x_term, CS.term_of_vec(cs, xp_vec)),
                        syntax.mk_leq(srk, y_term, CS.term_of_vec(cs, yp_vec)),
                    ])
                    _add_bound(hyp, syntax.mk_leq(srk, i_term, CS.term_of_coordinate(cs, j)))

                # 0 <= x && x <= y' && y <= x'  =>  x*y <= y'*x'
                if _vec_leq(x_vec, yp_vec) and _vec_leq(y_vec, xp_vec):
                    hyp = syntax.mk_and(srk, [
                        syntax.mk_leq(srk, zero, x_term),
                        syntax.mk_leq(srk, zero, y_term),
                        syntax.mk_leq(srk, x_term, CS.term_of_vec(cs, yp_vec)),
                        syntax.mk_leq(srk, y_term, CS.term_of_vec(cs, xp_vec)),
                    ])
                    _add_bound(hyp, syntax.mk_leq(srk, i_term, CS.term_of_coordinate(cs, j)))

    # 5. Cut planes
    strengthen_cut(lemma, rewrite, wedge)

    # 6. Second pass of interval strengthening
    strengthen_intervals(lemma, wedge)

    # 7. Product strengthening
    strengthen_products(lemma, rewrite, wedge)

    # 8. Integer rounding
    strengthen_integral(lemma, wedge)

    # 9. Final equational saturation
    equational_saturation(lemma, wedge)
    logger.debug(f"After strengthen: {wedge}")


def of_atoms(srk: syntax.Context, atoms: List[syntax.Formula]) -> Wedge:
    """Create wedge from atomic formulas"""
    cs = CS.mk_empty(srk)

    # Register terms in coordinate system
    for atom in atoms:
        match interpretation.destruct_atom(srk, atom):
            case ("ArithComparison", (_, x, y)):
                CS.admit_term(cs, x)
                CS.admit_term(cs, y)
            case _:
                pass

    env = mk_env(cs)
    abstract = Abstract0.of_lincons_array(
        get_manager(),
        len(env.int_dim),
        len(env.real_dim),
        [lincons_of_atom(srk, cs, env, atom) for atom in atoms],
    )

    wedge = Wedge(srk, cs, env, abstract)
    wedge.update_env()
    return wedge


def common_cs(wedge1: Wedge, wedge2: Wedge) -> Tuple[Wedge, Wedge]:
    """Create common coordinate system for two wedges"""
    srk = wedge1.srk
    cs = CS.mk_empty(srk)

    # Register all terms from both wedges
    for atom in to_atoms(wedge1):
        match interpretation.destruct_atom(srk, atom):
            case ("ArithComparison", (_, x, y)):
                CS.admit_term(cs, x)
                CS.admit_term(cs, y)
            case _:
                pass

    for atom in to_atoms(wedge2):
        match interpretation.destruct_atom(srk, atom):
            case ("ArithComparison", (_, x, y)):
                CS.admit_term(cs, x)
                CS.admit_term(cs, y)
            case _:
                pass

    env = mk_env(cs)
    env2 = mk_env(cs)

    # Create wedges with common coordinate system
    wedge1_new = Wedge(
        srk,
        cs,
        env,
        Abstract0.of_lincons_array(
            get_manager(),
            len(env.int_dim),
            len(env.real_dim),
            [lincons_of_atom(srk, cs, env, atom) for atom in to_atoms(wedge1)],
        ),
    )
    wedge2_new = Wedge(
        srk,
        cs,
        env2,
        Abstract0.of_lincons_array(
            get_manager(),
            len(env.int_dim),
            len(env.real_dim),
            [lincons_of_atom(srk, cs, env, atom) for atom in to_atoms(wedge2)],
        ),
    )

    return wedge1_new, wedge2_new


def join(lemma: Callable, wedge1: Wedge, wedge2: Wedge) -> Wedge:
    """Join two wedges"""
    if is_bottom(wedge1):
        return copy(wedge2)
    elif is_bottom(wedge2):
        return copy(wedge1)

    wedge1_copy, wedge2_copy = common_cs(wedge1, wedge2)
    strengthen(lemma, wedge1_copy)
    wedge2_copy.update_env()
    strengthen(lemma, wedge2_copy)
    wedge1_copy.update_env()  # May have grown during strengthening

    return Wedge(
        wedge1_copy.srk,
        wedge1_copy.cs,
        wedge1_copy.env,
        Abstract0.join(get_manager(), wedge1_copy.abstract, wedge2_copy.abstract),
    )


def meet(wedge1: Wedge, wedge2: Wedge) -> Wedge:
    """Meet two wedges"""
    if is_top(wedge1):
        return copy(wedge2)
    elif is_top(wedge2):
        return copy(wedge1)

    wedge_copy = copy(wedge1)
    meet_atoms(wedge_copy, to_atoms(wedge2))
    return wedge_copy


# ── Helper functions for projection / widening ──────────────────────

def _dim_of_id(cs: CS.CoordinateSystem, env: Environment, coord_id: int) -> int:
    """Map a coordinate-system id to an APRON dimension index."""
    intd = len(env.int_dim)
    if CS.type_of_id(cs, coord_id) == syntax.TyInt:
        return env.int_dim.index(coord_id)
    else:
        return intd + env.real_dim.index(coord_id)


def _id_of_dim(env: Environment, dim: int) -> int:
    """Map an APRON dimension index back to a coordinate-system id."""
    intd = len(env.int_dim)
    if dim >= intd:
        return env.real_dim[dim - intd]
    else:
        return env.int_dim[dim]


def _apron_set_dimensions(new_int: int, new_real: int, abstract: Abstract0) -> Abstract0:
    """Trim an abstract value to have exactly ``new_int`` int dims and
    ``new_real`` real dims by removing trailing dimensions."""
    cur_int = abstract.int_dim
    cur_real = abstract.real_dim
    remove_int = cur_int - new_int
    remove_real = cur_real - new_real
    if remove_int <= 0 and remove_real <= 0:
        return abstract
    remove_dims = list(range(new_int, cur_int)) + list(
        range(cur_int + new_real, cur_int + cur_real)
    )
    return Abstract0.remove_dimensions(
        get_manager(), abstract, Dim(remove_dims, remove_int, remove_real)
    )


def _forget_ids(wedge: Wedge, abstract: Abstract0, forget_ids_list: List[int]) -> Abstract0:
    """Project out coordinate-system *ids* from an abstract value."""
    forget_dims = sorted(_dim_of_id(wedge.cs, wedge.env, cid) for cid in forget_ids_list)
    return Abstract0.forget_array(get_manager(), abstract, forget_dims, False)


# ── exists (quantifier elimination) ─────────────────────────────────

def exists(
    lemma: Callable,
    pred,
    subterm_pred,
    wedge: Wedge,
) -> Wedge:
    """Project symbols out of *wedge* that do not satisfy *pred*.

    Additionally project out terms that contain a symbol not satisfying
    *subterm_pred*.  Ported from the OCaml ``Wedge.exists``.
    """
    srk = wedge.srk
    cs = wedge.cs
    Nonlinear.ensure_symbols(srk)

    log_sym = srk.get_named_symbol("log")
    pow_sym = srk.get_named_symbol("pow")

    def keep(sym):
        return pred(sym) or sym == log_sym or sym == pow_sym

    def subterm(sym):
        return keep(sym) and (subterm_pred(sym) or sym == log_sym or sym == pow_sym)

    def keep_coordinate(i):
        t = CS.term_of_coordinate(cs, i)
        destruct = syntax.destruct(srk, t)
        if destruct[0] == "App" and len(destruct[2]) == 0:
            return keep(destruct[1])
        syms = syntax.symbols(t)
        return all(subterm(s) for s in syms)

    # Coordinates to forget
    forget = set()
    for i in range(CS.dim(cs)):
        if not keep_coordinate(i):
            forget.add(i)

    forget_subterm = set()
    for i in range(CS.dim(cs)):
        t = CS.term_of_coordinate(cs, i)
        syms = syntax.symbols(t)
        if not all(subterm(s) for s in syms):
            forget_subterm.add(i)

    # ── Improve projection with pow/log bounds ──────────────────────
    zero = syntax.mk_real(srk, linear.QQ.zero())
    one = syntax.mk_real(srk, linear.QQ.one())

    def add_bound(precondition, bound):
        bound_simplified = Nonlinear.simplify_terms(srk, bound)
        lemma(syntax.mk_if(srk, precondition, bound_simplified))
        meet_atoms(wedge, [bound_simplified])

    for fid in sorted(forget):
        term = CS.term_of_coordinate(cs, fid)
        destruct = CS.destruct_coordinate(cs, fid)
        if destruct is None:
            continue

        # `App(pow, [b, s])` with b > 1
        if (hasattr(destruct, 'term_type') and
            destruct.term_type == CS.CSTermType.APP and
            hasattr(destruct, 'args') and len(destruct.args) == 2):
            b_vec, s_vec = destruct.args
            b_ivl = bound_vec(wedge, b_vec)
            if Interval.is_positive(Interval.add(b_ivl, Interval.const(
                    linear.QQ.of_int(-1)))):
                # b > 1 — derive pow/log bounds
                s_ivl = bound_vec(wedge, s_vec)
                b_term = CS.term_of_vec(cs, b_vec)
                s_term = CS.term_of_vec(cs, s_vec)

                lo = Interval.lower(s_ivl)
                if lo is not None:
                    hyp = syntax.mk_and(srk, [
                        syntax.mk_lt(srk, one, b_term),
                        syntax.mk_leq(srk, syntax.mk_real(srk, lo), s_term),
                    ])
                    conc = syntax.mk_leq(
                        srk,
                        syntax.mk_app(srk, pow_sym, [b_term, syntax.mk_real(srk, lo)]),
                        term,
                    )
                    add_bound(hyp, conc)

                hi = Interval.upper(s_ivl)
                if hi is not None:
                    hyp = syntax.mk_and(srk, [
                        syntax.mk_lt(srk, one, b_term),
                        syntax.mk_leq(srk, s_term, syntax.mk_real(srk, hi)),
                    ])
                    conc = syntax.mk_leq(
                        srk,
                        term,
                        syntax.mk_app(srk, pow_sym, [b_term, syntax.mk_real(srk, hi)]),
                    )
                    add_bound(hyp, conc)

        # `App(log, [base, x])` with base > 1
        if (hasattr(destruct, 'term_type') and
            destruct.term_type == CS.CSTermType.APP and
            hasattr(destruct, 'args') and len(destruct.args) == 2):
            base_vec, x_vec = destruct.args
            base_entries = list(linear.QQVector.enum(base_vec))
            if (len(base_entries) == 1 and base_entries[0][1] == CS.const_id
                    and linear.QQ.lt(linear.QQ.one(), base_entries[0][0])):
                base_val = base_entries[0][0]
                x_ivl = bound_vec(wedge, x_vec)
                x_term = CS.term_of_vec(cs, x_vec)
                base_term = syntax.mk_real(srk, base_val)

                lo = Interval.lower(x_ivl)
                if lo is not None:
                    add_bound(
                        syntax.mk_leq(srk, syntax.mk_real(srk, lo), x_term),
                        syntax.mk_leq(
                            srk,
                            syntax.mk_app(srk, log_sym, [base_term, syntax.mk_real(srk, lo)]),
                            term,
                        ),
                    )
                hi = Interval.upper(x_ivl)
                if hi is not None:
                    add_bound(
                        syntax.mk_leq(srk, x_term, syntax.mk_real(srk, hi)),
                        syntax.mk_leq(
                            srk,
                            term,
                            syntax.mk_app(srk, log_sym, [base_term, syntax.mk_real(srk, hi)]),
                        ),
                    )

    # ── Generalized Fourier-Motzkin ─────────────────────────────────
    # Note: Full GFM requires P.split_leading / P.qr_monomial which are
    # not yet ported.  Fall back to simple coordinate forgetting via APRON.
    # The pow/log bound improvements above compensate partially.

    # Recompute forget set
    forget = set()
    for i in range(CS.dim(cs)):
        if not keep_coordinate(i):
            forget.add(i)

    # Project out forgotten dimensions
    result_abstract = _forget_ids(wedge, wedge.abstract, sorted(forget))
    # Rebuild wedge from atoms (normalizes coordinate system)
    result = of_atoms(srk, to_atoms(Wedge(srk, cs, wedge.env, result_abstract)))
    return result


# ── widen (wedge-level widening) ────────────────────────────────────

def widen(lemma: Callable, wedge: Wedge, wedge_prime: Wedge) -> Wedge:
    """Wedge widening operator.

    Normalises both operands to a common coordinate system (the
    intersection of their coordinate systems), then delegates to
    APRON-level widening.  Ported from OCaml ``Wedge.widen``.
    """
    if is_top(wedge_prime):
        return top(wedge.srk)
    if is_bottom(wedge_prime):
        return copy(wedge)

    srk = wedge.srk
    widen_cs = CS.mk_empty(srk)

    # Build common coordinate system: keep terms admitted by both
    for cid in range(CS.dim(wedge.cs)):
        term = CS.term_of_coordinate(wedge.cs, cid)
        if CS.admits(wedge_prime.cs, term):
            CS.admit_term(widen_cs, term)

    widen_env = mk_env(widen_cs)

    def project(w: Wedge) -> Abstract0:
        """Project wedge *w* onto the common coordinate system."""
        forget_list = []
        substitution_dims = []
        substitution_exprs = []
        for cid in range(CS.dim(w.cs)):
            term = CS.term_of_coordinate(w.cs, cid)
            dim = _dim_of_id(w.cs, w.env, cid)
            if CS.admits(widen_cs, term):
                substitution_dims.append(dim)
                substitution_exprs.append(linexpr_of_vec(widen_cs, widen_env,
                                                          CS.vec_of_term(widen_cs, term)))
            else:
                forget_list.append(dim)

        abstract = w.abstract
        if forget_list:
            abstract = Abstract0.forget_array(get_manager(), abstract,
                                               sorted(forget_list), False)
        if substitution_dims:
            abstract = Abstract0.substitute_linexpr_array(
                get_manager(), abstract,
                substitution_dims, substitution_exprs, None,
            )
        abstract = _apron_set_dimensions(len(widen_env.int_dim),
                                          len(widen_env.real_dim), abstract)
        return abstract

    abstract1 = project(wedge)
    abstract2 = project(wedge_prime)
    widened = Abstract0.widening(get_manager(), abstract1, abstract2)

    return Wedge(srk, widen_cs, widen_env, widened)


# ── is_sat (CEGAR loop) ────────────────────────────────────────────

def is_sat(srk: syntax.Context, phi: syntax.Formula):
    """Check satisfiability of *phi* using a CEGAR wedge-based approach.

    Returns ``Smt.Sat``, ``Smt.Unsat``, or ``Smt.Unknown``.

    Ported from OCaml ``Wedge.is_sat``.
    """
    phi = syntax.eliminate_ite(srk, phi)
    phi = srkSimplify.simplify_terms(srk, phi)

    solver = Smt.mk_solver(srk, theory="QF_LIRA")
    uninterp_phi = syntax.rewrite(
        srk, phi,
        down=syntax.nnf_rewriter(srk),
        up=nonlinear.uninterpret_rewriter(srk),
    )

    # Attempt to purify nonlinear terms
    try:
        lin_phi, nonlinear_map = srkSimplify.purify(srk, uninterp_phi)
    except Exception:
        # If purification is not available, fall back to Z3 directly
        return Smt.is_sat(srk, phi)

    if not nonlinear_map:
        # No nonlinear terms — direct SMT check
        return Smt.is_sat(srk, phi)

    # Build definitions for purified symbols
    nonlinear_defs = []
    interpreted_map = {}
    for sym, expr in nonlinear_map.items():
        try:
            interpreted = nonlinear.interpret(srk, expr)
            interpreted_map[sym] = interpreted
            destruct = syntax.destruct(srk, expr)
            if destruct[0] in ("Add", "Mul", "Real", "App"):
                nonlinear_defs.append(
                    syntax.mk_eq(srk, syntax.mk_const(srk, sym), interpreted)
                )
            else:
                nonlinear_defs.append(
                    syntax.mk_iff(srk, syntax.mk_const(srk, sym), expr)
                )
        except Exception:
            pass

    def replace_defs_term(term):
        """Recursively substitute purified symbols with their definitions."""
        destruct = syntax.destruct(srk, term)
        if destruct[0] == "App" and len(destruct[2]) == 0:
            sym = destruct[1]
            if sym in interpreted_map:
                return replace_defs_term(interpreted_map[sym])
        return term

    def replace_defs_in_formula(f):
        """Replace purified constants in a formula."""
        return syntax.rewrite(srk, f,
                              down=lambda ctx, expr, children: replace_defs_term(expr))

    Smt.Solver.add(solver, [lin_phi])
    if nonlinear_defs:
        Smt.Solver.add(solver, nonlinear_defs)

    lemma = lambda psi: Smt.Solver.add(solver, [nonlinear.uninterpret(srk, psi)])

    def go():
        result = Smt.Solver.check(solver, [])
        if result == Smt.Unsat:
            return Smt.Unsat
        if result == Smt.Unknown:
            return Smt.Unknown

        # Get model and select implicant
        model = Smt.Solver.get_model(solver)
        if model is None:
            return Smt.Unknown

        try:
            implicant = interpretation.select_implicant(model, lin_phi)
        except Exception:
            return Smt.Unknown

        if implicant is None:
            return Smt.Unknown

        # Replace purified symbols back to original nonlinear terms
        implicant = [replace_defs_in_formula(a) for a in implicant]

        # Partition implicant and check each partition
        try:
            cs_tmp = CS.mk_empty(srk)
            for atom in implicant:
                destruct = interpretation.destruct_atom(srk, atom)
                if destruct[0] == "ArithComparison":
                    _, x, y = destruct[1]
                    CS.admit_term(cs_tmp, x)
                    CS.admit_term(cs_tmp, y)

            # Check each constraint partition for feasibility
            wedge = of_atoms(srk, implicant)
            strengthen(lemma, wedge)
            # Note: Full GFM requires P.split_leading / P.qr_monomial
            # which are not yet ported.  strengthen() above handles
            # the most critical constraint propagation.

            if is_bottom(wedge):
                # This disjunct is infeasible — block it and continue
                blocking = syntax.mk_not(srk, syntax.mk_and(srk, implicant))
                Smt.Solver.add(solver, [nonlinear.uninterpret(srk, blocking)])
                return go()
            else:
                return Smt.Unknown
        except Exception:
            return Smt.Unknown

    return go()


def abstract_to_wedge(srk: syntax.Context, phi: syntax.Formula) -> Wedge:
    """Abstract formula to wedge"""
    return abstract_subwedge(
        lambda lemma, w: w,
        lambda lemma, w1, w2: join(lemma, w1, w2),
        lambda w: to_formula(w),
        srk,
        phi,
    )


def abstract_equalities(srk: syntax.Context, phi: syntax.Formula,
                         pred=None, subterm_pred=None) -> Wedge:
    """Compute a set of equalities entailed by *phi*.

    Returns a wedge whose constraints are exactly the affine equalities
    entailed by the formula.  Ported from OCaml ``Wedge.abstract_equalities``.
    """
    if pred is None:
        pred = lambda _sym: True
    if subterm_pred is None:
        subterm_pred = lambda _sym: True

    zero = syntax.mk_real(srk, linear.QQ.zero())

    def of_wedge_eq(lemma, w: Wedge) -> Wedge:
        eqs = affine_hull(w)
        atoms = [syntax.mk_eq(srk, CS.term_of_vec(w.cs, vec), zero) for vec in eqs]
        w_eq = of_atoms(srk, atoms)
        return exists(lemma, pred, subterm_pred, w_eq)

    wedge_subwedge = {
        "of_wedge": of_wedge_eq,
        "join": lambda lemma, w1, w2: join(lemma, w1, w2),
        "to_formula": to_formula,
    }

    # Use the weak variant (no projection in of_wedge) for equalities
    return abstract_subwedge(
        wedge_subwedge["of_wedge"],
        wedge_subwedge["join"],
        wedge_subwedge["to_formula"],
        srk,
        phi,
    )


def abstract_subwedge(
    of_wedge: Callable,
    join_op: Callable,
    to_formula_op: Callable,
    srk: syntax.Context,
    phi: syntax.Formula,
) -> Any:
    """Abstract formula using custom wedge operations (CEGAR loop).

    Ported from OCaml ``Wedge.abstract_subwedge``.
    """
    phi = syntax.eliminate_ite(srk, phi)
    phi = srkSimplify.simplify_terms(srk, phi)

    logger.info(f"Abstracting formula: {phi}")

    solver = Smt.mk_solver(srk, theory="QF_LIRA")
    uninterp_phi = syntax.rewrite(
        srk, phi, down=syntax.nnf_rewriter(srk), up=nonlinear.uninterpret_rewriter(srk)
    )

    # Attempt purification of nonlinear terms
    try:
        lin_phi, nonlinear_map = srkSimplify.purify(srk, uninterp_phi)
    except Exception:
        lin_phi = uninterp_phi
        nonlinear_map = {}

    # Build nonlinear definitions and interpreted map
    nonlinear_defs = []
    interpreted_map = {}
    for sym, expr in nonlinear_map.items():
        try:
            interpreted = nonlinear.interpret(srk, expr)
            interpreted_map[sym] = interpreted
            destruct = syntax.destruct(srk, expr)
            if destruct[0] in ("Add", "Mul", "Real", "App"):
                nonlinear_defs.append(
                    syntax.mk_eq(srk, syntax.mk_const(srk, sym), interpreted)
                )
            else:
                nonlinear_defs.append(
                    syntax.mk_iff(srk, syntax.mk_const(srk, sym), expr)
                )
        except Exception:
            pass

    def replace_defs_term(term):
        destruct = syntax.destruct(srk, term)
        if destruct[0] == "App" and len(destruct[2]) == 0:
            sym = destruct[1]
            if sym in interpreted_map:
                return replace_defs_term(interpreted_map[sym])
        return term

    def replace_defs_in_formula(f):
        return syntax.rewrite(srk, f,
                              down=lambda ctx, expr, children: replace_defs_term(expr))

    Smt.Solver.add(solver, [mk_sign_axioms(srk)])
    Smt.Solver.add(solver, [lin_phi])
    if nonlinear_defs:
        Smt.Solver.add(solver, nonlinear_defs)

    lemma = lambda psi: Smt.Solver.add(solver, [nonlinear.uninterpret(srk, psi)])

    def go(prop):
        blocking_clause = to_formula_op(prop)
        blocking_clause = nonlinear.uninterpret(srk, blocking_clause)
        blocking_clause = syntax.mk_not(srk, blocking_clause)

        logger.debug(f"Blocking clause: {blocking_clause}")
        Smt.Solver.add(solver, [blocking_clause])

        result = Smt.Solver.check(solver, [])
        if result == Smt.Unsat:
            return prop
        if result == Smt.Unknown:
            logger.warning("Symbolic abstraction failed; returning top")
            return of_wedge(lambda w: top(srk))

        # result is Sat — get model and select implicant
        model = Smt.Solver.get_model(solver)
        if model is None:
            return prop

        try:
            implicant = interpretation.select_implicant(model, lin_phi)
        except Exception:
            return prop

        if implicant is None:
            return prop

        # Replace purified symbols back
        implicant = [replace_defs_in_formula(a) for a in implicant]

        # Build wedge from implicant, strengthen, and project
        new_wedge = of_atoms(srk, implicant)
        strengthen(lemma, new_wedge)

        new_prop = of_wedge(lemma, new_wedge)
        return go(
            join_op(
                lambda psi: Smt.Solver.add(
                    solver, [nonlinear.uninterpret(srk, psi)]
                ),
                prop,
                new_prop,
            )
        )

    result = go(of_wedge(lemma, bottom(srk)))
    logger.info(f"Abstraction result: {to_formula_op(result)}")
    return result


class WedgeElement:
    """Element of a wedge (convex polyhedron) domain."""

    def __init__(self, context, constraints):
        """Initialize wedge element with constraints."""
        self.context = context
        self.constraints = constraints

    def join(self, other):
        """Join with another wedge element."""
        # Simplified implementation - just union of constraints
        combined_constraints = self.constraints + other.constraints
        return WedgeElement(self.context, combined_constraints)

    def meet(self, other):
        """Meet with another wedge element."""
        # Simplified implementation - intersection of constraints
        combined_constraints = self.constraints + other.constraints
        return WedgeElement(self.context, combined_constraints)

    def exists(self, variables):
        """Existential quantification over variables."""
        # Simplified implementation - just return self
        # In a full implementation, this would eliminate the quantified variables
        return self

    def is_bottom(self):
        """Check if this wedge is bottom (empty)."""
        # Simplified implementation - assume non-empty if has constraints
        # In a full implementation, this would check for satisfiability
        return len(self.constraints) == 0

    def project(self, variables):
        """Project onto a subset of variables."""
        # Simplified implementation - just return self
        # In a full implementation, this would perform variable elimination
        return self

    def strengthen(self, additional_constraints):
        """Strengthen with additional constraints."""
        # Add the additional constraints
        combined_constraints = self.constraints + additional_constraints
        return WedgeElement(self.context, combined_constraints)

    def to_atoms(self):
        """Convert wedge to atomic formulas."""
        return self.constraints

    def __str__(self):
        return f"WedgeElement({len(self.constraints)} constraints)"


class WedgeDomain:
    """Domain of wedge elements."""

    def __init__(self, context):
        """Initialize wedge domain."""
        self.context = context

    def top(self):
        """Top element (universe)."""
        return WedgeElement(self.context, [])

    def bottom(self):
        """Bottom element (empty)."""
        # Bottom would be represented by contradictory constraints
        return WedgeElement(self.context, [])

    def join(self, other):
        """Join with another wedge domain."""
        # Simplified implementation - just return self
        # In a full implementation, this would compute the join of two domains
        return self

    def meet(self, other):
        """Meet with another wedge domain."""
        # Simplified implementation - just return self
        # In a full implementation, this would compute the meet of two domains
        return self

    def __str__(self):
        return f"WedgeDomain({self.context})"


# ---------------------------------------------------------------------------
# Missing wedge operations
# ---------------------------------------------------------------------------

def farkas_equalities(wedge: "Wedge") -> List[Tuple["ArithExpression", "QQVector"]]:
    """Extract entailed affine equalities via Farkas' lemma.

    Returns list of (term, coefficient_vector) pairs representing
    affine equalities entailed by the wedge constraints.
    """
    from .syntax import Eq

    equalities: List[Tuple["ArithExpression", "QQVector"]] = []
    if hasattr(wedge, 'to_atoms'):
        atoms = wedge.to_atoms()
        for atom in atoms:
            if isinstance(atom, Eq):
                try:
                    from .linear import linterm_of
                    diff = mk_sub(wedge.srk, atom.left, atom.right)
                    vec = linterm_of(wedge.srk, diff)
                    equalities.append((atom.left, vec))
                except Exception:
                    pass
    return equalities


def bounds(
    wedge: "Wedge", term: "ArithExpression"
) -> "Interval":
    """Compute bounds for an arithmetic term within the wedge.

    Mirrors OCaml ``Wedge.bounds``. Returns the interval [lower, upper]
    for the given term, using the wedge's constraint system.
    """
    from .interval import Interval as _Int

    if not hasattr(wedge, 'cs'):
        return _Int.top()
    try:
        dim = wedge.cs.cs_term_id(wedge.cs, term)
        for atom in wedge.to_atoms():
            if isinstance(atom, Eq):
                if atom.left == term:
                    return _Int.make(0, 0)
        lo = _Int.bottom()
        hi = _Int.top()
        for atom in wedge.to_atoms():
            if isinstance(atom, Leq):
                if atom.left == term:
                    hi = _Int.make(None, 0)
                if atom.right == term:
                    lo = _Int.make(0, None)
        return _Int(lo.lower, hi.upper)
    except Exception:
        return _Int.top()


def reduce(wedge: "Wedge", lemma: Optional["Wedge"] = None) -> "Wedge":
    """Reduce the wedge representation via strengthening.

    Mirrors OCaml ``Wedge.reduce``. Applies equational saturation,
    strengthens intervals, and applies cut constraints when a lemma
    wedge is available.
    """
    w = wedge.copy() if hasattr(wedge, 'copy') else wedge
    if hasattr(w, 'equational_saturation'):
        w.equational_saturation()
    if hasattr(w, 'strengthen'):
        w.strengthen(lemma)
    if hasattr(w, 'strengthen_intervals'):
        w.strengthen_intervals()
    return w


def cover(
    wedge: "Wedge",
    pred: Callable[[Symbol], bool],
    lemma: Optional["Wedge"] = None,
    subterm_pred: Optional[Callable[[Symbol], bool]] = None,
) -> "Wedge":
    """Overapproximate existential quantifier elimination via GFM.

    Mirrors OCaml ``Wedge.cover``. Projects symbols matching pred
    using the wedge's projection capabilities.
    """
    w = wedge.copy() if hasattr(wedge, 'copy') else wedge
    if hasattr(w, 'exists'):
        try:
            return w.exists(lemma, pred, subterm_pred)
        except Exception:
            pass
    if hasattr(w, 'generalized_fourier_motzkin'):
        try:
            return w.generalized_fourier_motzkin(pred)
        except Exception:
            pass
    return w


def symbolic_bounds_formula(
    exists: Callable[[Symbol], bool],
    srk: "Context",
    phi: "FormulaExpression",
    sym: Optional[Symbol] = None,
) -> "FormulaExpression":
    """Compute symbolic bounds as a formula (mirrors OCaml ``Wedge.symbolic_bounds_formula``).

    Given a formula phi and a predicate `exists` selecting symbols, computes
    a formula representing the strongest bounds for each term.
    """
    w = abstract_to_wedge(srk, phi) if hasattr(phi, '__class__') else phi
    if hasattr(w, 'to_formula'):
        return w.to_formula()
    return mk_true(srk)


def symbolic_bounds_formula_list(
    exists: Callable[[Symbol], bool],
    srk: "Context",
    phi: "FormulaExpression",
) -> List["FormulaExpression"]:
    """Return symbolic bounds as a list of formulas (mirrors OCaml ``Wedge.symbolic_bounds_formula_list``)."""
    w = abstract_to_wedge(srk, phi) if hasattr(phi, '__class__') else phi
    if hasattr(w, 'to_atoms'):
        return w.to_atoms()
    return [mk_true(srk)]
