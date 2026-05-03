"""
Approximate transitive closure computation using abstract interpretation.

This module implements algorithms for computing approximate transitive closures
of transition relations, which is useful for program verification and analysis.

Based on src/iteration.ml from the OCaml implementation.
"""

from __future__ import annotations
from typing import (
    Dict,
    List,
    Set,
    Tuple,
    Optional,
    Union,
    Any,
    Protocol,
    TypeVar,
    Generic,
    Callable,
)
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from fractions import Fraction
import logging

from aria.srk.syntax import (
    Context,
    Symbol,
    Type,
    Expression,
    FormulaExpression,
    ArithExpression,
    mk_true,
    mk_false,
    mk_and,
    mk_or,
    mk_not,
    mk_eq,
    mk_leq,
    mk_lt,
    mk_real,
    mk_const,
    mk_var,
    mk_symbol,
    mk_add,
    mk_mul,
    mk_sub,
    mk_if,
    mk_iff,
    mk_int,
    mk_one,
    symbols as get_symbols,
    substitute,
    substitute_const,
    substitute_map,
    rewrite,
    nnf_rewriter,
    free_vars,
)
from .qQ import QQ
from .linear import QQVector, QQMatrix
from . import transitionFormula as TF
from . import smt as Smt

logger = logging.getLogger(__name__)

T = TypeVar("T")
U = TypeVar("U")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _ZZ:
    """Minimal integer arithmetic helpers (mirrors OCaml zZ.ml)."""

    @staticmethod
    def one() -> int:
        return 1

    @staticmethod
    def lcm(a: int, b: int) -> int:
        from math import gcd
        return abs(a * b) // gcd(abs(a), abs(b)) if a and b else 0


ZZ = _ZZ()


def _term_of_vec_shifted(
    srk: Context,
    pre_symbols: List[ArithExpression],
    vec: QQVector,
) -> ArithExpression:
    """Convert a QQVector to an arithmetic term.

    Dimensions 1..n map to pre_symbols[0..n-1] (dimension shift by 1
    to match the OCaml convention where dim 0 is the constant term).
    """
    terms: List[ArithExpression] = []
    for coeff, dim in QQVector.entries(vec):
        if dim == 0:
            terms.append(mk_real(srk, coeff))
        elif 1 <= dim <= len(pre_symbols):
            terms.append(mk_mul(srk, [mk_real(srk, coeff), pre_symbols[dim - 1]]))
    if not terms:
        return mk_real(srk, QQ.zero())
    return mk_add(srk, terms)


class PreDomain(Protocol[T]):
    """Protocol for pre-domains used in iteration."""

    def abstract(self, context: Context, transition_formula: Any) -> T:
        """Abstract a transition formula to this domain."""
        ...

    def exp(
        self,
        context: Context,
        symbols: List[Tuple[Symbol, Symbol]],
        loop_counter: ArithExpression,
        domain_element: T,
    ) -> FormulaExpression:
        """Compute exponential expression (transitive closure) in this domain."""
        ...

    def pp(
        self,
        context: Context,
        symbols: List[Tuple[Symbol, Symbol]],
        formatter: Any,
        domain_element: T,
    ) -> None:
        """Pretty print domain element."""
        ...


class PreDomainIter(PreDomain[T]):
    """Pre-domain with iteration operations (join, widen, equal)."""

    def join(
        self, context: Context, symbols: List[Tuple[Symbol, Symbol]], elem1: T, elem2: T
    ) -> T:
        """Join two domain elements (least upper bound)."""
        ...

    def widen(
        self, context: Context, symbols: List[Tuple[Symbol, Symbol]], elem1: T, elem2: T
    ) -> T:
        """Widen two domain elements (acceleration)."""
        ...

    def equal(
        self, context: Context, symbols: List[Tuple[Symbol, Symbol]], elem1: T, elem2: T
    ) -> bool:
        """Check if two domain elements are equal."""
        ...


class PreDomainWedge(PreDomain[T]):
    """Pre-domain that can abstract through wedge domain."""

    def abstract_wedge(
        self, context: Context, symbols: List[Tuple[Symbol, Symbol]], wedge_element: Any
    ) -> T:
        """Abstract a wedge element to this domain."""
        ...


class Domain(Protocol[T]):
    """Protocol for complete iteration domains."""

    def abstract(self, context: Context, transition_formula: Any) -> T:
        """Abstract a transition formula."""
        ...

    def closure(self, domain_element: T) -> FormulaExpression:
        """Compute the transitive closure formula."""
        ...

    def tr_symbols(self, domain_element: T) -> List[Tuple[Symbol, Symbol]]:
        """Get transition symbols."""
        ...

    def pp(self, formatter: Any, domain_element: T) -> None:
        """Pretty print domain element."""
        ...


@dataclass(frozen=True)
class WedgeGuardElement:
    """Element of wedge guard domain: (precondition, postcondition)."""

    precondition: Any  # Wedge element
    postcondition: Any  # Wedge element

    def __str__(self) -> str:
        return f"WedgeGuard(pre={self.precondition}, post={self.postcondition})"


class WedgeGuard:
    """Wedge-based guard for iteration using wedge abstract domain.

    This domain separates a transition into precondition and postcondition
    using the wedge abstract domain. The wedge domain tracks linear
    inequalities and provides precise analysis of linear programs.

    Implements the WedgeGuard module from src/iteration.ml.
    """

    def abstract(self, srk: Context, tf: Any) -> WedgeGuardElement:
        """Abstract transition formula using wedge domain."""
        try:
            from .wedge import wedge_hull
            from .transitionFormula import (
                symbols as tf_symbols,
                post_symbols as tf_post,
                pre_symbols as tf_pre,
            )

            # Compute wedge hull of the transition formula
            wedge = wedge_hull(srk, tf)

            # Get pre and post symbols
            tr_symbols = tf_symbols(tf)
            pre_syms = tf_pre(tr_symbols)
            post_syms = tf_post(tr_symbols)

            # Project onto pre-state (eliminate post-state variables)
            precondition = wedge
            if hasattr(wedge, "exists"):
                precondition = wedge.exists(lambda s: s not in post_syms)

            # Project onto post-state (eliminate pre-state variables)
            postcondition = wedge
            if hasattr(wedge, "exists"):
                postcondition = wedge.exists(lambda s: s not in pre_syms)

            return WedgeGuardElement(precondition, postcondition)

        except (ImportError, AttributeError) as e:
            logger.warning(f"Failed to abstract with wedge: {e}")
            return WedgeGuardElement(None, None)

    def abstract_wedge(
        self, srk: Context, tr_symbols: List[Tuple[Symbol, Symbol]], wedge: Any
    ) -> WedgeGuardElement:
        """Abstract a wedge element to this domain."""
        from .transitionFormula import post_symbols as tf_post, pre_symbols as tf_pre

        pre_syms = tf_pre(tr_symbols)
        post_syms = tf_post(tr_symbols)

        # Project wedge onto pre and post spaces
        precondition = wedge
        postcondition = wedge

        if hasattr(wedge, "exists"):
            # Precondition: eliminate post-state variables
            precondition = wedge.exists(lambda s: s not in post_syms)
            # Postcondition: eliminate pre-state variables
            postcondition = wedge.exists(lambda s: s not in pre_syms)

        return WedgeGuardElement(precondition, postcondition)

    def exp(
        self,
        srk: Context,
        tr_symbols: List[Tuple[Symbol, Symbol]],
        loop_counter: ArithExpression,
        guard: WedgeGuardElement,
    ) -> FormulaExpression:
        """Compute exponential expression (loop iteration) in wedge domain.

        Returns (K = 0 ∧ identity) ∨ (K ≥ 1 ∧ pre ∧ post).
        """
        from .transitionFormula import identity, formula as tf_formula

        # Case 1: zero iterations: identity transition holds
        zero_case = mk_and(
            srk,
            [
                mk_eq(srk, loop_counter, mk_real(srk, QQ.zero())),
                tf_formula(identity(srk, tr_symbols)),
            ],
        )

        # Case 2: at least one iteration
        if guard.precondition is not None and guard.postcondition is not None:
            # Convert wedge to formula
            pre_formula = (
                guard.precondition.to_formula()
                if hasattr(guard.precondition, "to_formula")
                else mk_true(srk)
            )
            post_formula = (
                guard.postcondition.to_formula()
                if hasattr(guard.postcondition, "to_formula")
                else mk_true(srk)
            )

            at_least_one_case = mk_and(
                srk,
                [
                    mk_leq(srk, mk_real(srk, QQ.one()), loop_counter),
                    pre_formula,
                    post_formula,
                ],
            )
        else:
            at_least_one_case = mk_and(
                srk, [mk_leq(srk, mk_real(srk, QQ.one()), loop_counter)]
            )

        return mk_or(srk, [zero_case, at_least_one_case])

    def equal(
        self,
        srk: Context,
        tr_symbols: List[Tuple[Symbol, Symbol]],
        guard1: WedgeGuardElement,
        guard2: WedgeGuardElement,
    ) -> bool:
        """Check equality of wedge guard elements."""
        if guard1.precondition is None or guard2.precondition is None:
            return guard1.precondition is None and guard2.precondition is None

        if guard1.postcondition is None or guard2.postcondition is None:
            return guard1.postcondition is None and guard2.postcondition is None

        # Check using wedge equality
        pre_equal = False
        post_equal = False

        if hasattr(guard1.precondition, "equal"):
            pre_equal = guard1.precondition.equal(guard2.precondition)
        else:
            pre_equal = guard1.precondition == guard2.precondition

        if hasattr(guard1.postcondition, "equal"):
            post_equal = guard1.postcondition.equal(guard2.postcondition)
        else:
            post_equal = guard1.postcondition == guard2.postcondition

        return pre_equal and post_equal

    def join(
        self,
        srk: Context,
        tr_symbols: List[Tuple[Symbol, Symbol]],
        guard1: WedgeGuardElement,
        guard2: WedgeGuardElement,
    ) -> WedgeGuardElement:
        """Join two wedge guard elements."""
        if guard1.precondition is None or guard2.precondition is None:
            # If either is None, return the other (conservative)
            return guard2 if guard1.precondition is None else guard1

        # Join preconditions and postconditions
        pre_joined = guard1.precondition
        post_joined = guard1.postcondition

        if hasattr(guard1.precondition, "join") and guard2.precondition is not None:
            pre_joined = guard1.precondition.join(guard2.precondition)

        if hasattr(guard1.postcondition, "join") and guard2.postcondition is not None:
            post_joined = guard1.postcondition.join(guard2.postcondition)

        return WedgeGuardElement(pre_joined, post_joined)

    def widen(
        self,
        srk: Context,
        tr_symbols: List[Tuple[Symbol, Symbol]],
        guard1: WedgeGuardElement,
        guard2: WedgeGuardElement,
    ) -> WedgeGuardElement:
        """Widen two wedge guard elements."""
        if guard1.precondition is None or guard2.precondition is None:
            return guard2 if guard1.precondition is None else guard1

        # Widen preconditions and postconditions
        pre_widened = guard2.precondition
        post_widened = guard2.postcondition

        if hasattr(guard1.precondition, "widen") and guard2.precondition is not None:
            pre_widened = guard1.precondition.widen(guard2.precondition)

        if hasattr(guard1.postcondition, "widen") and guard2.postcondition is not None:
            post_widened = guard1.postcondition.widen(guard2.postcondition)

        return WedgeGuardElement(pre_widened, post_widened)

    def pp(
        self,
        srk: Context,
        tr_symbols: List[Tuple[Symbol, Symbol]],
        formatter: Any,
        guard: WedgeGuardElement,
    ) -> None:
        """Pretty print wedge guard element."""
        if formatter:
            formatter.write(
                f"pre:\\n  {guard.precondition}\\npost:\\n  {guard.postcondition}"
            )


@dataclass(frozen=True)
class PolyhedronGuardElement:
    """Element in the polyhedron guard domain."""

    def __init__(
        self, srk: Context, tr_symbols: List[Tuple[Symbol, Symbol]], polyhedron: Any
    ):
        self.srk = srk
        self.tr_symbols = tr_symbols
        self.polyhedron = polyhedron

    def __str__(self) -> str:
        return f"PolyhedronGuardElement({self.polyhedron})"


class PolyhedronGuard:
    """Polyhedron guard domain."""

    def abstract(self, srk: Context, tf: Any) -> PolyhedronGuardElement:
        """Abstract a transition formula into the polyhedron domain."""
        from .polyhedron import abstract as poly_abstract

        tr_symbols = tf.symbols
        poly = poly_abstract(srk, tf.formula)
        return PolyhedronGuardElement(srk, tr_symbols, poly)

    def join(
        self, srk: Context, elem1: PolyhedronGuardElement, elem2: PolyhedronGuardElement
    ) -> PolyhedronGuardElement:
        """Join two polyhedron elements."""
        from .polyhedron import union as poly_union

        joined_poly = poly_union(elem1.polyhedron, elem2.polyhedron)
        # Use the union of transition symbols
        all_symbols = list(set(elem1.tr_symbols + elem2.tr_symbols))
        return PolyhedronGuardElement(srk, all_symbols, joined_poly)

    def closure(self, elem: PolyhedronGuardElement) -> FormulaExpression:
        """Compute closure of domain element."""
        from .polyhedron import closure as poly_closure

        return poly_closure(elem.srk, elem.polyhedron)

    def tr_symbols(self, elem: PolyhedronGuardElement) -> List[Tuple[Symbol, Symbol]]:
        """Get transition symbols."""
        return elem.tr_symbols

    def pp(self, formatter: Any, elem: PolyhedronGuardElement) -> None:
        """Pretty print."""
        print(f"Polyhedron: {elem.polyhedron}", file=formatter)

    def widen(
        self, srk: Context, elem1: PolyhedronGuardElement, elem2: PolyhedronGuardElement
    ) -> PolyhedronGuardElement:
        """Widen two polyhedron elements."""
        from .polyhedron import widen as poly_widen

        widened_poly = poly_widen(elem1.polyhedron, elem2.polyhedron)
        all_symbols = list(set(elem1.tr_symbols + elem2.tr_symbols))
        return PolyhedronGuardElement(srk, all_symbols, widened_poly)

    def exp(self, srk: Context, elem: PolyhedronGuardElement, tr_symbols: List[Tuple[Symbol, Symbol]], loop_count: Any) -> FormulaExpression:
        """Compute concretization of a polyhedron guard domain element."""
        from .polyhedron import to_formula as poly_to_formula

        return poly_to_formula(elem.polyhedron)

    def equal(self, elem1: PolyhedronGuardElement, elem2: PolyhedronGuardElement) -> bool:
        """Check equality of two polyhedron guard elements."""
        return elem1.polyhedron == elem2.polyhedron

    def precondition(self, elem: PolyhedronGuardElement) -> FormulaExpression:
        """Get the pre-state condition of the guard (pre-map of polyhedron)."""
        from .transitionFormula import pre_map

        poly_formula = self.exp(elem.srk, elem, elem.tr_symbols, None)
        pre_map_fn = pre_map(elem.srk, list(zip([s for s, _ in elem.tr_symbols], [s for s, _ in elem.tr_symbols])))
        return pre_map_fn(poly_formula)

    def postcondition(self, elem: PolyhedronGuardElement) -> FormulaExpression:
        """Get the post-state condition of the guard."""
        from .polyhedron import to_formula as poly_to_formula

        return poly_to_formula(elem.polyhedron)


class LinearGuardElement:
    """Element of linear guard domain: (precondition, postcondition) formulas."""

    precondition: FormulaExpression
    postcondition: FormulaExpression


class LinearGuard:
    """Linear guard for iteration using LIA formulas.

    This domain uses linear integer arithmetic formulas for preconditions
    and postconditions, with MBP (model-based projection) for abstraction.

    Implements the LinearGuard module from src/iteration.ml.
    """

    def abstract(self, srk: Context, tf: Any) -> LinearGuardElement:
        """Abstract transition formula using linear domain."""
        try:
            from .transitionFormula import formula as tf_formula, symbols as tf_symbols
            from .transitionFormula import (
                post_symbols as tf_post,
                pre_symbols as tf_pre,
            )
            from .transitionFormula import exists as tf_exists
            from .nonlinear import linearize
            from .quantifier import mbp

            # Get the formula and linearize it
            phi = tf_formula(tf)
            phi = rewrite(srk, phi, down=nnf_rewriter(srk))
            lin_phi = linearize(srk, phi)

            # Get symbols
            tr_symbols = tf_symbols(tf)
            pre_syms = tf_pre(tr_symbols)
            post_syms = tf_post(tr_symbols)
            exists_pred = tf_exists(tf)

            # Precondition: project out post-state vars and existentials not in pre
            precondition = mbp(
                srk, lambda x: exists_pred(x) and x not in post_syms, lin_phi
            )

            # Postcondition: project out pre-state vars and existentials not in post
            postcondition = mbp(
                srk, lambda x: exists_pred(x) and x not in pre_syms, lin_phi
            )

            return LinearGuardElement(precondition, postcondition)

        except (ImportError, AttributeError) as e:
            logger.warning(f"Failed to abstract with linear guard: {e}")
            return LinearGuardElement(mk_true(srk), mk_true(srk))

    def exp(
        self,
        srk: Context,
        tr_symbols: List[Tuple[Symbol, Symbol]],
        loop_counter: ArithExpression,
        guard: LinearGuardElement,
    ) -> FormulaExpression:
        """Compute exponential expression."""
        from .transitionFormula import identity, formula as tf_formula

        # (K = 0 ∧ identity) ∨ (K ≥ 1 ∧ pre ∧ post)
        zero_case = mk_and(
            srk,
            [
                mk_eq(srk, loop_counter, mk_real(srk, QQ.zero())),
                tf_formula(identity(srk, tr_symbols)),
            ],
        )

        at_least_one_case = mk_and(
            srk,
            [
                mk_leq(srk, mk_real(srk, QQ.one()), loop_counter),
                guard.precondition,
                guard.postcondition,
            ],
        )

        return mk_or(srk, [zero_case, at_least_one_case])

    def join(
        self,
        srk: Context,
        tr_symbols: List[Tuple[Symbol, Symbol]],
        guard1: LinearGuardElement,
        guard2: LinearGuardElement,
    ) -> LinearGuardElement:
        """Join linear guards (disjunction)."""
        pre = mk_or(srk, [guard1.precondition, guard2.precondition])
        post = mk_or(srk, [guard1.postcondition, guard2.postcondition])
        return LinearGuardElement(pre, post)

    def widen(
        self,
        srk: Context,
        tr_symbols: List[Tuple[Symbol, Symbol]],
        guard1: LinearGuardElement,
        guard2: LinearGuardElement,
    ) -> LinearGuardElement:
        """Widen linear guards using polyhedra."""
        try:
            from .smt import equiv
            from .abstract import abstract as apron_abstract
            from .apron import formula_of_property, widen as apron_widen

            # Try Apron
            try:
                import apron

                man = apron.Manager("polka_strict")

                def widen_formula(
                    phi: FormulaExpression, psi: FormulaExpression
                ) -> FormulaExpression:
                    if equiv(srk, phi, psi) == "Yes":
                        return phi
                    else:
                        p = apron_abstract(srk, man, phi)
                        p_prime = apron_abstract(srk, man, psi)
                        return formula_of_property(apron_widen(p, p_prime))

                pre = widen_formula(guard1.precondition, guard2.precondition)
                post = widen_formula(guard1.postcondition, guard2.postcondition)
                return LinearGuardElement(pre, post)
            except ImportError:
                # Fallback to join if Apron not available
                return self.join(srk, tr_symbols, guard1, guard2)

        except Exception as e:
            logger.warning(f"Widening failed: {e}, falling back to join")
            return self.join(srk, tr_symbols, guard1, guard2)

    def equal(
        self,
        srk: Context,
        tr_symbols: List[Tuple[Symbol, Symbol]],
        guard1: LinearGuardElement,
        guard2: LinearGuardElement,
    ) -> bool:
        """Check equality using SMT equivalence."""
        try:
            from .smt import equiv

            pre_eq = equiv(srk, guard1.precondition, guard2.precondition) == "Yes"
            post_eq = equiv(srk, guard1.postcondition, guard2.postcondition) == "Yes"
            return pre_eq and post_eq
        except Exception:
            return False

    def pp(
        self,
        srk: Context,
        tr_symbols: List[Tuple[Symbol, Symbol]],
        formatter: Any,
        guard: LinearGuardElement,
    ) -> None:
        """Pretty print linear guard."""
        if formatter:
            formatter.write(
                f"precondition: {guard.precondition}\\npostcondition: {guard.postcondition}"
            )


@dataclass(frozen=True)
class LossyTranslationElement:
    """Lossy translation: list of (term, op, constant) constraints."""

    constraints: List[Tuple[ArithExpression, str, QQ]]  # (term, '≥'|'=', constant)


class LossyTranslation:
    """Abstract transition by lossy translation (recurrence inequations).

    Abstracts F(x,x') by inequations of the form:
        a(x') ≥ a(x) + c  or  a(x') = a(x) + c
    where a is a linear map and c is a scalar.

    Implements the LossyTranslation module from src/iteration.ml.
    """

    def abstract(self, srk: Context, tf: Any) -> LossyTranslationElement:
        """Abstract transition formula using lossy translation."""
        try:
            from .transitionFormula import formula as tf_formula, symbols as tf_symbols
            from .nonlinear import linearize
            from .abstract import abstract as apron_abstract
            from .apron import formula_of_property
            from .linear import linterm_of, const_dim

            # Linearize the formula
            phi = tf_formula(tf)
            phi = rewrite(srk, phi, down=nnf_rewriter(srk))
            lin_phi = linearize(srk, phi)

            # Create delta variables: delta_x = x' - x
            tr_symbols = tf_symbols(tf)
            delta_syms = []
            delta_map = {}

            for s, s_prime in tr_symbols:
                delta_name = f"delta_{s.name if hasattr(s, 'name') else str(s)}"
                delta_sym = mk_symbol(
                    srk, delta_name, s.typ if hasattr(s, "typ") else Type.INT
                )
                delta_syms.append(delta_sym)
                delta_map[delta_sym] = mk_sub(
                    srk, mk_const(srk, s_prime), mk_const(srk, s)
                )

            # Use Apron to compute delta polyhedron
            try:
                import apron

                man = apron.Manager("polka_strict")

                exists_pred = lambda x: x in delta_map
                delta_constraints = [
                    mk_eq(srk, mk_const(srk, delta), diff)
                    for delta, diff in delta_map.items()
                ]

                delta_phi = mk_and(srk, [lin_phi] + delta_constraints)
                delta_polyhedron = apron_abstract(
                    srk, man, delta_phi, exists=exists_pred
                )
                delta_formula = formula_of_property(delta_polyhedron)

                # Extract constraints from the polyhedron
                constraints = self._extract_constraints(srk, delta_formula, delta_map)
                return LossyTranslationElement(constraints)

            except ImportError:
                # Fallback: return empty constraints
                logger.warning("Apron not available for lossy translation")
                return LossyTranslationElement([])

        except Exception as e:
            logger.warning(f"Lossy translation abstraction failed: {e}")
            return LossyTranslationElement([])

    def _extract_constraints(
        self, srk: Context, formula: FormulaExpression, delta_map: Dict
    ) -> List[Tuple[ArithExpression, str, QQ]]:
        """Extract constraints of the form delta ≥ c or delta = c."""
        # This is a simplified extraction - a full implementation would
        # recursively analyze the formula structure
        return []

    def exp(
        self,
        srk: Context,
        tr_symbols: List[Tuple[Symbol, Symbol]],
        loop_counter: ArithExpression,
        elem: LossyTranslationElement,
    ) -> FormulaExpression:
        """Compute exponential: multiply each constraint by K."""
        formulas = []
        for delta, op, c in elem.constraints:
            if op == "=":
                formulas.append(
                    mk_eq(srk, mk_mul(srk, [mk_real(srk, c), loop_counter]), delta)
                )
            else:  # op == '≥'
                formulas.append(
                    mk_leq(srk, mk_mul(srk, [mk_real(srk, c), loop_counter]), delta)
                )

        return mk_and(srk, formulas) if formulas else mk_true(srk)

    def pp(
        self,
        srk: Context,
        tr_symbols: List[Tuple[Symbol, Symbol]],
        formatter: Any,
        elem: LossyTranslationElement,
    ) -> None:
        """Pretty print lossy translation."""
        if formatter:
            for term, op, c in elem.constraints:
                formatter.write(f"{term} {op} {c}\\n")


# ---------------------------------------------------------------------------
# GuardedTranslation: loops with constant affine increments
# ---------------------------------------------------------------------------


@dataclass
class GuardedTranslationElement:
    """Element of the guarded translation domain.

    Captures loops where x' = x + d for some constant direction d,
    modulo a guard invariant.
    """

    simulation: List[ArithExpression]  # linear combinations of pre-state vars
    translation: List[Fraction]  # constant increments per dimension
    guard: FormulaExpression  # invariant over simulation vars

    def __str__(self) -> str:
        return (
            f"GuardedTranslation(sim={self.simulation}, "
            f"trans={self.translation}, guard={self.guard})"
        )


class GuardedTranslation:
    """PreDomain for guarded affine translations.

    Captures loops where x' = x + d for some constant direction d,
    modulo a guard invariant.  The *simulation* vectors are linear
    functions whose difference across the transition is a known
    constant (vanishing-space / recurrence analysis).
    """

    def abstract(self, srk: Context, tf: Any) -> GuardedTranslationElement:
        """Abstract a transition formula into a guarded translation.

        Finds linear functions *f* such that f(x') - f(x) is constant
        (vanishing-space analysis).  Uses ``Abstract.vanishing_space`` on
        the vector [1, x0'-x0, ..., xn'-xn] to discover constant-increment
        recurrences, then builds a guard via ``Quantifier.mbp``.
        """
        from .abstract import vanishing_space
        from . import quantifier

        zz_symbols = [
            (s, sp) for s, sp in TF.symbols(tf)
            if hasattr(s, "typ") and s.typ == Type.INT
            and hasattr(sp, "typ") and sp.typ == Type.INT
        ]
        if not zz_symbols:
            zz_symbols = TF.symbols(tf)
        if not zz_symbols:
            return GuardedTranslationElement([], [], mk_true(srk))

        # delta = [1, x0' - x0, ..., xn' - xn]
        delta_terms: List[ArithExpression] = [mk_one(srk)]
        for s, sp in zz_symbols:
            delta_terms.append(mk_sub(srk, mk_const(srk, sp), mk_const(srk, s)))

        pre_term_arr = [mk_const(srk, s) for s, _ in zz_symbols]

        vanishing_vecs = vanishing_space(srk, TF.formula(tf), delta_terms)

        simulation: List[ArithExpression] = []
        translation: List[Fraction] = []

        for vec in vanishing_vecs:
            # Scale vec so that it has integer coefficients.
            common_denom = ZZ.one()
            for coeff, _dim in QQVector.entries(vec):
                common_denom = ZZ.lcm(common_denom, QQ.denominator(coeff))
            vec = QQVector.scalar_mul(QQ.of_zz(common_denom), vec)

            const_coeff = QQVector.get(vec, 0)
            functional = QQVector.add_term(QQ.negate(const_coeff), 0, vec)

            # Build term from functional part (dims 1..n map to pre_symbols[i-1])
            term = _term_of_vec_shifted(srk, pre_term_arr, functional)
            simulation.append(term)
            translation.append(const_coeff)

        # Build guard: exists x,x'. F(x,x') /\ Sx = y, then project to vars 0..n
        fresh_symbols = [
            mk_symbol(srk, f"gt_{i}", Type.INT) for i in range(len(simulation))
        ]
        sym_to_var: Dict[Symbol, Expression] = {}
        for i, sym in enumerate(fresh_symbols):
            sym_to_var[sym] = mk_var(srk, i, Type.INT)

        sx_eq_y = [
            mk_eq(srk, mk_const(srk, sym), sim_t)
            for sym, sim_t in zip(fresh_symbols, simulation)
        ]

        try:
            guard_formula = quantifier.mbp(
                srk,
                lambda x: x in sym_to_var,
                mk_and(srk, [TF.formula(tf)] + sx_eq_y),
            )
            guard = substitute_map(guard_formula, sym_to_var)
        except Exception:
            guard = mk_true(srk)

        return GuardedTranslationElement(
            simulation=simulation,
            translation=translation,
            guard=guard,
        )

    def exp(
        self,
        srk: Context,
        tr_symbols: List[Tuple[Symbol, Symbol]],
        loop_counter: ArithExpression,
        gt: GuardedTranslationElement,
    ) -> FormulaExpression:
        """Compute the K-th iterate formula using subcounter + MBP.

        For each dimension: post(sim_i) = sim_i + translation_i * K.
        The guard must hold for every intermediate step:
            forall subcounter. 0 <= subcounter < K => G(Sx + t*subcounter)
        We encode this via double negation + MBP.
        """
        from . import quantifier

        post_map = TF.post_map(srk, tr_symbols)

        def postify(expr: Expression) -> Expression:
            return substitute_const(srk, lambda sym: post_map.get(sym, mk_const(srk, sym)), expr)

        # forall subcounter. 0 <= subcounter < K => G(Sx + t*subcounter)
        subcounter_sym = mk_symbol(srk, "subcounter", Type.INT)
        subcounter_term = mk_const(srk, subcounter_sym)

        # Build cf[i] = sim_i + subcounter * translation_i
        cf = [
            mk_add(srk, [gt.simulation[i], mk_mul(srk, [subcounter_term, mk_real(srk, gt.translation[i])])])
            for i in range(len(gt.simulation))
        ]

        # G(Sx + t*subcounter): substitute sim_i -> cf[i] in guard
        guard_at_cf = substitute(srk, lambda i_var, typ: cf[i_var[0]] if i_var[0] < len(cf) else mk_var(srk, i_var[0], typ), gt.guard)
        # TODO: The above substitution uses De Bruijn vars; the guard was built
        # with mk_var(srk, i, Type.INT) so this should work.

        # mk_if(0 <= sc < K, guard_at_cf) encoded as:
        # not(0 <= sc /\ sc < K) \/ guard_at_cf
        # Then negate, MBP on subcounter, negate again
        impl = mk_or(srk, [
            mk_not(srk, mk_and(srk, [
                mk_leq(srk, mk_int(srk, 0), subcounter_term),
                mk_lt(srk, subcounter_term, loop_counter),
            ])),
            guard_at_cf,
        ])
        try:
            guard = mk_not(srk, quantifier.mbp(
                srk,
                lambda x: x != subcounter_sym,
                mk_not(srk, impl),
            ))
        except Exception:
            guard = mk_true(srk)

        # delta[i] = postify(sim_i) - sim_i
        delta_formulas = [
            mk_eq(
                srk,
                mk_sub(srk, postify(gt.simulation[i]), gt.simulation[i]),
                mk_mul(srk, [mk_real(srk, gt.translation[i]), loop_counter]),
            )
            for i in range(len(gt.simulation))
        ]

        return mk_and(srk, [guard] + delta_formulas)

    def pp(
        self,
        srk: Context,
        tr_symbols: List[Tuple[Symbol, Symbol]],
        formatter: Any,
        gt: GuardedTranslationElement,
    ) -> None:
        """Pretty print guarded translation."""
        if formatter:
            formatter.write(
                f"simulation: {gt.simulation}\\n"
                f"translation: {gt.translation}\\n"
                f"guard: {gt.guard}"
            )


# ---------------------------------------------------------------------------
# Split: decompose a loop on an invariant predicate
# ---------------------------------------------------------------------------


@dataclass
class SplitElement:
    """Element of the split domain.

    Maps each split predicate ``(not psi) -> (left, right)`` to a pair
    of inner-domain abstractions.  When no split is found, a single
    entry ``{true: (full_body, base_bottom)}`` is used.
    """

    splits: Dict[Expression, Tuple[Any, Any]]  # predicate -> (left, right)
    inner_domain: Any = None

    def __str__(self) -> str:
        n = len(self.splits)
        return f"Split({n} split(s))"


class Split:
    """PreDomain that splits a loop based on invariant predicates.

    If a predicate *psi* is invariant under the loop body
    (psi => post(psi)), then  body* = ([not psi] body)* ; ([psi] body)*.
    Each sub-loop is abstracted independently with *inner_domain*.
    Multiple splits can be discovered and composed.
    """

    def abstract(
        self,
        srk: Context,
        tf: Any,
        inner_domain: Optional[Any] = None,
    ) -> SplitElement:
        """Find invariant predicates and split the loop.

        Extracts all arithmetic atoms from the loop body as candidate
        predicates, filters complementary pairs, and checks each for
        invariance via SMT.
        """
        if inner_domain is None:
            inner_domain = WedgeGuard()

        body = TF.formula(tf)
        tr_symbols = TF.symbols(tf)
        exists_fn = tf.exists if hasattr(tf, "exists") else (lambda s: True)
        post_symbols = TF.post_symbols(tr_symbols)

        # --- 1. Extract candidate predicates from the body ---
        prestate_pred = lambda sym: exists_fn(sym) and sym not in post_symbols

        collected: Set[int] = set()  # use id() to track collected expressions
        predicates: List[Expression] = []

        def collect_rr(expr: Expression) -> Expression:
            from .srkSimplify import simplify_terms
            if isinstance(expr, Not):
                phi = expr.arg
                phi_syms = get_symbols(phi)
                if all(prestate_pred(s) for s in phi_syms):
                    expr_id = id(phi)
                    if expr_id not in collected:
                        collected.add(expr_id)
                        predicates.append(phi)
            elif hasattr(expr, "op") and hasattr(expr, "left") and hasattr(expr, "right"):
                # Atom(Arith(op, s, t))
                from .syntax import Eq, Lt, Leq
                s_expr = expr.left if hasattr(expr, "left") else None
                t_expr = expr.right if hasattr(expr, "right") else None
                if s_expr is not None and t_expr is not None:
                    if isinstance(expr, Eq):
                        phi = mk_eq(srk, s_expr, t_expr)
                    elif isinstance(expr, Lt):
                        phi = mk_lt(srk, s_expr, t_expr)
                    elif isinstance(expr, Leq):
                        phi = mk_leq(srk, s_expr, t_expr)
                    else:
                        return expr
                    phi_syms = get_symbols(phi)
                    if all(prestate_pred(s) for s in phi_syms):
                        # Check for complementary redundancy
                        expr_id = id(phi)
                        if expr_id not in collected:
                            collected.add(expr_id)
                            predicates.append(phi)
            return expr

        try:
            rewrite(srk, body, up=collect_rr)
        except Exception:
            pass

        # --- 2. Check which predicates are true split predicates ---
        from .nonlinear import uninterpret_rewriter

        uninterp_body = rewrite(srk, body, up=uninterpret_rewriter(srk))
        solver = Smt.mk_solver(srk)
        try:
            solver.add([uninterp_body])
        except Exception:
            pass

        def sat_modulo_body(psi: Expression) -> str:
            psi_uninterp = rewrite(srk, psi, up=uninterpret_rewriter(srk))
            solver.push()
            try:
                solver.add([psi_uninterp])
                result = solver.check()
                return result
            finally:
                solver.pop()

        def is_split_predicate(psi: Expression) -> bool:
            r1 = sat_modulo_body(psi)
            r2 = sat_modulo_body(mk_not(srk, psi))
            return (r1 == Smt.Sat and r2 == Smt.Sat)

        # --- 3. Build post_map for postify ---
        post_map = TF.post_map(srk, tr_symbols)

        def postify(expr: Expression) -> Expression:
            return substitute_const(srk, lambda sym: post_map.get(sym, mk_const(srk, sym)), expr)

        def abstract_formula(formula: Expression) -> Any:
            tf_new = TF.make(formula, tr_symbols, exists=exists_fn)
            return inner_domain.abstract(srk, tf_new)

        def base_bottom() -> Any:
            return abstract_formula(mk_false(srk))

        # --- 4. For each valid split predicate, check invariance ---
        splits: Dict[Expression, Tuple[Any, Any]] = {}

        for psi in predicates:
            if not is_split_predicate(psi):
                continue
            not_psi = mk_not(srk, psi)
            post_psi = postify(psi)
            post_not_psi = postify(not_psi)

            psi_body = mk_and(srk, [body, psi])
            not_psi_body = mk_and(srk, [body, not_psi])

            # Check {psi} body {not_psi} unsat  (i.e. psi is invariant)
            inv_check = mk_and(srk, [psi, post_not_psi])
            if sat_modulo_body(inv_check) == Smt.Unsat:
                # psi is invariant -> body* = ([not psi]body)* ([psi]body)*
                left_abstract = abstract_formula(not_psi_body)
                right_abstract = abstract_formula(psi_body)
                splits[not_psi] = (left_abstract, right_abstract)
            elif sat_modulo_body(mk_and(srk, [not_psi, post_psi])) == Smt.Unsat:
                # not_psi is invariant -> body* = ([psi]body)* ([not psi]body)*
                left_abstract = abstract_formula(psi_body)
                right_abstract = abstract_formula(not_psi_body)
                splits[psi] = (left_abstract, right_abstract)

        # --- 5. Fallback: no split found ---
        if not splits:
            whole = inner_domain.abstract(srk, tf)
            splits[mk_true(srk)] = (whole, base_bottom())

        return SplitElement(splits=splits, inner_domain=inner_domain)

    def exp(
        self,
        srk: Context,
        tr_symbols: List[Tuple[Symbol, Symbol]],
        loop_counter: ArithExpression,
        split: SplitElement,
    ) -> FormulaExpression:
        """Compute the K-th iterate for a split loop.

        For each split predicate, creates independent left/right loop
        counters and composes via TF.mul.  The total K = left_K + right_K.
        """
        if split.inner_domain is None:
            return mk_true(srk)

        conjuncts: List[FormulaExpression] = []

        for predicate, (left, right) in split.splits.items():
            not_predicate = mk_not(srk, predicate)

            left_counter = mk_const(srk, mk_symbol(srk, "K", Type.INT))
            right_counter = mk_const(srk, mk_symbol(srk, "K", Type.INT))

            left_closure = mk_and(srk, [
                split.inner_domain.exp(srk, tr_symbols, left_counter, left),
                mk_or(srk, [
                    mk_eq(srk, mk_real(srk, QQ.zero()), left_counter),
                    predicate,
                ]),
            ])
            right_closure = mk_and(srk, [
                split.inner_domain.exp(srk, tr_symbols, right_counter, right),
                mk_or(srk, [
                    mk_eq(srk, mk_real(srk, QQ.zero()), right_counter),
                    not_predicate,
                ]),
            ])

            # Compose via TF.mul (sequential composition)
            try:
                left_tf = TF.make(left_closure, tr_symbols)
                right_tf = TF.make(right_closure, tr_symbols)
                composed = TF.mul(srk, left_tf, right_tf)
                left_right = TF.formula(composed)
            except Exception:
                left_right = mk_and(srk, [left_closure, right_closure])

            conjuncts.append(left_right)
            conjuncts.append(mk_eq(srk, mk_add(srk, [left_counter, right_counter]), loop_counter))

        return mk_and(srk, conjuncts) if conjuncts else mk_true(srk)

    def pp(
        self,
        srk: Context,
        tr_symbols: List[Tuple[Symbol, Symbol]],
        formatter: Any,
        split: SplitElement,
    ) -> None:
        """Pretty print split element."""
        if formatter:
            formatter.write(f"predicate: {split.predicate}\\n")
            formatter.write(f"left: {split.left}\\n")
            formatter.write(f"right: {split.right}")


# ---------------------------------------------------------------------------
# Invariant transition predicates and phase partitioning
# ---------------------------------------------------------------------------


def _invariant_transition_predicates(
    srk: Context,
    tf: Any,
    predicates: List[Expression],
) -> List[Expression]:
    """Find the subset of predicates that are invariant for the transition.

    A predicate p(x,x') is invariant if T(x,x') /\\ T(x',x'') /\\ p(x,x')
    implies p(x',x'').  Uses SMT model caching for efficiency.
    """
    tr_symbols = TF.symbols(tf)
    exists_fn = tf.exists if hasattr(tf, "exists") else (lambda s: True)

    # map' sends primed vars to midpoints; map sends unprimed vars to midpoints
    map_prime: Dict[Symbol, Expression] = {}
    map_unprime: Dict[Symbol, Expression] = {}
    for sym, sym_prime in tr_symbols:
        mid_sym = mk_symbol(srk, f"mid_{sym.name if hasattr(sym, 'name') else sym}", sym.typ if hasattr(sym, "typ") else Type.INT)
        mid = mk_const(srk, mid_sym)
        map_prime[sym_prime] = mid
        map_unprime[sym] = mid

    # Build seq = T(x, x_mid) /\ T(x_mid, x')
    # For the first copy, rename Skolem constants to fresh symbols
    skolem_cache: Dict[Symbol, Symbol] = {}

    def rename_skolem(sym: Symbol) -> Symbol:
        if sym not in skolem_cache:
            skolem_cache[sym] = mk_symbol(srk, f"sk_{len(skolem_cache)}", sym.typ if hasattr(sym, "typ") else Type.INT)
        return skolem_cache[sym]

    def subst1(sym: Symbol) -> Expression:
        if sym in map_prime:
            return map_prime[sym]
        elif exists_fn(sym):
            return mk_const(srk, sym)
        else:
            return mk_const(srk, rename_skolem(sym))

    seq = mk_and(srk, [
        substitute_const(srk, subst1, TF.formula(tf)),
        substitute_map(TF.formula(tf), map_unprime),
    ])

    solver = Smt.mk_solver(srk)
    try:
        solver.add([seq])
    except Exception:
        return []

    # Check satisfiability of seq
    if solver.check() == Smt.Unsat:
        return []

    cached_models: List[Any] = []

    def is_invariant(p: Expression) -> bool:
        # p(x_mid, x') <-> p(x, x_mid)
        inv = mk_or(srk, [
            mk_not(srk, substitute_map(p, map_prime)),
            substitute_map(p, map_unprime),
        ])
        # Check cached models first
        for m in cached_models:
            try:
                if not m.evaluate_expression(inv):
                    return False
            except Exception:
                return False
        # Query solver
        solver.push()
        try:
            solver.add([mk_not(srk, inv)])
            result = solver.check()
            if result == Smt.Sat:
                model = solver.get_model()
                if model is not None:
                    cached_models.append(model)
                return False
            elif result == Smt.Unsat:
                return True
            else:
                return False
        finally:
            solver.pop()

    return [p for p in predicates if is_invariant(p)]


def _invariant_partition(
    srk: Context,
    candidates: List[Expression],
    tf: Any,
) -> Tuple[List[Expression], List[Tuple[Tuple[Any, Any], Expression]]]:
    """Partition transitions by invariant predicate valuations.

    Returns (predicates, cells) where each cell is
    ((pos_indices, neg_indices), cell_formula).
    """
    # Linearize first
    try:
        tf = TF.linearize(srk, tf)
    except Exception:
        pass

    predicates = _invariant_transition_predicates(srk, tf, candidates)
    predicates_arr = list(predicates)

    if not predicates_arr:
        return predicates_arr, []

    solver = Smt.mk_solver(srk)
    try:
        solver.add([TF.formula(tf)])
    except Exception:
        return predicates_arr, []

    cells: List[Tuple[Tuple[Set[int], Set[int]], Expression]] = []

    while True:
        solver.push()
        result = solver.check()
        if result == Smt.Sat:
            model = solver.get_model()
            if model is None:
                solver.pop()
                break
            cell_vals = []
            for pred in predicates_arr:
                try:
                    val = bool(model.evaluate_expression(pred))
                    cell_vals.append(val)
                except Exception:
                    cell_vals.append(False)

            pos_set = {i for i, v in enumerate(cell_vals) if v}
            neg_set = {i for i, v in enumerate(cell_vals) if not v}

            cell_formula = mk_and(srk, [
                pred if val else mk_not(srk, pred)
                for pred, val in zip(predicates_arr, cell_vals)
            ])
            cells.append(((pos_set, neg_set), cell_formula))
            solver.pop()
            try:
                solver.add([mk_not(srk, cell_formula)])
            except Exception:
                break
        else:
            solver.pop()
            break

    return predicates_arr, cells


def phase_graph(
    srk: Context,
    tf: Any,
    candidates: List[Expression],
    algebra: Any,
) -> Any:
    """Build a phase transition graph from invariant predicates.

    Nodes are cells (partitioned by invariant predicate valuations).
    Edges exist between cells where the rank (number of true predicates)
    of the source is <= the rank of the target.  Each cell has a
    self-loop weighted by the cell's constrained transition formula.
    """
    from .weightedGraph import WeightedGraph

    inv_predicates, cells = _invariant_partition(srk, candidates, tf)
    num_cells = len(cells)

    if num_cells == 0:
        return WeightedGraph(algebra)

    # Build ranked cells: rank -> list of (cell_index, (pos_set, neg_set))
    ranked: Dict[int, List[Tuple[int, Tuple[Set[int], Set[int]]]]] = {}
    for i, ((pos_set, neg_set), _) in enumerate(cells):
        rank = len(pos_set)
        ranked.setdefault(rank, []).append((i, (pos_set, neg_set)))

    levels = sorted(ranked.keys())

    # Build weighted graph with self-loops
    wg = WeightedGraph(algebra)
    for cell_ind in range(num_cells):
        wg = wg.add_vertex(cell_ind)

    for cell_ind, ((pos_set, neg_set), cell_formula) in enumerate(cells):
        cell_tf = TF.map_formula(lambda f: mk_and(srk, [f, cell_formula]), tf)
        wg = wg.add_edge(cell_ind, cell_tf, cell_ind)

    # Add edges between cells of increasing rank
    # (simplified: connect lower-rank cells to higher-rank cells
    #  if they can follow, using a conservative approximation)
    ancestors: List[Set[int]] = [set() for _ in range(num_cells)]
    descendants: List[Set[int]] = [set() for _ in range(num_cells)]

    for current_level_idx in range(1, len(levels)):
        current_level = levels[current_level_idx]
        targets = ranked.get(current_level, [])
        for prev_level_idx in range(current_level_idx - 1, -1, -1):
            prev_level = levels[prev_level_idx]
            sources = ranked.get(prev_level, [])
            for i, cell_i in sources:
                for j, cell_j in targets:
                    if j not in descendants[i]:
                        # Conservative: add edge if i's pos_set is subset of j's pos_set
                        if cell_i[0].issubset(cell_j[0]):
                            wg = wg.add_edge(i, algebra.one, j)
                            ancestors[j] = ancestors[j] | {i} | ancestors[i]
                            for k in ancestors[j]:
                                descendants[k].add(j)

    return wg


def phase_mp(
    srk: Context,
    candidate_predicates: List[Expression],
    tf: Any,
    nonterm: Any,
) -> Any:
    """Phase transition graph termination analysis.

    Builds a phase transition graph using invariant predicates and
    computes the omega path weight (non-termination condition) from
    a virtual entry node.
    """
    from .weightedGraph import WeightedGraph, omega_path_weight
    from . import weightedGraph as WG_module

    def star(transition_tf: Any) -> Any:
        """Approximate transitive closure using LossyTranslation."""
        k_sym = mk_symbol(srk, "K", Type.INT)
        exists_fn = lambda x: x != k_sym and (transition_tf.exists(x) if hasattr(transition_tf, "exists") else True)
        lt = LossyTranslation()
        elem = lt.abstract(srk, transition_tf)
        formula = lt.exp(srk, TF.symbols(transition_tf), mk_const(srk, k_sym), elem)
        return TF.make(formula, TF.symbols(transition_tf), exists=exists_fn)

    # Build algebras
    tr_symbols = TF.symbols(tf)

    tf_alg = WG_module.Algebra(
        mul=lambda a, b: TF.mul(srk, a, b),
        add=lambda a, b: TF.add(srk, a, b),
        one=TF.identity(srk, tr_symbols),
        zero=TF.zero(srk, tr_symbols),
        star=star,
    )

    mp_alg = WG_module.OmegaAlgebra(
        omega=nonterm,
        omega_add=lambda p1, p2: mk_or(srk, [p1, p2]),
        omega_mul=lambda transition, state: TF.preimage(srk, transition, state),
    )

    wg = phase_graph(srk, tf, candidate_predicates, tf_alg)

    # Add virtual entry node (-1) with edges to all isolated vertices
    wg = wg.add_vertex(-1)
    for v in list(wg.vertices()):
        if v == -1:
            continue
        # A vertex is "isolated" if it only has its self-loop
        succs = wg.successors(v)
        if len(succs) <= 1:
            wg = wg.add_edge(-1, tf_alg.one, v)

    return omega_path_weight(wg, mp_alg, -1)


# ---------------------------------------------------------------------------
# InvariantDirection: phase decomposition by direction of change
# ---------------------------------------------------------------------------


@dataclass
class InvariantDirectionElement:
    """Element of the direction-based phase domain.

    The loop is partitioned into *phases* according to invariant
    direction predicates (x < x', x' < x, x = x').  Each phase group
    contains cells with the same rank (number of true predicates),
    and lower-rank phases must precede higher-rank ones.
    """

    phases: List[List[Any]]  # list of phase groups, each a list of inner elements
    inner_domain: Any

    def __str__(self) -> str:
        n = sum(len(g) for g in self.phases)
        return f"InvariantDirection({n} phases)"


class InvariantDirection:
    """PreDomain that decomposes a loop into phases based on variable direction.

    For each variable the direction of change (increasing / decreasing /
    constant) is probed.  Only truly *invariant* direction predicates
    (those that hold on all iterations) are kept.  The transition space
    is partitioned into cells by the valuation of these predicates, and
    cells are grouped by rank into consecutive phase groups.
    """

    def abstract(
        self,
        srk: Context,
        tf: Any,
        inner_domain: Optional[Any] = None,
    ) -> InvariantDirectionElement:
        """Decompose a transition into direction-based phases.

        Uses ``invariant_transition_predicates`` to find truly invariant
        direction predicates, then enumerates cells via SMT model
        enumeration, sorts by rank, and groups consecutive equal-rank
        cells into phase groups.
        """
        if inner_domain is None:
            inner_domain = WedgeGuard()

        # Linearize the transition formula
        try:
            tf = TF.linearize(srk, tf)
        except Exception:
            pass

        tr_symbols = TF.symbols(tf)
        exists_fn = tf.exists if hasattr(tf, "exists") else (lambda s: True)

        if not tr_symbols:
            return InvariantDirectionElement([[tf]], inner_domain)

        # Create direction predicates: x < x', x' < x, x = x'
        candidate_predicates: List[Expression] = []
        for x_sym, x_prime_sym in tr_symbols:
            x = mk_const(srk, x_sym)
            x_prime = mk_const(srk, x_prime_sym)
            candidate_predicates.append(mk_lt(srk, x, x_prime))
            candidate_predicates.append(mk_lt(srk, x_prime, x))
            candidate_predicates.append(mk_eq(srk, x, x_prime))

        # Filter to invariant predicates
        inv_predicates = _invariant_transition_predicates(srk, tf, candidate_predicates)
        predicates_arr = list(inv_predicates)

        if not predicates_arr:
            # No invariant predicates found — single phase
            abstracted = inner_domain.abstract(srk, tf)
            return InvariantDirectionElement([[abstracted]], inner_domain)

        # Enumerate cells via SMT model enumeration
        solver = Smt.mk_solver(srk)
        try:
            solver.add([TF.formula(tf)])
        except Exception:
            pass

        cells: List[Tuple[List[bool], Expression]] = []
        while True:
            solver.push()
            result = solver.check()
            if result == Smt.Sat:
                model = solver.get_model()
                if model is None:
                    solver.pop()
                    break
                # Evaluate each predicate in the model
                cell_vals = []
                for pred in predicates_arr:
                    try:
                        val = model.evaluate_expression(pred)
                        cell_vals.append(bool(val))
                    except Exception:
                        cell_vals.append(False)

                cell_formula = mk_and(srk, [
                    pred if val else mk_not(srk, pred)
                    for pred, val in zip(predicates_arr, cell_vals)
                ])
                cells.append((cell_vals, cell_formula))

                # Block this cell
                solver.pop()
                try:
                    solver.add([mk_not(srk, cell_formula)])
                except Exception:
                    break
            else:
                solver.pop()
                break

        # Sort by weight (number of true predicates)
        def cell_weight(cell: Tuple[List[bool], Expression]) -> int:
            return sum(1 for v in cell[0] if v)

        cells.sort(key=cell_weight)

        # Group consecutive equal-weight cells
        from itertools import groupby

        phase_groups: List[List[Any]] = []
        for _weight, group_iter in groupby(cells, key=cell_weight):
            group_elements = []
            for cell_vals, cell_formula in group_iter:
                cell_predicates = [
                    pred if val else mk_not(srk, pred)
                    for pred, val in zip(predicates_arr, cell_vals)
                ]
                tf_constrained = TF.make(
                    mk_and(srk, [TF.formula(tf)] + cell_predicates),
                    tr_symbols,
                    exists=exists_fn,
                )
                group_elements.append(inner_domain.abstract(srk, tf_constrained))
            phase_groups.append(group_elements)

        if not phase_groups:
            abstracted = inner_domain.abstract(srk, tf)
            phase_groups = [[abstracted]]

        return InvariantDirectionElement(phases=phase_groups, inner_domain=inner_domain)

    def exp(
        self,
        srk: Context,
        tr_symbols: List[Tuple[Symbol, Symbol]],
        loop_counter: ArithExpression,
        id_elem: InvariantDirectionElement,
    ) -> FormulaExpression:
        """Compute the K-th iterate for a direction-decomposed loop.

        When multiple phase groups exist, creates intermediate symbols
        (mid_x) between groups and independent loop counters per group.
        The total K = k_0 + k_1 + ... + k_{n-1}.
        """
        if len(id_elem.phases) == 1 and len(id_elem.phases[0]) == 1:
            return id_elem.inner_domain.exp(
                srk, tr_symbols, loop_counter, id_elem.phases[0][0]
            )

        def exp_group(
            syms: List[Tuple[Symbol, Symbol]],
            k: ArithExpression,
            cells: List[Any],
        ) -> FormulaExpression:
            """Disjunction of cell exponentials under the given symbols."""
            return mk_or(srk, [
                id_elem.inner_domain.exp(srk, syms, k, cell)
                for cell in cells
            ])

        # Recursive decomposition: for each group except the last,
        # create intermediate symbols and compose.
        def go(
            groups: List[List[Any]],
            current_tr_symbols: List[Tuple[Symbol, Symbol]],
            exp_formulas: List[FormulaExpression],
            loop_counters: List[ArithExpression],
        ) -> Tuple[List[FormulaExpression], List[ArithExpression]]:
            if not groups:
                return (
                    [TF.formula(TF.identity(srk, current_tr_symbols))],
                    [mk_real(srk, QQ.zero())],
                )
            if len(groups) == 1:
                k = mk_const(srk, mk_symbol(srk, "k", Type.INT))
                return exp_formulas + [exp_group(current_tr_symbols, k, groups[0])], loop_counters + [k]

            # Create intermediate symbols
            mid = [
                (mk_symbol(srk, f"mid_{sym.name if hasattr(sym, 'name') else sym}", Type.INT), sym)
                for sym, _ in current_tr_symbols
            ]
            # tr_symbols1: pre -> mid
            tr_symbols1 = [
                (current_tr_symbols[i][0], mid[i][0])
                for i in range(len(current_tr_symbols))
            ]
            # tr_symbols2: mid -> post
            tr_symbols2 = [
                (mid[i][0], current_tr_symbols[i][1])
                for i in range(len(current_tr_symbols))
            ]
            k = mk_const(srk, mk_symbol(srk, "k", Type.INT))
            return go(
                groups[1:],
                tr_symbols2,
                exp_formulas + [exp_group(tr_symbols1, k, groups[0])],
                loop_counters + [k],
            )

        formulas, counters = go(id_elem.phases, tr_symbols, [], [])
        return mk_and(srk, [mk_eq(srk, loop_counter, mk_add(srk, counters))] + formulas)

    def pp(
        self,
        srk: Context,
        tr_symbols: List[Tuple[Symbol, Symbol]],
        formatter: Any,
        id_elem: InvariantDirectionElement,
    ) -> None:
        """Pretty print direction-decomposed element."""
        if formatter:
            formatter.write(f"phases ({len(id_elem.phases)} groups):\\n")
            for i, group in enumerate(id_elem.phases):
                formatter.write(f"  group {i}: {len(group)} phase(s)\\n")


class Product:
    """Product of two domains."""

    def __init__(self, domain_a: Any, domain_b: Any):
        self.domain_a = domain_a
        self.domain_b = domain_b

    def abstract(self, srk: Context, tf: Any) -> Tuple[Any, Any]:
        """Abstract in both domains."""
        return (self.domain_a.abstract(srk, tf), self.domain_b.abstract(srk, tf))

    def exp(
        self,
        srk: Context,
        tr_symbols: List[Tuple[Symbol, Symbol]],
        loop_counter: ArithExpression,
        elem: Tuple[Any, Any],
    ) -> FormulaExpression:
        """Compute exponential in product."""
        a_exp = self.domain_a.exp(srk, tr_symbols, loop_counter, elem[0])
        b_exp = self.domain_b.exp(srk, tr_symbols, loop_counter, elem[1])
        return mk_and(srk, [a_exp, b_exp])

    def pp(
        self,
        srk: Context,
        tr_symbols: List[Tuple[Symbol, Symbol]],
        formatter: Any,
        elem: Tuple[Any, Any],
    ) -> None:
        """Pretty print product."""
        self.domain_a.pp(srk, tr_symbols, formatter, elem[0])
        formatter.write("\\n")
        self.domain_b.pp(srk, tr_symbols, formatter, elem[1])


# ---------------------------------------------------------------------------
# ProductWedge: optimized product when both domains are Wedge-based
# ---------------------------------------------------------------------------


class ProductWedge:
    """Product domain optimized for Wedge-based sub-domains.

    Shares ``wedge_hull`` computation when both sub-domains are
    PreDomainWedge instances, avoiding redundant abstraction.
    """

    def __init__(self, domain_a: Any, domain_b: Any):
        self.domain_a = domain_a
        self.domain_b = domain_b

    def abstract(self, srk: Context, tf: Any) -> Tuple[Any, Any]:
        """Abstract using shared wedge hull when both domains support it."""
        if hasattr(self.domain_a, "abstract_wedge") and hasattr(self.domain_b, "abstract_wedge"):
            try:
                wedge = TF.wedge_hull(srk, tf)
                tr_symbols = TF.symbols(tf)
                a = self.domain_a.abstract_wedge(srk, tr_symbols, wedge)
                b = self.domain_b.abstract_wedge(srk, tr_symbols, wedge)
                return (a, b)
            except Exception:
                pass
        return (self.domain_a.abstract(srk, tf), self.domain_b.abstract(srk, tf))

    def abstract_wedge(self, srk: Context, tr_symbols: List[Tuple[Symbol, Symbol]], wedge: Any) -> Tuple[Any, Any]:
        """Abstract from a pre-computed wedge."""
        a = self.domain_a.abstract_wedge(srk, tr_symbols, wedge) if hasattr(self.domain_a, "abstract_wedge") else self.domain_a.abstract(srk, wedge)
        b = self.domain_b.abstract_wedge(srk, tr_symbols, wedge) if hasattr(self.domain_b, "abstract_wedge") else self.domain_b.abstract(srk, wedge)
        return (a, b)

    def exp(
        self,
        srk: Context,
        tr_symbols: List[Tuple[Symbol, Symbol]],
        loop_counter: ArithExpression,
        elem: Tuple[Any, Any],
    ) -> FormulaExpression:
        """Compute exponential in product."""
        a_exp = self.domain_a.exp(srk, tr_symbols, loop_counter, elem[0])
        b_exp = self.domain_b.exp(srk, tr_symbols, loop_counter, elem[1])
        return mk_and(srk, [a_exp, b_exp])

    def pp(
        self,
        srk: Context,
        tr_symbols: List[Tuple[Symbol, Symbol]],
        formatter: Any,
        elem: Tuple[Any, Any],
    ) -> None:
        """Pretty print product."""
        self.domain_a.pp(srk, tr_symbols, formatter, elem[0])
        formatter.write("\\n")
        self.domain_b.pp(srk, tr_symbols, formatter, elem[1])


# ---------------------------------------------------------------------------
# NonlinearRecurrenceInequation: abstract nonlinear recurrences
# ---------------------------------------------------------------------------


@dataclass
class NRIElement:
    """Element of the nonlinear recurrence inequation domain.

    Each entry is (lhs_term, op, rhs_term) where op is '=' or '>='.
    Captures constraints of the form  a*x' >= a*x + t(y)  or
    a*x' = a*x + t(y)  for nonlinear t.
    """

    constraints: List[Tuple[ArithExpression, str, ArithExpression]]

    def __str__(self) -> str:
        return f"NRI({len(self.constraints)} constraints)"


class NonlinearRecurrenceInequation:
    """PreDomain for nonlinear recurrence inequations.

    Abstracts a transition formula into constraints of the form
    ``lhs >= rhs`` or ``lhs = rhs`` where the left-hand side involves
    delta terms (x' - x) and the right-hand side involves nonlinear
    terms over pre-state variables.

    Ported from OCaml's NonlinearRecurrenceInequation module.
    """

    def abstract(self, srk: Context, tf: Any) -> NRIElement:
        """Abstract transition formula into nonlinear recurrence constraints."""
        try:
            from .wedge import (
                Wedge, copy as wedge_copy, meet_atoms, exists as wedge_exists,
                is_bottom as wedge_is_bottom, to_atoms, abstract as wedge_abstract,
            )
        except ImportError:
            return NRIElement([])

        tr_symbols = TF.symbols(tf)
        exists_fn = tf.exists if hasattr(tf, "exists") else (lambda s: True)

        # Create delta symbols and delta_map
        delta_syms, delta_map = _make_deltas(srk, tr_symbols)

        # Build the set of pre/post symbols
        syms: Set[Symbol] = set()
        for s, sp in tr_symbols:
            syms.add(s)
            syms.add(sp)

        # Build delta wedge: copy the transition formula's wedge,
        # add delta constraints, then project
        delta_constraints = [
            mk_eq(srk, mk_const(srk, d), diff)
            for d, diff in delta_map.items()
        ]

        try:
            delta_wedge = wedge_abstract(
                srk,
                mk_and(srk, [TF.formula(tf)] + delta_constraints),
                exists=lambda x: delta_map.get(x) is not None or (exists_fn(x) and x not in syms),
                subterm=lambda x: delta_map.get(x) is None,
            )
        except Exception:
            return NRIElement([])

        return _abstract_delta_wedge(srk, delta_wedge, delta_syms, delta_map)

    def abstract_wedge(
        self,
        srk: Context,
        tr_symbols: List[Tuple[Symbol, Symbol]],
        wedge: Any,
    ) -> NRIElement:
        """Abstract from a pre-computed wedge."""
        try:
            from .wedge import (
                copy as wedge_copy, meet_atoms, exists as wedge_exists,
                is_bottom as wedge_is_bottom, to_atoms,
            )
        except ImportError:
            return NRIElement([])

        delta_syms, delta_map = _make_deltas(srk, tr_symbols)

        syms: Set[Symbol] = set()
        for s, sp in tr_symbols:
            syms.add(s)
            syms.add(sp)

        delta_constraints = [
            mk_eq(srk, mk_const(srk, d), diff)
            for d, diff in delta_map.items()
        ]

        try:
            delta_wedge = wedge_copy(wedge)
            meet_atoms(delta_wedge, delta_constraints)
            delta_wedge = wedge_exists(
                delta_wedge,
                exists=lambda x: x not in syms,
                subterm=lambda x: delta_map.get(x) is None,
            )
        except Exception:
            return NRIElement([])

        return _abstract_delta_wedge(srk, delta_wedge, delta_syms, delta_map)

    def exp(
        self,
        srk: Context,
        tr_symbols: List[Tuple[Symbol, Symbol]],
        loop_counter: ArithExpression,
        elem: NRIElement,
    ) -> FormulaExpression:
        """Compute exponential: multiply each constraint by K."""
        formulas: List[FormulaExpression] = []
        for delta, op, rhs in elem.constraints:
            scaled_rhs = mk_mul(srk, [rhs, loop_counter])
            if op == "=":
                formulas.append(mk_eq(srk, delta, scaled_rhs))
            else:  # '>='
                formulas.append(mk_leq(srk, scaled_rhs, delta))
        return mk_and(srk, formulas) if formulas else mk_true(srk)

    def pp(
        self,
        srk: Context,
        tr_symbols: List[Tuple[Symbol, Symbol]],
        formatter: Any,
        elem: NRIElement,
    ) -> None:
        """Pretty print NRI element."""
        if formatter:
            for lhs, op, rhs in elem.constraints:
                formatter.write(f"{lhs} {op} {rhs}\\n")


def _make_deltas(
    srk: Context,
    tr_symbols: List[Tuple[Symbol, Symbol]],
) -> Tuple[List[Symbol], Dict[Symbol, Expression]]:
    """Create delta symbols and mapping for a set of transition symbols."""
    delta_syms: List[Symbol] = []
    delta_map: Dict[Symbol, Expression] = {}
    for s, sp in tr_symbols:
        d = mk_symbol(srk, f"delta_{s.name if hasattr(s, 'name') else s}", s.typ if hasattr(s, "typ") else Type.INT)
        delta_syms.append(d)
        delta_map[d] = mk_sub(srk, mk_const(srk, sp), mk_const(srk, s))
    return delta_syms, delta_map


def _abstract_delta_wedge(
    srk: Context,
    delta_wedge: Any,
    delta_syms: List[Symbol],
    delta_map: Dict[Symbol, Expression],
) -> NRIElement:
    """Extract NRI constraints from a delta wedge."""
    try:
        from .wedge import is_bottom as wedge_is_bottom, to_atoms
        from .linear import QQVector, const_dim
    except ImportError:
        return NRIElement([])

    try:
        if wedge_is_bottom(delta_wedge):
            return NRIElement([])

        atoms = to_atoms(delta_wedge)
        delta_dim_set: Set[int] = set()

        # Identify which coordinates correspond to delta symbols
        if hasattr(delta_wedge, 'cs'):
            cs = delta_wedge.cs
            for d in delta_syms:
                try:
                    dim_id = cs.cs_term_id(("App", d, []))
                    delta_dim_set.add(dim_id)
                except (KeyError, AttributeError, TypeError):
                    pass

        constraints: List[Tuple[ArithExpression, str, ArithExpression]] = []
        for atom in atoms:
            try:
                if isinstance(atom, Eq):
                    constraints.append((atom.left, "=", atom.right))
                elif isinstance(atom, Lt):
                    constraints.append((atom.left, "<", atom.right))
                elif isinstance(atom, Leq):
                    constraints.append((atom.left, "<=", atom.right))
                elif hasattr(atom, "left") and hasattr(atom, "right"):
                    # Handle wrapper objects that have left/right attributes
                    left = getattr(atom, "left")
                    right = getattr(atom, "right")
                    if left is not None and right is not None:
                        op = "="
                        if hasattr(atom, '__class__'):
                            cls = atom.__class__.__name__
                            if cls == 'Lt': op = "<"
                            elif cls == 'Leq': op = "<="
                        constraints.append((left, op, right))
            except Exception:
                pass

        return NRIElement(constraints)
    except Exception:
        return NRIElement([])


@dataclass(frozen=True)
class IterationDomainElement:
    """Element of an iteration domain."""

    srk: Context
    tr_symbols: List[Tuple[Symbol, Symbol]]
    iter_element: Any


class MakeDomain:
    """Make a complete domain from a pre-domain."""

    def __init__(self, iter_domain: Any):
        self.iter_domain = iter_domain

    def abstract(self, srk: Context, tf: Any) -> IterationDomainElement:
        """Abstract transition formula."""
        from .transitionFormula import symbols_of as tf_symbols

        elem = self.iter_domain.abstract(srk, tf)
        tr_syms = tf_symbols(tf)
        return IterationDomainElement(srk, tr_syms, elem)

    def closure(self, elem: IterationDomainElement) -> FormulaExpression:
        """Compute transitive closure."""
        loop_counter_sym = mk_symbol(elem.srk, "K", Type.INT)
        loop_counter = mk_const(elem.srk, loop_counter_sym)

        closure_formula = self.iter_domain.exp(
            elem.srk, elem.tr_symbols, loop_counter, elem.iter_element
        )

        # Add constraint K ≥ 0
        return mk_and(
            elem.srk,
            [
                closure_formula,
                mk_leq(elem.srk, mk_real(elem.srk, QQ.zero()), loop_counter),
            ],
        )

    def tr_symbols(self, elem: IterationDomainElement) -> List[Tuple[Symbol, Symbol]]:
        """Get transition symbols."""
        return elem.tr_symbols

    def pp(self, formatter: Any, elem: IterationDomainElement) -> None:
        """Pretty print."""
        self.iter_domain.pp(elem.srk, elem.tr_symbols, formatter, elem.iter_element)


# Convenience functions
def make_wedge_guard() -> WedgeGuard:
    """Create a wedge guard domain."""
    return WedgeGuard()


def make_polyhedron_guard() -> PolyhedronGuard:
    """Create a polyhedron guard domain."""
    return PolyhedronGuard()


def make_linear_guard() -> LinearGuard:
    """Create a linear guard domain."""
    return LinearGuard()


def make_lossy_translation() -> LossyTranslation:
    """Create a lossy translation domain."""
    return LossyTranslation()


def make_guarded_translation() -> GuardedTranslation:
    """Create a guarded translation domain."""
    return GuardedTranslation()


def make_split() -> Split:
    """Create a split domain."""
    return Split()


def make_invariant_direction() -> InvariantDirection:
    """Create an invariant direction domain."""
    return InvariantDirection()


def make_product(domain_a: Any, domain_b: Any) -> Product:
    """Create a product domain."""
    return Product(domain_a, domain_b)


def make_product_wedge(domain_a: Any, domain_b: Any) -> ProductWedge:
    """Create an optimized wedge product domain."""
    return ProductWedge(domain_a, domain_b)


def make_nonlinear_recurrence() -> NonlinearRecurrenceInequation:
    """Create a nonlinear recurrence inequation domain."""
    return NonlinearRecurrenceInequation()


def make_iteration_domain(iter_domain: Any) -> MakeDomain:
    """Create a complete iteration domain from a pre-domain."""
    return MakeDomain(iter_domain)


class IterationEngine:
    """Engine for computing iterations and transitive closures."""

    def __init__(self, domain: Any):
        """Initialize iteration engine with a domain."""
        self.domain = domain

    def compute_closure(self, srk: Context, tf: Any) -> Any:
        """Compute the closure of a transition formula."""
        elem = self.domain.abstract(srk, tf)
        return self.domain.closure(elem)


def compare(a: Any, b: Any) -> int:
    """Syntactic comparison of domain elements (mirrors OCaml PreDomain.compare)."""
    if a is b: return 0
    ha = hash(a) if hasattr(a, '__hash__') else id(a)
    hb = hash(b) if hasattr(b, '__hash__') else id(b)
    return -1 if ha < hb else (1 if ha > hb else 0)

    def iterate(
        self, srk: Context, initial: Any, transition: Any, max_iterations: int = 10
    ) -> Any:
        """Iterate a transition formula multiple times."""
        current = initial
        for _ in range(max_iterations):
            next_elem = self.domain.abstract(srk, transition)
            current = self.domain.join(srk, current, next_elem)
        return current
