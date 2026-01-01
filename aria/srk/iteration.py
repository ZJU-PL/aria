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
    symbols as get_symbols,
    substitute,
    rewrite,
    nnf_rewriter,
)
from .qQ import QQ
from .linear import QQVector, QQMatrix

logger = logging.getLogger(__name__)

T = TypeVar("T")
U = TypeVar("U")


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
        # Use the union of transition symbols
        all_symbols = list(set(elem1.tr_symbols + elem2.tr_symbols))
        return PolyhedronGuardElement(srk, all_symbols, widened_poly)


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
        from .transitionFormula import symbols as tf_symbols

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


def make_product(domain_a: Any, domain_b: Any) -> Product:
    """Create a product domain."""
    return Product(domain_a, domain_b)


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

    def iterate(
        self, srk: Context, initial: Any, transition: Any, max_iterations: int = 10
    ) -> Any:
        """Iterate a transition formula multiple times."""
        current = initial
        for _ in range(max_iterations):
            next_elem = self.domain.abstract(srk, transition)
            current = self.domain.join(srk, current, next_elem)
        return current
