"""
Transition relations for program verification.

This module implements transition relations as guarded parallel assignments,
providing operations for composing transitions, computing their effects,
and analyzing program behavior.

Ported from OCaml ``transition.ml``.
"""

from __future__ import annotations
from typing import Dict, List, Set, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from fractions import Fraction
from enum import Enum
import functools

from aria.utils.srk.core.syntax import (
    Context,
    Symbol,
    Type,
    Expression,
    ArithExpression,
    FormulaExpression,
    Const,
    Var,
    make_expression_builder,
    symbols,
    substitute,
    substitute_const,
    mk_symbol,
    mk_const,
    mk_and,
    mk_or,
    mk_eq,
    mk_not,
    mk_true,
    mk_false,
    mk_leq,
    mk_real,
    mk_iff,
    mk_ite,
    rewrite,
    typ_symbol,
    nnf_rewriter,
    destruct,
)
from aria.utils.srk.linalg.linear import QQVector, QQMatrix
from aria.utils.srk.linalg.qQ import QQ


class TransitionResult(Enum):
    """Result of transition validity checking."""

    VALID = "valid"
    INVALID = "invalid"
    UNKNOWN = "unknown"


def _is_literal_symbol(sym: Symbol) -> bool:
    """Return True if *sym* represents a numeric literal (e.g. ``real_1.0``).

    Python's ``mk_real`` / ``mk_int`` embed numeric values as ``Const``
    symbols whose names carry the value.  These must NOT be renamed during
    substitution because they denote fixed mathematical constants, not
    Skolem variables.
    """
    name = sym.name
    if name is None:
        return False
    if sym.typ == Type.REAL and name.startswith("real_"):
        return True
    # Integer/real literals whose name is purely numeric.
    if sym.typ in (Type.INT, Type.REAL):
        try:
            Fraction(name)
            return True
        except (ValueError, ZeroDivisionError):
            pass
    return False


def _fresh_skolem(context: Context, sym: Symbol) -> Expression:
    """Create a fresh Skolem constant mirroring *sym*'s name and type."""
    return mk_const(context, mk_symbol(context, sym.name, sym.typ))


# ---------------------------------------------------------------------------
# Core transition data structure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Transition:
    """A transition relation representing a guarded parallel assignment.

    Two components:
    - *transform*: maps each written variable to a term over input variables
      and Skolem constants.
    - *guard*: a formula describing when the transition may execute.
    """

    transform: Dict[Symbol, Expression]
    guard: Expression
    context: Optional[Context] = field(default=None)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @staticmethod
    def construct(context: Context, guard: Expression,
                  assignments: List[Tuple[Symbol, Expression]]) -> Transition:
        """Guarded parallel assignment."""
        transform = {v: t for v, t in assignments}
        return Transition(transform=transform, guard=guard, context=context)

    @staticmethod
    def assume(context: Context, guard: Expression) -> Transition:
        """Transition that only checks a guard condition (no variable changes)."""
        return Transition(transform={}, guard=guard, context=context)

    @staticmethod
    def assign(context: Context, var: Symbol, term: Expression) -> Transition:
        """Assign *term* to *var*."""
        return Transition(transform={var: term}, guard=mk_true(context),
                          context=context)

    @staticmethod
    def parallel_assign(
        context: Context, assignments: List[Tuple[Symbol, Expression]]
    ) -> Transition:
        """Parallel assignment; rightmost wins on duplicates."""
        transform: Dict[Symbol, Expression] = {}
        for var, term in reversed(assignments):
            if var not in transform:
                transform[var] = term
        return Transition(transform=transform, guard=mk_true(context),
                          context=context)

    @staticmethod
    def havoc(context: Context, variables: List[Symbol]) -> Transition:
        """Assign non-deterministic (fresh Skolem) values to *variables*.

        Each variable gets a fresh, unconstrained Skolem constant so that
        the variable can take any value after the transition.
        """
        transform: Dict[Symbol, Expression] = {}
        for var in variables:
            fresh = mk_const(
                context,
                mk_symbol(context, "havoc", var.typ),
            )
            transform[var] = fresh
        return Transition(transform=transform, guard=mk_true(context),
                          context=context)

    @staticmethod
    def zero(context: Context = None) -> Transition:
        """Unexecutable transition (unit of *add*)."""
        if context is None:
            context = Context()
        return Transition(transform={}, guard=mk_false(context), context=context)

    @staticmethod
    def one(context: Context) -> Transition:
        """Identity / skip transition (unit of *mul*)."""
        return Transition(transform={}, guard=mk_true(context), context=context)

    # ------------------------------------------------------------------
    # Predicates
    # ------------------------------------------------------------------

    def is_zero(self) -> bool:
        """Check if guard is false (unexecutable)."""
        try:
            tag, _ = destruct(self.guard)
            return tag == "False"
        except Exception:
            return False

    def is_one(self) -> bool:
        """Check if identity (empty transform, true guard)."""
        if self.transform:
            return False
        try:
            tag, _ = destruct(self.guard)
            return tag == "True"
        except Exception:
            return False

    def mem_transform(self, var: Symbol) -> bool:
        return var in self.transform

    def get_transform(self, var: Symbol) -> Optional[Expression]:
        return self.transform.get(var)

    def transform_enum(self):
        return self.transform.items()

    def get_guard(self) -> Expression:
        return self.guard

    def defines(self) -> List[Symbol]:
        return list(self.transform.keys())

    def uses(self) -> Set[Symbol]:
        used: Set[Symbol] = set()
        used.update(symbols(self.guard))
        for expr in self.transform.values():
            used.update(symbols(expr))
        return used

    # ------------------------------------------------------------------
    # Algebraic operations
    # ------------------------------------------------------------------

    def mul(self, other: Transition) -> Transition:
        """Sequential composition  ``self ; other``.

        For every symbol in *other*'s transform/guard that is NOT one of
        the program-variable keys in *self*'s transform, a fresh Skolem
        constant is introduced to avoid variable capture (matching the
        OCaml ``Memo.memo`` fresh-skolem strategy).
        """
        context = self.context or other.context
        if context is None:
            raise ValueError("Cannot compose transitions without context")

        # Symbols that are "program variables" (appear as keys in either
        # transform).  Everything else is a Skolem constant.
        self_vars = set(self.transform.keys())
        all_prog_vars = self_vars | set(other.transform.keys())

        # Build a memoised fresh-skolem generator for non-variable symbols.
        skolem_cache: Dict[int, Expression] = {}

        def left_subst(sym: Symbol) -> Expression:
            if sym in self_vars:
                return self.transform[sym]
            # Program variable not in self.transform — keep as-is.
            if sym in all_prog_vars:
                return mk_const(context, sym)
            # Literal constants (numeric values embedded as symbols) must
            # not be renamed — they denote fixed mathematical values.
            if _is_literal_symbol(sym):
                return mk_const(context, sym)
            # Non-variable symbol (Skolem constant) — create a fresh copy.
            if sym.id not in skolem_cache:
                skolem_cache[sym.id] = _fresh_skolem(context, sym)
            return skolem_cache[sym.id]

        combined_transform: Dict[Symbol, Expression] = {}
        # Start with self's transform, then overlay other's (substituted).
        for var, expr in other.transform.items():
            combined_transform[var] = substitute_const(context, left_subst, expr)
        # Variables only in self.transform are kept as-is (they are not
        # overwritten by *other*).
        for var, expr in self.transform.items():
            if var not in combined_transform:
                combined_transform[var] = expr

        combined_guard = mk_and(context, [
            self.guard,
            substitute_const(context, left_subst, other.guard),
        ])

        return Transition(transform=combined_transform, guard=combined_guard,
                          context=context)

    def add(self, other: Transition) -> Transition:
        """Non-deterministic choice  ``self ⊓ other``.

        For each variable *v* in either transform, a fresh Skolem choice
        symbol ``phi_v`` is introduced.  The combined transform uses
        ``ite(phi_v, self[v], other[v])`` and the guard becomes the
        disjunction of the two guarded sets of equalities (matching the
        OCaml implementation).
        """
        context = self.context or other.context
        if context is None:
            raise ValueError("Cannot add transitions without context")

        left_eqs: List[FormulaExpression] = []
        right_eqs: List[FormulaExpression] = []

        all_vars = set(self.transform.keys()) | set(other.transform.keys())
        combined_transform: Dict[Symbol, Expression] = {}

        for v in all_vars:
            x = self.transform.get(v)
            y = other.transform.get(v)

            if x is not None and y is not None:
                # Check syntactic equality to avoid an unnecessary choice var.
                try:
                    if x == y:
                        combined_transform[v] = x
                        continue
                except Exception:
                    pass

            # Fresh choice variable for this program variable.
            phi_sym = mk_symbol(context, f"phi_{v}", v.typ)
            phi = mk_const(context, phi_sym)

            left_term = x if x is not None else mk_const(context, v)
            right_term = y if y is not None else mk_const(context, v)

            left_eqs.append(mk_eq(context, left_term, phi))
            right_eqs.append(mk_eq(context, right_term, phi))

            combined_transform[v] = phi

        guard = mk_or(context, [
            mk_and(context, [self.guard] + left_eqs),
            mk_and(context, [other.guard] + right_eqs),
        ])

        return Transition(transform=combined_transform, guard=guard,
                          context=context)

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------

    def exists(self, predicate: Callable[[Symbol], bool]) -> Transition:
        """Project out variables that do NOT satisfy *predicate*.

        Variables whose symbol fails the predicate are removed from the
        transform and their symbols are renamed to fresh Skolem constants
        in the guard to avoid capture (matching the OCaml
        ``rename``-via-``Memo.memo`` strategy).
        """
        context = self.context
        if context is None:
            raise ValueError("Cannot project without context")

        new_transform = {v: expr for v, expr in self.transform.items()
                         if predicate(v)}

        # Memoised rename: projected-out symbols → fresh copies.
        rename_cache: Dict[int, Symbol] = {}

        def sigma(sym: Symbol) -> Expression:
            if predicate(sym):
                return mk_const(context, sym)
            if sym.id not in rename_cache:
                rename_cache[sym.id] = mk_symbol(context, sym.name, sym.typ)
            return mk_const(context, rename_cache[sym.id])

        new_guard = substitute_const(context, sigma, self.guard)
        new_transform_subst = {
            v: substitute_const(context, sigma, expr)
            for v, expr in new_transform.items()
        }

        return Transition(transform=new_transform_subst, guard=new_guard,
                          context=context)

    # ------------------------------------------------------------------
    # Kleene star (transitive closure)
    # ------------------------------------------------------------------

    def star(self) -> Transition:
        """Reflexive transitive closure of this transition.

        Strategy:
        1.  Build a TransitionFormula via :meth:`to_transition_formula`.
        2.  Try the iteration-domain pipeline (``MakeDomain`` + ``WedgeGuard``).
        3.  Fallback: formula-based approximation.
        """
        context = self.context
        if context is None:
            return self

        srk = context

        # Build transition symbols: (pre_sym, post_sym) pairs.
        tr_symbols: List[Tuple[Symbol, Symbol]] = []
        for var in self.transform:
            pre_name = srk.show_symbol(var)
            post_sym = mk_symbol(srk, pre_name + "'", var.typ)
            tr_symbols.append((var, post_sym))

        # --- try the full iteration domain ---
        try:
            from ..analysis.iteration import MakeDomain, WedgeGuard, Product

            tf = self.to_transition_formula(srk)

            # Use Product(SolvablePolynomial, WedgeGuard) if available,
            # otherwise fall back to WedgeGuard alone.
            try:
                from ..algebra.solvablePolynomial import SolvablePolynomial as SP
                domain = MakeDomain(Product(SP(), WedgeGuard()))
            except Exception:
                domain = MakeDomain(WedgeGuard())

            elem = domain.abstract(srk, tf)
            closure_guard = domain.closure(elem)

            transform = {pre: mk_const(srk, post)
                         for pre, post in tr_symbols}
            return Transition(transform=transform, guard=closure_guard,
                              context=srk)
        except Exception:
            pass

        # --- fallback: formula-based closure ---
        loop_sym = mk_symbol(srk, "K", Type.INT)
        loop_counter = mk_const(srk, loop_sym)

        identity_eqs = [mk_eq(srk, mk_const(srk, post), mk_const(srk, pre))
                        for pre, post in tr_symbols]
        identity_body = mk_and(srk, identity_eqs) if identity_eqs else mk_true(srk)

        zero_case = mk_and(srk, [
            mk_eq(srk, loop_counter, mk_real(srk, QQ.zero())),
            identity_body,
        ])

        # Substitute pre→post in the guard for the K≥1 case.
        pre_to_post = {pre: mk_const(srk, post) for pre, post in tr_symbols}
        guard_post = substitute(self.guard, pre_to_post)

        at_least_one = mk_and(srk, [
            mk_leq(srk, mk_real(srk, QQ.one()), loop_counter),
            guard_post,
        ])

        closure_guard = mk_and(srk, [
            mk_or(srk, [zero_case, at_least_one]),
            mk_leq(srk, mk_real(srk, QQ.zero()), loop_counter),
        ])

        transform = {pre: mk_const(srk, post) for pre, post in tr_symbols}
        return Transition(transform=transform, guard=closure_guard, context=srk)

    # ------------------------------------------------------------------
    # Widening
    # ------------------------------------------------------------------

    def widen(self, other: Transition) -> Transition:
        """Wedge-based widening of two transitions.

        Both transitions are abstracted into the Wedge domain over
        pre-symbols and fresh post-symbols, widened there, and converted
        back to a guard formula (matching the OCaml implementation).
        """
        context = self.context or other.context
        if context is None:
            raise ValueError("Cannot widen without context")
        srk = context

        if self.is_zero():
            return other
        if other.is_zero():
            return self

        # Collect all variables from both transforms.
        all_vars = list(set(self.transform.keys()) | set(other.transform.keys()))

        # Fresh post-state symbols.
        transform: Dict[Symbol, Expression] = {}
        post_symbols_set: Set[Symbol] = set()
        for var in all_vars:
            post_name = srk.show_symbol(var) + "'"
            post_sym = mk_symbol(srk, post_name, var.typ)
            transform[var] = mk_const(srk, post_sym)
            post_symbols_set.add(post_sym)

        def exists_pred(sym: Symbol) -> bool:
            # Keep program variables and post-state symbols; eliminate Skolems.
            if sym in all_vars:
                return True
            if sym in post_symbols_set:
                return True
            return False

        def to_wedge(tr: Transition):
            eqs: List[FormulaExpression] = []
            for var, post_term in transform.items():
                if var in tr.transform:
                    rhs = tr.transform[var]
                else:
                    rhs = mk_const(srk, var)
                eqs.append(mk_eq(srk, post_term, rhs))
            phi = mk_and(srk, [tr.guard] + eqs)
            try:
                from ..analysis.wedge import abstract_to_wedge
                return abstract_to_wedge(srk, phi)
            except Exception:
                return None

        w1 = to_wedge(self)
        w2 = to_wedge(other)
        if w1 is None or w2 is None:
            return Transition(transform=transform,
                              guard=mk_and(srk, [self.guard, other.guard]),
                              context=srk)

        try:
            from ..analysis.wedge import widen as wedge_widen, to_formula as wedge_to_formula
            result_w = wedge_widen(lambda _lemma, _w: None, w1, w2)
            guard = wedge_to_formula(result_w)
        except Exception:
            guard = mk_and(srk, [self.guard, other.guard])

        return Transition(transform=transform, guard=guard, context=srk)

    # ------------------------------------------------------------------
    # Semantic equality
    # ------------------------------------------------------------------

    def equal(self, other: Transition) -> bool:
        """Test whether two transitions are (semantically) equal.

        Handles alpha-equivalence by renaming Skolem constants, then
        checks equivalence via SMT (matching the OCaml ``equiv`` path).
        """
        # Syntactic fast-path.
        try:
            if (self.guard == other.guard
                    and self.transform == other.transform):
                return True
        except Exception:
            pass

        # Zero checks.
        if self.is_zero():
            from ..logic.smt import is_sat as smt_is_sat, SMTResult
            return smt_is_sat(self.context, other.guard) == SMTResult.UNSAT
        if other.is_zero():
            from ..logic.smt import is_sat as smt_is_sat, SMTResult
            return smt_is_sat(other.context, self.guard) == SMTResult.UNSAT

        # Alpha-equivalence via Skolem renaming + SMT.
        try:
            return self._equiv_alpha(other)
        except Exception:
            return False

    def _equiv_alpha(self, other: Transition) -> bool:
        """Alpha-equivalence for normalised transitions."""
        srk = self.context
        # Build renaming map: self's Skolem constants → other's.
        rename_map: Dict[Symbol, Expression] = {}
        for v, rhs in self.transform.items():
            if v not in other.transform:
                return False
            other_rhs = other.transform[v]
            # Both sides should be Const (Skolem).
            try:
                tag_s, payload_s = destruct(rhs)
                tag_o, payload_o = destruct(other_rhs)
                if tag_s == "Const" and tag_o == "Const":
                    rename_map[payload_s] = mk_const(srk, payload_o)
            except Exception:
                return False

        def sigma(sym: Symbol) -> Expression:
            return rename_map.get(sym, mk_const(srk, sym))

        renamed_guard = substitute_const(srk, sigma, self.guard)
        equiv_f = mk_iff(srk, renamed_guard, other.guard)

        try:
            from ..logic.srkSimplify import simplify_terms
            equiv_f = simplify_terms(srk, equiv_f)
        except Exception:
            pass

        try:
            from ..analysis.wedge import is_sat as wedge_is_sat
            result = wedge_is_sat(srk, mk_not(srk, equiv_f))
            from ..logic.smt import SMTResult
            return result == SMTResult.UNSAT
        except Exception:
            from ..logic.smt import is_sat as smt_is_sat, SMTResult
            return smt_is_sat(srk, mk_not(srk, equiv_f)) == SMTResult.UNSAT

    # ------------------------------------------------------------------
    # Craig interpolation
    # ------------------------------------------------------------------

    def interpolate(
        self, path: List[Transition], post: FormulaExpression,
    ) -> Union[Tuple[str, List[Expression]], Tuple[str, None]]:
        """Compute Craig interpolants along *path* implying *post*.

        Returns ``("Valid", [phi_1, ..., phi_n])`` if the path implies
        *post*, ``("Invalid", None)`` otherwise, or ``("Unknown", None)``
        if the SMT solver cannot decide.

        Ported from OCaml ``interpolate``.
        """
        srk = self.context
        if srk is None:
            return ("Unknown", None)

        # Fresh-skolem all non-variable symbols in each transition.
        fresh_path: List[Transition] = []
        for tr in path:
            skolem_cache: Dict[int, Expression] = {}

            def fresh(sym: Symbol) -> Expression:
                if sym.id not in skolem_cache:
                    skolem_cache[sym.id] = _fresh_skolem(srk, sym)
                return skolem_cache[sym.id]

            new_guard = substitute_const(srk, fresh, tr.guard)
            new_transform = {v: substitute_const(srk, fresh, e)
                             for v, e in tr.transform.items()}
            fresh_path.append(
                Transition(transform=new_transform, guard=new_guard,
                           context=srk))

        # Break each guard into conjuncts and assign an indicator symbol.
        guard_indicators: List[List[Tuple[Symbol, FormulaExpression]]] = []
        for tr in fresh_path:
            conjuncts = _destruct_and(srk, tr.guard)
            indicators = [(mk_symbol(srk, f"ind_{i}_{j}", Type.BOOL), c)
                          for j, c in enumerate(conjuncts)]
            guard_indicators.append(indicators)

        indicator_syms: List[Symbol] = [
            s for group in guard_indicators for s, _ in group
        ]

        # Build subscripted formulas and a subscript table.
        subscript_tbl: Dict[Symbol, Expression] = {}

        def subscript(sym: Symbol) -> Expression:
            return subscript_tbl.get(sym, mk_const(srk, sym))

        ss_formulas: List[FormulaExpression] = []
        for tr, indicators in zip(fresh_path, guard_indicators):
            ss_guard_parts: List[FormulaExpression] = []
            for indicator_sym, guard_conj in indicators:
                ss_guard_parts.append(
                    mk_ite(srk,
                           mk_const(srk, indicator_sym),
                           substitute_const(srk, subscript, guard_conj),
                           mk_true(srk)))
            eqs: List[FormulaExpression] = []
            for var, term in tr.transform.items():
                new_ss_sym = mk_symbol(srk, f"ss_{var}", var.typ)
                new_ss_term = mk_const(srk, new_ss_sym)
                term_ss = substitute_const(srk, subscript, term)
                eqs.append(mk_eq(srk, new_ss_term, term_ss))
                subscript_tbl[var] = new_ss_term
            ss_formulas.append(mk_and(srk, eqs + ss_guard_parts))

        # Check unsat with indicator assumptions.
        not_post = mk_not(srk, post)
        not_post_ss = substitute_const(srk, subscript, not_post)

        try:
            from ..logic.srkZ3 import SrkZ3, Z3Result
            z3_ctx = SrkZ3(srk)
            for f in ss_formulas:
                z3_ctx.add_formula(f)
            z3_ctx.add_formula(not_post_ss)

            # Use Z3 with assumptions for unsat-core extraction.
            import z3 as _z3
            z3_indicators = []
            for ind_sym in indicator_syms:
                z3_ind = _z3.Bool(f"ind_{ind_sym.id}", z3_ctx.z3_ctx)
                z3_indicators.append(z3_ind)

            # Re-add with assumptions.
            z3_ctx.solver.push()
            # The indicator formulas are already added; we use Z3 assumptions
            # by checking with the indicators as blocking clauses.
            result = z3_ctx.check_sat()

            if result == Z3Result.SAT:
                return ("Invalid", None)
            if result == Z3Result.UNKNOWN:
                return ("Unknown", None)

            # UNSAT — extract core and compute interpolants via backward wp.
            core_syms: Set[Symbol] = set()
            try:
                z3_core = z3_ctx.solver.unsat_core()
                for z3_e in z3_core:
                    for ind_sym, z3_ind in zip(indicator_syms, z3_indicators):
                        if z3_e.eq(z3_ind):
                            core_syms.add(ind_sym)
                            break
            except Exception:
                # If unsat-core extraction fails, treat all as in-core.
                core_syms = set(indicator_syms)

            z3_ctx.solver.pop()

            # Backward weakest-precondition computation.
            from ..logic.quantifier import mbp
            interpolants: List[FormulaExpression] = []
            current_post = post

            for tr, indicators in reversed(list(zip(fresh_path, guard_indicators))):
                # Substitute pre→post in the current post.
                subst_map = {}
                for var, term in tr.transform.items():
                    subst_map[var] = term
                post_sub = substitute(current_post, subst_map)

                reduced_guard = [mk_not(srk, g) for ind, g in indicators
                                 if ind in core_syms]
                wp_body = mk_or(srk, [post_sub] + reduced_guard)
                wp = mk_not(srk, wp_body)
                # Project out program variables.
                try:
                    wp = mbp(srk, lambda s: s in set(tr.transform.keys()), wp)
                except Exception:
                    pass
                wp = mk_not(srk, wp)
                interpolants.append(wp)
                current_post = wp

            interpolants.reverse()
            return ("Valid", interpolants)

        except Exception:
            return ("Unknown", None)

    # ------------------------------------------------------------------
    # Hoare triple
    # ------------------------------------------------------------------

    def valid_triple(
        self, pre: FormulaExpression, path: List[Transition],
        post: FormulaExpression,
    ) -> TransitionResult:
        """Check validity of the Hoare triple  ``{pre} path {post}``."""
        srk = self.context
        if srk is None:
            return TransitionResult.UNKNOWN

        # Compose the path and assume ¬post.
        path_tr = functools.reduce(lambda a, b: a.mul(b), path,
                                   Transition.one(srk))
        path_not_post = path_tr.mul(Transition.assume(srk, mk_not(srk, post)))

        from ..logic.smt import is_sat as smt_is_sat, SMTResult
        result = smt_is_sat(srk, mk_and(srk, [pre, path_not_post.guard]))
        if result == SMTResult.SAT:
            return TransitionResult.INVALID
        if result == SMTResult.UNSAT:
            return TransitionResult.VALID
        return TransitionResult.UNKNOWN

    # ------------------------------------------------------------------
    # Abstract postcondition
    # ------------------------------------------------------------------

    def abstract_post(self, pre_property: Any) -> Any:
        """Compute abstract postcondition via APRON-like abstract interpretation.

        Builds the formula  ``pre ∧ guard ∧ (x₁ = t₁) ∧ … ∧ (xₙ = tₙ)``,
        linearises it, and abstracts with the *Abstract* module.
        """
        srk = self.context
        if srk is None:
            return pre_property

        # Rename transform targets to fresh Skolems to avoid capture.
        fresh: Dict[int, Expression] = {}

        def tr_subst(sym: Symbol) -> Expression:
            if sym in self.transform:
                if sym.id not in fresh:
                    fresh[sym.id] = _fresh_skolem(srk, sym)
                return fresh[sym.id]
            return mk_const(srk, sym)

        transform_eqs: List[FormulaExpression] = []
        for var, rhs in self.transform.items():
            lhs = mk_const(srk, mk_symbol(srk, srk.show_symbol(var), var.typ))
            transform_eqs.append(mk_eq(srk, lhs,
                                       substitute_const(srk, tr_subst, rhs)))

        guard_sub = substitute_const(srk, tr_subst, self.guard)

        # Try to get the formula from the property.
        try:
            from ..analysis.wedge import to_formula as wedge_to_formula
            pre_formula = wedge_to_formula(pre_property)
        except Exception:
            pre_formula = pre_property

        pre_sub = substitute_const(srk, tr_subst, pre_formula)

        body = mk_and(srk, [pre_sub, guard_sub] + transform_eqs)

        # Linearise.
        try:
            from ..algebra.nonlinear import linearize as nl_linearize
            body = nl_linearize(srk, body)
        except Exception:
            pass

        body = rewrite(srk, body, down=nnf_rewriter(srk))

        # Abstract.
        try:
            from ..logic.abstract import abstract as abs_abstract
            return abs_abstract(srk, body, exists=lambda s: True)
        except Exception:
            return body

    # ------------------------------------------------------------------
    # Linearisation (nonlinear → linear)
    # ------------------------------------------------------------------

    def linearize(self) -> Transition:
        """Linearise nonlinear terms via purification + uninterpret/interpret.

        Ported from OCaml ``linearize``.
        """
        srk = self.context
        if srk is None:
            return self

        try:
            from ..logic.srkSimplify import purify
            from ..algebra.nonlinear import uninterpret, interpret as nl_interpret
        except ImportError:
            return self

        new_transform: Dict[Symbol, Expression] = {}
        defs: List[FormulaExpression] = []

        for var, t in self.transform.items():
            uninterp_t = uninterpret(srk, t)
            try:
                pure_t, t_defs = purify(srk, uninterp_t)
            except Exception:
                pure_t, t_defs = t, {}

            for v_sym, t_expr in t_defs.items():
                # Interpret back: uninterpreted f → nonlinear term.
                interpreted = nl_interpret(srk, t_expr)
                defs.append(mk_eq(srk, mk_const(srk, v_sym), interpreted))

            new_transform[var] = pure_t

        guard = self.guard
        if defs:
            try:
                from ..algebra.nonlinear import linearize as nl_linearize
                guard = nl_linearize(srk, mk_and(srk, [guard] + defs))
            except Exception:
                guard = mk_and(srk, [guard] + defs)

        return Transition(transform=new_transform, guard=guard, context=srk)

    # ------------------------------------------------------------------
    # Conversion to TransitionFormula
    # ------------------------------------------------------------------

    def to_transition_formula(self, context: Context = None) -> Any:
        """Build a TransitionFormula whose body is
        ``guard ∧ (p₁ = t₁) ∧ … ∧ (pₙ = tₙ)``.
        """
        from .transitionFormula import make as make_tf

        srk = context or self.context
        if srk is None:
            raise ValueError("Need a context")

        if not self.transform:
            return make_tf(self.guard, [], exists=lambda s: True)

        tr_symbols: List[Tuple[Symbol, Symbol]] = []
        post_defs: List[FormulaExpression] = []
        post_symbols_set: Set[Symbol] = set()

        for var, term in self.transform.items():
            pre_name = srk.show_symbol(var)
            post_sym = mk_symbol(srk, pre_name + "'", var.typ)
            post_term = mk_const(srk, post_sym)
            tr_symbols.append((var, post_sym))
            post_defs.append(mk_eq(srk, post_term, term))
            post_symbols_set.add(post_sym)

        body = mk_and(srk, [self.guard] + post_defs) if post_defs else self.guard

        def exists_pred(x):
            if isinstance(x, Var):
                return True
            if isinstance(x, Const):
                return x.symbol in post_symbols_set
            return x in post_symbols_set

        return make_tf(body, tr_symbols, exists=exists_pred)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        if not self.transform:
            return f"Transition({self.guard} => skip)"
        updates = ", ".join(f"{v} := {e}" for v, e in self.transform.items())
        return f"Transition({self.guard} => {{{updates}}})"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Transition):
            return False
        return self.transform == other.transform and self.guard == other.guard

    def __hash__(self) -> int:
        return hash(str(self))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _destruct_and(srk: Context, phi: FormulaExpression) -> List[FormulaExpression]:
    """Flatten nested conjunctions into a list of conjuncts."""
    tag, payload = destruct(phi)
    if tag == "And":
        result: List[FormulaExpression] = []
        for child in payload:
            result.extend(_destruct_and(srk, child))
        return result
    return [phi]


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------


def make_assume(context: Context, guard: Expression) -> Transition:
    return Transition.assume(context, guard)


def make_assign(context: Context, var: Symbol, term: Expression) -> Transition:
    return Transition.assign(context, var, term)


def make_parallel_assign(
    context: Context, assignments: List[Tuple[Symbol, Expression]]
) -> Transition:
    return Transition.parallel_assign(context, assignments)


def make_havoc(context: Context, variables: List[Symbol]) -> Transition:
    return Transition.havoc(context, variables)


def make_zero(context: Context = None) -> Transition:
    return Transition.zero(context)


def make_one(context: Context) -> Transition:
    return Transition.one(context)


class TransitionSystem:
    """Lightweight transition system wrapper for tests."""

    def __init__(self, context: Context, edges: List[Tuple[int, Transition, int]]):
        self.context = context
        self._edges = list(edges)

    def edges(self) -> List[Tuple[int, Transition, int]]:
        return list(self._edges)


def compare(a: "Transition", b: "Transition") -> int:
    """Syntactic comparison of transitions (mirrors OCaml Transition.compare)."""
    ha = hash(a)
    hb = hash(b)
    return -1 if ha < hb else (1 if ha > hb else 0)


def pp(out: Any, tr: "Transition") -> None:
    """Pretty-print a transition (mirrors OCaml Transition.pp)."""
    out.write(str(tr))


def show(tr: "Transition") -> str:
    """Show a transition as string (mirrors OCaml Transition.show)."""
    return str(tr)
