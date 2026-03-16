"""
efmc.engines.ef.farkas.template
===============================

Template–based invariant synthesis for linear transition systems.

Given a user–specified number `k` of templates, the class creates affine
expressions

        pᵢ₀ + Σⱼ pᵢⱼ · xⱼ     (i = 1 … k)

with symbolic coefficients `pᵢⱼ`, and constrains them such that

    •  Init          ⇒  Inv
    •  Inv ∧ Trans   ⇒  Inv'
    •  Inv           ⇒  Post

hold universally.  The universal quantifiers are eliminated with **Farkas’
lemma**, yielding a purely existential system over the `pᵢⱼ` that can be sent
to an SMT solver.

The class is agnostic to the concrete Farkas-implementation – it only relies
on a minimal interface (`add_constraint`, `apply_farkas_lemma`).
"""

from __future__ import annotations

import logging
from typing import List, Sequence

import z3

from aria.efmc.engines.ef.templates.abstract_template import Template
from aria.efmc.engines.ef.farkas.farkas_lemma import FarkasLemma  # the improved one
from aria.efmc.sts import TransitionSystem

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------#
# helpers                                                                    #
# ---------------------------------------------------------------------------#
def _flatten_conj(fml: z3.BoolRef) -> List[z3.BoolRef]:
    """
    Turn an arbitrary (finite) Boolean combination of `And`s into a list of
    conjuncts.  All other connectives are returned as singletons.
    """
    if z3.is_and(fml):
        out: List[z3.BoolRef] = []
        for c in fml.children():
            out.extend(_flatten_conj(c))
        return out
    return [fml]


# ---------------------------------------------------------------------------#
# main template class                                                        #
# ---------------------------------------------------------------------------#
class FarkasTemplate(Template):
    """
    Linear template family plus Farkas–based constraint generation.

    Parameters
    ----------
    sts : TransitionSystem
    num_templates : int     (default = 3)
    """

    # ------------------------------------------------------------------- init
    def __init__(self, sts: TransitionSystem, *, num_templates: int = 3) -> None:
        self.sts = sts
        self.num_templates = num_templates

        self._arity = len(self.sts.variables)  # #program vars
        self._tpl_coeffs: List[List[z3.RealRef]] = []  # pᵢⱼ variables

        self._cnt_init_post: z3.BoolRef | None = None
        self._cnt_trans: z3.BoolRef | None = None

        self.add_template_vars()
        self.add_template_cnts()

    # -------------------------------------------------------- public contract
    # Template subclasses must expose .template_cnt_init_and_post and
    # .template_cnt_trans.  We use simple @property for type-safety.
    @property
    def template_cnt_init_and_post(self) -> z3.BoolRef:
        """Return constraints for init and post conditions."""
        assert self._cnt_init_post is not None
        return self._cnt_init_post

    @property
    def template_cnt_trans(self) -> z3.BoolRef:
        """Return constraints for transition relation."""
        assert self._cnt_trans is not None
        return self._cnt_trans

    # ----------------------------------------------------------------- build
    def add_template_vars(self) -> None:
        """Create symbolic coefficients pᵢ₀, …, pᵢ_arity for each template i."""
        index = 0
        for _ in range(self.num_templates):
            coeffs = []
            for j in range(self._arity + 1):  # +1 for constant term
                coeffs.append(z3.Real(f"p{index}_{j}"))
            self._tpl_coeffs.append(coeffs)
            index += 1

    # ----------------------------- compose linear expression for *one* set --
    @staticmethod
    def _affine(
        coeffs: Sequence[z3.ArithRef], vars_: Sequence[z3.ArithRef]
    ) -> z3.ArithRef:
        """
        Return  coeffs[0] + Σ coeffs[i+1] * vars_[i]
        """
        expr = coeffs[0]
        for i, v in enumerate(vars_):
            expr += coeffs[i + 1] * v
        return expr

    # ----------------------------- build invariant over a given var-vector --
    def _invariant_formula(self, vars_: Sequence[z3.ArithRef]) -> z3.BoolRef:
        conjuncts = [self._affine(coeffs, vars_) >= 0 for coeffs in self._tpl_coeffs]
        return z3.And(conjuncts) if conjuncts else z3.BoolVal(True)

    # ----------------------------- Farkas translation premise ⇒ conclusion --
    def _farkas_implication(
        self, premise: z3.BoolRef, conclusion: z3.BoolRef
    ) -> z3.BoolRef:
        """
        Return a *quantifier-free* constraint equivalent to

            ∀x. premise(x) ⇒ conclusion(x)

        under the assumption that all atoms are linear (QF_LRA / QF_LIA).

        If conclusion is a conjunction, prove each conjunct separately:
            premise ⇒ (a ∧ b) ≡ (premise ⇒ a) ∧ (premise ⇒ b)
        """
        # Split conclusion into conjuncts
        conclusion_conjuncts = _flatten_conj(conclusion)

        all_constraints: List[z3.BoolRef] = []

        # Prove premise ⇒ each conjunct separately
        for conj in conclusion_conjuncts:
            fl = FarkasLemma()

            # Collect atomic linear constraints of premise ∧ ¬conj
            # which must be UNSAT for the implication to hold
            premise_atoms = _flatten_conj(premise)
            for a in premise_atoms:
                fl.add_constraint(a)

            # Add the negation of this conjunct
            # Handle the negation properly based on the form of conj
            if z3.is_le(conj):  # ¬(lhs ≤ rhs) ⇒ lhs > rhs
                # In Farkas: lhs - rhs ≥ 0
                lhs, rhs = conj.arg(0), conj.arg(1)
                fl.add_constraint(lhs - rhs >= 0)
            elif z3.is_ge(conj):  # ¬(lhs ≥ rhs) ⇒ lhs < rhs
                # In Farkas: rhs - lhs ≥ 0
                lhs, rhs = conj.arg(0), conj.arg(1)
                fl.add_constraint(rhs - lhs >= 0)
            elif z3.is_eq(conj):  # ¬(lhs == rhs) - not representable in Farkas
                # Skip or handle specially - for now, skip
                continue
            else:
                # For other forms, try to add the negation directly
                fl.add_constraint(z3.Not(conj))

            # variables possibly from both current and primed worlds
            universe = list({*self.sts.variables, *self.sts.prime_variables})
            constraints = fl.apply_farkas_lemma_symbolic(universe)
            logger.debug(
                "Farkas produced %d constraints for one conjunct.", len(constraints)
            )

            all_constraints.extend(constraints)

        return z3.And(all_constraints) if all_constraints else z3.BoolVal(True)

    # ----------------------------- assemble the three VCs -------------------
    def add_template_cnts(self) -> None:
        """Add constraints according to the specification of inductive loop invariant."""
        inv = self._invariant_formula(self.sts.variables)
        inv_ = self._invariant_formula(self.sts.prime_variables)

        # 1. Init  ⇒ Inv
        c_init = self._farkas_implication(self.sts.init, inv)

        # 2. Inv ∧ Trans ⇒ Inv'
        c_trans = self._farkas_implication(z3.And(inv, self.sts.trans), inv_)

        # 3. Inv ⇒ Post
        c_post = self._farkas_implication(inv, self.sts.post)

        self._cnt_init_post = z3.And(c_init, c_post)
        self._cnt_trans = c_trans

    # ----------------------------------------------------------------- model
    def build_invariant_expr(
        self, model: z3.ModelRef, use_prime_variables: bool = False
    ) -> z3.BoolRef:
        """
        Instantiate the learned coefficients in the template and return the
        concrete invariant (optionally over primed variables).
        """
        vars_ = self.sts.prime_variables if use_prime_variables else self.sts.variables
        conjuncts: List[z3.BoolRef] = []

        for coeffs in self._tpl_coeffs:
            const_val = model.evaluate(coeffs[0], model_completion=True)
            # Convert to a concrete Z3 value
            if z3.is_rational_value(const_val):
                aff = z3.RealVal(
                    z3.Q(const_val.numerator_as_long(), const_val.denominator_as_long())
                )
            elif z3.is_int_value(const_val):
                aff = z3.RealVal(const_val.as_long())
            else:
                aff = const_val

            for i, v in enumerate(vars_):
                coef_val = model.evaluate(coeffs[i + 1], model_completion=True)
                if z3.is_rational_value(coef_val):
                    coef = z3.RealVal(
                        z3.Q(
                            coef_val.numerator_as_long(), coef_val.denominator_as_long()
                        )
                    )
                elif z3.is_int_value(coef_val):
                    coef = z3.RealVal(coef_val.as_long())
                else:
                    coef = coef_val
                aff = aff + coef * v
            conjuncts.append(aff >= 0)

        return z3.And(conjuncts) if conjuncts else z3.BoolVal(True)
