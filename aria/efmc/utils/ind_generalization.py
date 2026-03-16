"""Inductive Generalization

Counterexample-guided inductive generalization using CTI (counterexamples to induction).
Checks three conditions: initiation, inductiveness, and safety.
"""

import logging
from typing import Optional, Tuple

import z3

from aria.efmc.utils.z3_solver_utils import is_entail
from aria.efmc.sts import TransitionSystem

logger = logging.getLogger(__name__)


class InductiveGeneralizer:
    """Main class for inductive generalization"""

    def __init__(self, sts: TransitionSystem):
        self.sts = sts
        self.logger = logging.getLogger(__name__)
        self.timeout = 10000

    def set_timeout(self, timeout_ms: int):
        """Set timeout for SMT solver calls"""
        self.timeout = timeout_ms

    def check_invariant(
        self, inv: z3.ExprRef
    ) -> Tuple[bool, Optional[str], Optional[z3.ModelRef]]:
        """Check if invariant satisfies initiation, inductiveness, and safety"""
        # Check initiation: init => inv
        if not is_entail(self.sts.init, inv):
            s = z3.Solver()
            s.set("timeout", self.timeout)
            s.add(self.sts.init, z3.Not(inv))
            if s.check() == z3.sat:
                return False, "initiation", s.model()

        # Check inductiveness: inv && trans => inv'
        inv_prime = z3.substitute(
            inv, list(zip(self.sts.variables, self.sts.prime_variables))
        )
        if not is_entail(z3.And(inv, self.sts.trans), inv_prime):
            s = z3.Solver()
            s.set("timeout", self.timeout)
            s.add(inv, self.sts.trans, z3.Not(inv_prime))
            if s.check() == z3.sat:
                return False, "inductiveness", s.model()

        # Check safety: inv => post
        if not is_entail(inv, self.sts.post):
            s = z3.Solver()
            s.set("timeout", self.timeout)
            s.add(inv, z3.Not(self.sts.post))
            if s.check() == z3.sat:
                return False, "safety", s.model()

        return True, None, None

    def get_cti(self, inv: z3.ExprRef) -> Tuple[Optional[str], Optional[z3.ModelRef]]:
        """Get counterexample to induction (CTI)"""
        valid, condition, model = self.check_invariant(inv)
        if valid:
            return None, None
        return condition, model

    def strengthen_from_cti(self, inv: z3.ExprRef, cti: z3.ModelRef) -> z3.ExprRef:
        """Strengthen invariant by excluding counterexample state"""
        state_constraints = []
        for v in self.sts.variables:
            try:
                val = cti.eval(v)
                if val is not None:
                    state_constraints.append(v == val)
            except Exception:
                # Skip variables that don't have values in the model
                pass

        state = z3.And(*state_constraints) if state_constraints else z3.BoolVal(True)
        return z3.simplify(z3.And(inv, z3.Not(state)))

    def weaken_from_cti(self, inv: z3.ExprRef, cti: z3.ModelRef) -> z3.ExprRef:
        """Weaken invariant to include counterexample state"""
        state_constraints = []
        for v in self.sts.variables:
            try:
                val = cti.eval(v)
                if val is not None:
                    state_constraints.append(v == val)
            except Exception:
                # Skip variables that don't have values in the model
                pass

        state = z3.And(*state_constraints) if state_constraints else z3.BoolVal(True)
        return z3.simplify(z3.Or(inv, state))

    def generalize_by_dropping_literals(self, inv: z3.ExprRef) -> z3.ExprRef:
        """Generalize invariant by dropping unnecessary literals"""
        if inv.decl().kind() == z3.Z3_OP_AND:
            clauses = inv.children()
        else:
            clauses = [inv]

        necessary_clauses = []
        for i, clause in enumerate(clauses):
            remaining_clauses = clauses[:i] + clauses[i + 1 :]
            if not remaining_clauses:
                necessary_clauses.append(clause)
                continue

            candidate = (
                z3.And(*remaining_clauses)
                if len(remaining_clauses) > 1
                else remaining_clauses[0]
            )
            inv_prime = z3.substitute(
                candidate, list(zip(self.sts.variables, self.sts.prime_variables))
            )

            if is_entail(z3.And(candidate, self.sts.trans), inv_prime) and is_entail(
                candidate, self.sts.post
            ):
                return self.generalize_by_dropping_literals(candidate)
            necessary_clauses.append(clause)

        return (
            z3.And(*necessary_clauses)
            if len(necessary_clauses) > 1
            else necessary_clauses[0]
        )

    def generalize_by_craig_interpolation(
        self, expr_a: z3.ExprRef, expr_b: z3.ExprRef
    ) -> Optional[z3.ExprRef]:
        """Use Craig interpolation to find separating formula"""
        raise NotImplementedError

    def learn_invariant(self, max_iterations: int = 100) -> Optional[z3.ExprRef]:
        """Learn inductive invariant using counterexample-guided refinement"""
        inv = self.sts.post

        for _ in range(max_iterations):
            valid, condition, cti = self.check_invariant(inv)
            if valid:
                return inv

            if condition == "initiation":
                inv = self.weaken_from_cti(inv, cti)
            elif condition == "inductiveness":
                inv = self.strengthen_from_cti(inv, cti)
            elif condition == "safety":
                return None

            inv = self.generalize_by_dropping_literals(inv)

        return None
