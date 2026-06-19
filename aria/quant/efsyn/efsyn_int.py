"""Linear Int/Real exists-forall synthesis procedure.

Compared to efsyn_simple.py:

It keeps the dual-memory structure, but the synthesis steps are more concrete:

- $Y \to X$: weighted MaxSMT coverage **plus slack maximization**
- $X \to Y$: weighted breaking **plus violation maximization**
- counterexamples are generalized into **polyhedral guards** from linear atoms

So the linear version doesn't just ask "can I satisfy these sampled attacks?" — it asks:

> `can I satisfy them with margin?`

and on the attack side:

> `can I violate many strong witnesses with large violation?`
"""

from __future__ import annotations

from typing import List, Optional

from z3 import *

from aria.quant.efsyn.efsyn_bv import BVExistsForallCEGIS

try:
    from .efsyn_common import (
        Assignment,
        Attack,
        Candidate,
        DualMemoryCEGISBase,
        expr_mentions_any,
        extract_atoms,
        flatten_and,
        is_cmp_atom,
        is_int_sort,
        is_real_sort,
        one_for_sort,
        sum_expr,
        zero_for_sort,
    )
except ImportError:  # pragma: no cover - direct script/module execution fallback
    from efsyn_common import (
        Assignment,
        Attack,
        Candidate,
        DualMemoryCEGISBase,
        expr_mentions_any,
        extract_atoms,
        flatten_and,
        is_cmp_atom,
        is_int_sort,
        is_real_sort,
        one_for_sort,
        sum_expr,
        zero_for_sort,
    )


# =============================================================================
# 2) Linear arithmetic optimized version
# =============================================================================

class LinearExistsForallCEGIS(DualMemoryCEGISBase):
    """
    Linear arithmetic specialization.

    Best for conjunction-heavy linear predicates/domains.

    Main specialization:
      - exact checks use SolverFor("QF_LIA") for all-Int,
        SolverFor("QF_LRA") for all-Real, generic Solver otherwise.
      - witness synthesis adds a slack-maximization objective
      - attack synthesis adds a violation-maximization objective
      - counterexamples generalize to polyhedral guards from linear atoms
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for v in self.x_vars + self.y_vars:
            if not (is_int_sort(v.sort()) or is_real_sort(v.sort())):
                raise TypeError(
                    "LinearExistsForallCEGIS requires all X and Y variables to be Int or Real."
                )

    def _all_int(self) -> bool:
        return all(is_int_sort(v.sort()) for v in self.x_vars + self.y_vars)

    def _all_real(self) -> bool:
        return all(is_real_sort(v.sort()) for v in self.x_vars + self.y_vars)

    def _new_solver(self) -> Solver:
        if self._all_int():
            s = SolverFor("QF_LIA")
        elif self._all_real():
            s = SolverFor("QF_LRA")
        else:
            s = Solver()
        s.set(timeout=self.timeout_ms)
        return s

    def _strict_step(self, lhs: ArithRef, rhs: ArithRef) -> ArithRef:
        if is_int_sort(lhs.sort()) and is_int_sort(rhs.sort()):
            return IntVal(1)
        sort = lhs.sort() if is_int_sort(lhs.sort()) or is_real_sort(lhs.sort()) else rhs.sort()
        return zero_for_sort(sort)

    def _margin_expr(self, atom: BoolRef) -> Optional[ArithRef]:
        if not is_cmp_atom(atom):
            return None
        lhs, rhs = atom.arg(0), atom.arg(1)
        k = atom.decl().kind()

        if k == Z3_OP_LE:
            return rhs - lhs
        if k == Z3_OP_LT:
            return rhs - lhs - self._strict_step(lhs, rhs)
        if k == Z3_OP_GE:
            return lhs - rhs
        if k == Z3_OP_GT:
            return lhs - rhs - self._strict_step(lhs, rhs)
        if k == Z3_OP_EQ:
            return None
        return None

    def _violation_expr(self, atom: BoolRef) -> Optional[ArithRef]:
        if not is_cmp_atom(atom):
            return None
        lhs, rhs = atom.arg(0), atom.arg(1)
        k = atom.decl().kind()

        if k == Z3_OP_LE:
            return lhs - rhs
        if k == Z3_OP_LT:
            return lhs - rhs + self._strict_step(lhs, rhs)
        if k == Z3_OP_GE:
            return rhs - lhs
        if k == Z3_OP_GT:
            return rhs - lhs + self._strict_step(lhs, rhs)
        if k == Z3_OP_EQ:
            # Piecewise-linear fallback score
            z = zero_for_sort(lhs.sort())
            o = one_for_sort(lhs.sort())
            return If(lhs == rhs, z, o)
        return None

    def _positive_part(self, e: ArithRef) -> ArithRef:
        z = zero_for_sort(e.sort())
        return If(e > z, e, z)

    def _comparison_atoms_of_formula(self, formula: BoolRef) -> List[BoolRef]:
        atoms = []
        for a in flatten_and(formula):
            if is_cmp_atom(a):
                atoms.append(a)
        return atoms

    def _slack_score(self, formula: BoolRef) -> Optional[ArithRef]:
        terms: List[ArithRef] = []
        for atom in self._comparison_atoms_of_formula(formula):
            m = self._margin_expr(atom)
            if m is not None:
                z = zero_for_sort(m.sort())
                terms.append(If(atom, m, z))
        if not terms:
            return None
        return sum_expr(terms)

    def _violation_score(self, formula: BoolRef) -> Optional[ArithRef]:
        terms: List[ArithRef] = []
        for atom in self._comparison_atoms_of_formula(formula):
            v = self._violation_expr(atom)
            if v is not None:
                terms.append(self._positive_part(v))
        if not terms:
            return None
        return sum_expr(terms)

    def _basis_for_generalization(
        self,
        cand: Candidate,
        y_vals: Assignment,
        fail_formula: BoolRef,
    ) -> List[BoolRef]:
        if self.region_basis:
            raw_basis = [self._instantiate(a, self.x_vars, cand.vals) for a in self.region_basis]
        else:
            raw_basis = extract_atoms(fail_formula)

        basis: List[BoolRef] = []
        seen = set()

        # Prefer y-only arithmetic comparison atoms.
        for atom in raw_basis:
            atom = simplify(atom)
            if not is_cmp_atom(atom):
                continue
            if expr_mentions_any(atom, self.x_vars):
                continue
            if not expr_mentions_any(atom, self.y_vars):
                continue
            key = atom.sexpr()
            if key not in seen:
                basis.append(atom)
                seen.add(key)
            if len(basis) >= self.max_region_atoms:
                return basis

        # Fallback: any y-only boolean atom
        for atom in raw_basis:
            atom = simplify(atom)
            if not is_bool(atom):
                continue
            if expr_mentions_any(atom, self.x_vars):
                continue
            if not expr_mentions_any(atom, self.y_vars):
                continue
            key = atom.sexpr()
            if key not in seen:
                basis.append(atom)
                seen.add(key)
            if len(basis) >= self.max_region_atoms:
                break

        return basis

    def _synthesize_x(
        self,
        *,
        bundle: List[Attack],
        template: BoolRef,
        template_id: int,
        blocked: List[Assignment],
    ) -> Optional[Candidate]:
        opt = self._new_optimize()
        opt.add(self.domain_x, template)

        for vals in [c.vals for c in self.M_X] + list(blocked):
            neq = self._assignment_neq_clause(self.x_vars, vals)
            if not is_false(simplify(neq)):
                opt.add(neq)

        slack_terms: List[ArithRef] = []

        for atk in bundle:
            phi = self._instantiate(self.predicate, self.y_vars, atk.rep)
            weight = max(1, int(round(1 + 9 * atk.score)))
            opt.add_soft(phi, weight=str(weight), id="cover")

            slack = self._slack_score(phi)
            if slack is not None:
                slack_terms.append(weight * slack)
            else:
                slack_terms.append(If(phi, IntVal(weight), IntVal(0)))

        self._add_x_novelty_soft(opt, self.M_X[-4:])

        if slack_terms:
            opt.maximize(sum_expr(slack_terms))

        res = opt.check()
        if res == sat:
            vals = self._assignment_from_model(opt.model(), self.x_vars)
            return Candidate(vals=vals, source="y->x:slack-maxsat", template_id=template_id)

        s = self._new_solver()
        s.add(self.domain_x, template)

        for vals in [c.vals for c in self.M_X] + list(blocked):
            neq = self._assignment_neq_clause(self.x_vars, vals)
            if not is_false(simplify(neq)):
                s.add(neq)

        hard_subset = sorted(bundle, key=lambda a: (-a.score, -a.power, -a.uid))
        hard_subset = hard_subset[:max(1, len(hard_subset) // 2)] if hard_subset else []

        for atk in hard_subset:
            s.add(self._instantiate(self.predicate, self.y_vars, atk.rep))

        if s.check() == sat:
            vals = self._assignment_from_model(s.model(), self.x_vars)
            return Candidate(vals=vals, source="y->x:fallback", template_id=template_id)

        return None

    def _synthesize_y(
        self,
        *,
        cluster: List[Candidate],
        focus_guard: BoolRef,
        blocked: List[Assignment],
    ) -> Optional[Attack]:
        if not cluster:
            return None

        opt = self._new_optimize()
        opt.add(self.domain_y)
        if not is_true(simplify(focus_guard)):
            opt.add(focus_guard)

        for vals in [a.rep for a in self.M_Y] + list(blocked):
            neq = self._assignment_neq_clause(self.y_vars, vals)
            if not is_false(simplify(neq)):
                opt.add(neq)

        violation_terms: List[ArithRef] = []

        for cand in cluster:
            bad = Not(self._instantiate(self.predicate, self.x_vars, cand.vals))
            weight = max(1, int(round(1 + 9 * cand.coverage)))
            opt.add_soft(bad, weight=str(weight), id="break")

            phi = self._instantiate(self.predicate, self.x_vars, cand.vals)
            vio = self._violation_score(phi)
            if vio is not None:
                violation_terms.append(weight * vio)
            else:
                violation_terms.append(If(Not(phi), IntVal(weight), IntVal(0)))

        self._add_y_novelty_soft(opt, self.M_Y[-4:])

        if violation_terms:
            opt.maximize(sum_expr(violation_terms))

        if opt.check() == sat:
            y_vals = self._assignment_from_model(opt.model(), self.y_vars)
            broken = [c for c in cluster if not self._predicate_holds(c.vals, y_vals)]
            if broken:
                pivot = max(broken, key=lambda c: (c.score, c.coverage, c.uid))
                guard = self._generalize_failure(pivot, y_vals)
                return Attack(rep=y_vals, guard=guard, source="x->y:violation-maxsat")

        strongest = max(cluster, key=lambda c: (c.score, c.coverage, c.uid))
        s = self._new_solver()
        s.add(self.domain_y)
        if not is_true(simplify(focus_guard)):
            s.add(focus_guard)
        for vals in [a.rep for a in self.M_Y] + list(blocked):
            neq = self._assignment_neq_clause(self.y_vars, vals)
            if not is_false(simplify(neq)):
                s.add(neq)
        s.add(Not(self._instantiate(self.predicate, self.x_vars, strongest.vals)))

        if s.check() == sat:
            y_vals = self._assignment_from_model(s.model(), self.y_vars)
            guard = self._generalize_failure(strongest, y_vals)
            return Attack(rep=y_vals, guard=guard, source="x->y:fallback")

        return None


if __name__ == "__main__":

    print("=== Linear arithmetic example ===")
    #   exists x_lo, x_hi .
    #       0 <= x_lo <= x_hi <= 10
    #   and forall y in [0, 10] .
    #       x_lo <= y <= x_hi
    #
    # A valid witness is x_lo = 0, x_hi = 10.

    x_lo, x_hi = Ints("x_lo x_hi")
    yy = Int("yy")

    lin_solver = LinearExistsForallCEGIS(
        x_vars=[x_lo, x_hi],
        y_vars=[yy],
        predicate=And(x_lo <= yy, yy <= x_hi),
        domain_x=And(0 <= x_lo, x_lo <= x_hi, x_hi <= 10),
        domain_y=And(0 <= yy, yy <= 10),
        max_iters=20,
        max_x_memory=20,
        max_y_memory=20,
        verbose=True,
    )

    lin_result = lin_solver.solve()
    print("LIN status :", lin_result.status)
    print("LIN witness:", lin_result.witness)
    print("LIN iters  :", lin_result.iterations)
    print("LIN msg    :", lin_result.message)
