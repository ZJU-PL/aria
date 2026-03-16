"""
HyperLTL-like bounded encoding and prover for k-safety/hyperproperties.

This module provides a minimal DSL to express relational temporal properties
across k traces, and a prover that encodes them over a bounded unrolling.

Supported temporal operators (bounded semantics):
  - G (globally), F (eventually), X (next), U (until)

Quantification: we handle universal quantification over a fixed number of traces k
by checking existence of a counterexample with our solver (standard safety style).

Example:
  # Non-interference as G(low1 == low2) under G(high1 == high2)
  phi = Implies(G(Eq(Var("high",0), Var("high",1))), G(Eq(Var("low",0), Var("low",1))))
  prover = HyperLTLProver(sts, k=2, method="bounded_model_checking", max_bound=10)
  prover.set_formula(phi)
  res = prover.solve()
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import z3

from aria.efmc.utils.verification_utils import VerificationResult
from .base_prover import BaseKSafetyProver

logger = logging.getLogger(__name__)


# ---- DSL Terms and Formulas -------------------------------------------------


@dataclass(frozen=True)
class Var:
    """A reference to a state variable name on a given trace."""

    name: str
    trace: int


@dataclass(frozen=True)
class IntConst:
    """Integer constant."""

    value: int


@dataclass(frozen=True)
class RealConst:
    """Real constant."""

    value: float


Term = Union[Var, IntConst, RealConst]


@dataclass(frozen=True)
class Atom:
    """Atomic formula with comparison operator."""

    op: str  # '==','!=','<=','<','>=','>'
    left: Term
    right: Term


@dataclass(frozen=True)
class Not:
    """Negation."""

    phi: "HProp"


@dataclass(frozen=True)
class And:
    """Conjunction."""

    conjuncts: List["HProp"]


@dataclass(frozen=True)
class Or:
    """Disjunction."""

    disjuncts: List["HProp"]


@dataclass(frozen=True)
class Implies:
    """Implication."""

    left: "HProp"
    right: "HProp"


@dataclass(frozen=True)
class X:
    """Next temporal operator."""

    phi: "HProp"


@dataclass(frozen=True)
class G:
    """Globally temporal operator."""

    phi: "HProp"


@dataclass(frozen=True)
class F:
    """Eventually temporal operator."""

    phi: "HProp"


@dataclass(frozen=True)
class U:
    """Until temporal operator."""

    left: "HProp"
    right: "HProp"


HProp = Union[Atom, Not, And, Or, Implies, X, G, F, U]


def _to_z3_term(sts, step_vars, var_index_cache: Dict[str, int], term: Term, step: int):
    if isinstance(term, Var):
        idx = var_index_cache.get(term.name)
        if idx is None:
            # build cache lazily
            for i, v in enumerate(sts.variables):
                if str(v) == term.name:
                    idx = i
                    var_index_cache[term.name] = i
                    break
        if idx is None:
            raise ValueError(f"Variable '{term.name}' not found in transition system")
        return step_vars[term.trace][step][idx]
    if isinstance(term, IntConst):
        return z3.IntVal(term.value)
    if isinstance(term, RealConst):
        return z3.RealVal(term.value)
    raise NotImplementedError(f"Unsupported term: {term}")


def _encode_atom(sts, step_vars, cache, atom: Atom, step: int):
    l = _to_z3_term(sts, step_vars, cache, atom.left, step)
    r = _to_z3_term(sts, step_vars, cache, atom.right, step)
    if atom.op == "==":
        return l == r
    if atom.op == "!=":
        return l != r
    if atom.op == "<=":
        return l <= r
    if atom.op == "<":
        return l < r
    if atom.op == ">=":
        return l >= r
    if atom.op == ">":
        return l > r
    raise NotImplementedError(f"Unsupported atomic op: {atom.op}")


def _encode(sts, step_vars, cache, phi: HProp, step: int, bound: int) -> z3.ExprRef:
    if isinstance(phi, Atom):
        return _encode_atom(sts, step_vars, cache, phi, step)
    if isinstance(phi, Not):
        return z3.Not(_encode(sts, step_vars, cache, phi.phi, step, bound))
    if isinstance(phi, And):
        return z3.And(
            *[_encode(sts, step_vars, cache, c, step, bound) for c in phi.conjuncts]
        )
    if isinstance(phi, Or):
        return z3.Or(
            *[_encode(sts, step_vars, cache, d, step, bound) for d in phi.disjuncts]
        )
    if isinstance(phi, Implies):
        return z3.Implies(
            _encode(sts, step_vars, cache, phi.left, step, bound),
            _encode(sts, step_vars, cache, phi.right, step, bound),
        )
    if isinstance(phi, X):
        next_step = min(step + 1, bound)
        return _encode(sts, step_vars, cache, phi.phi, next_step, bound)
    if isinstance(phi, G):
        return z3.And(
            *[
                _encode(sts, step_vars, cache, phi.phi, t, bound)
                for t in range(step, bound + 1)
            ]
        )
    if isinstance(phi, F):
        return z3.Or(
            *[
                _encode(sts, step_vars, cache, phi.phi, t, bound)
                for t in range(step, bound + 1)
            ]
        )
    if isinstance(phi, U):
        ors = []
        for j in range(step, bound + 1):
            left_ands = [
                _encode(sts, step_vars, cache, phi.left, i, bound)
                for i in range(step, j)
            ]
            ors.append(
                z3.And(
                    *(left_ands + [_encode(sts, step_vars, cache, phi.right, j, bound)])
                )
            )
        return z3.Or(*ors) if ors else z3.BoolVal(False)
    raise NotImplementedError(f"Unsupported formula type: {type(phi)}")


# ---- Prover ---------------------------------------------------------------


class HyperLTLProver(BaseKSafetyProver):
    """Bounded HyperLTL-like prover over k traces."""

    def __init__(self, sts, k: int = 2, **kwargs):
        super().__init__(sts, k=k, **kwargs)
        self.formula: Optional[HProp] = None
        self.logger.info("Initialized HyperLTL prover")

    def set_formula(self, formula: HProp) -> None:
        self.formula = formula

    def _build_unrolled_antecedent(self, bound: int):
        """Build unrolled antecedent for bounded model checking."""
        # Create per-trace per-step variables using Base helper
        step_vars = self._make_step_variables(bound)
        conditions = []
        for trace_idx in range(self.k):
            # init at step 0
            init_subst = [
                (self.sts.variables[i], step_vars[trace_idx][0][i])
                for i, _ in enumerate(self.sts.variables)
            ]
            conditions.append(z3.substitute(self.sts.init, init_subst))
            # invariants 0..bound
            if getattr(self.sts, "invariants", None):
                for step in range(bound + 1):
                    inv_subst = [
                        (self.sts.variables[i], step_vars[trace_idx][step][i])
                        for i, _ in enumerate(self.sts.variables)
                    ]
                    for inv in self.sts.invariants:
                        conditions.append(z3.substitute(inv, inv_subst))
            # transitions 0..bound-1
            for step in range(bound):
                trans_subst = []
                for i, _ in enumerate(self.sts.variables):
                    trans_subst.append(
                        (self.sts.variables[i], step_vars[trace_idx][step][i])
                    )
                    trans_subst.append(
                        (self.sts.prime_variables[i], step_vars[trace_idx][step + 1][i])
                    )
                conditions.append(z3.substitute(self.sts.trans, trans_subst))
        antecedent = z3.And(*conditions) if conditions else z3.BoolVal(True)
        return antecedent, step_vars

    def bounded_model_checking(self, bound: int) -> VerificationResult:
        if self.formula is None:
            raise ValueError("HyperLTL formula must be set before solving")
        self.logger.info("Performing HyperLTL BMC with bound %s", bound)
        antecedent, step_vars = self._build_unrolled_antecedent(bound)
        cache: Dict[str, int] = {}
        prop = _encode(self.sts, step_vars, cache, self.formula, step=0, bound=bound)
        formula = z3.Implies(antecedent, prop)
        solver = z3.Solver()
        solver.add(z3.Not(formula))
        solver.set("timeout", self.timeout * 1000)
        res = solver.check()
        if res == z3.sat:
            return VerificationResult(False, None, solver.model(), is_unsafe=True)
        if res == z3.unsat:
            return VerificationResult(True, None)
        return VerificationResult(False, None, None, is_unknown=True, timed_out=True)
