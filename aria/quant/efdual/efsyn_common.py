"""Shared utilities and base classes for exists-forall synthesis CEGIS solvers."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

from z3 import *
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union


def is_bool_sort(sort: SortRef) -> bool:
    return sort.kind() == Z3_BOOL_SORT


def is_int_sort(sort: SortRef) -> bool:
    return sort.kind() == Z3_INT_SORT


def is_real_sort(sort: SortRef) -> bool:
    return sort.kind() == Z3_REAL_SORT


def is_bv_sort(sort: SortRef) -> bool:
    return sort.kind() == Z3_BV_SORT


# =============================================================================
# Shared utilities
# =============================================================================

Assignment = Tuple[ExprRef, ...]
AssignmentInput = Union[Sequence[object], Dict[object, object]]


def mk_and(parts: Iterable[BoolRef]) -> BoolRef:
    parts = list(parts)
    return And(*parts) if parts else BoolVal(True)


def mk_or(parts: Iterable[BoolRef]) -> BoolRef:
    parts = list(parts)
    return Or(*parts) if parts else BoolVal(False)


def flatten_and(expr: BoolRef) -> List[BoolRef]:
    expr = simplify(expr)
    if is_app_of(expr, Z3_OP_AND):
        out: List[BoolRef] = []
        for ch in expr.children():
            out.extend(flatten_and(ch))
        return out
    return [expr]


def z3_value_to_python(v: ExprRef):
    try:
        if is_true(v):
            return True
        if is_false(v):
            return False
        if is_int_value(v):
            return v.as_long()
        if is_rational_value(v):
            try:
                return v.as_fraction()
            except Exception:
                return str(v)
        if is_bv_value(v):
            return v.as_long()
        if hasattr(v, "as_long"):
            return v.as_long()
    except Exception:
        pass
    return str(v)


def expr_mentions_any(expr: ExprRef, vars_: Sequence[ExprRef]) -> bool:
    if not vars_:
        return False
    target_ids = {v.get_id() for v in vars_}
    seen = set()
    stack = [expr]
    while stack:
        e = stack.pop()
        h = e.hash()
        if h in seen:
            continue
        seen.add(h)
        if is_const(e) and e.get_id() in target_ids:
            return True
        stack.extend(e.children())
    return False


def extract_atoms(expr: BoolRef) -> List[BoolRef]:
    connectives = {
        Z3_OP_AND,
        Z3_OP_OR,
        Z3_OP_NOT,
        Z3_OP_IMPLIES,
        Z3_OP_IFF,
        Z3_OP_XOR,
        Z3_OP_ITE,
    }
    atoms: Dict[str, BoolRef] = {}
    seen = set()

    def rec(e: ExprRef) -> None:
        h = e.hash()
        if h in seen:
            return
        seen.add(h)

        if is_true(e) or is_false(e) or is_quantifier(e):
            return

        if is_app(e):
            k = e.decl().kind()
            if k in connectives:
                for ch in e.children():
                    rec(ch)
                return
            if is_bool(e):
                atoms.setdefault(e.sexpr(), e)
                return

        for ch in e.children():
            rec(ch)

    rec(simplify(expr))
    return list(atoms.values())


def is_cmp_atom(atom: ExprRef) -> bool:
    if not is_bool(atom) or not is_app(atom):
        return False
    return atom.decl().kind() in {Z3_OP_LE, Z3_OP_LT, Z3_OP_GE, Z3_OP_GT, Z3_OP_EQ}


def sum_expr(exprs: Sequence[ArithRef]) -> ArithRef:
    if not exprs:
        return IntVal(0)
    if len(exprs) == 1:
        return exprs[0]
    return Sum(*exprs)


def zero_for_sort(sort: SortRef) -> ArithRef:
    if is_real_sort(sort):
        return RealVal("0")
    if is_int_sort(sort):
        return IntVal(0)
    raise TypeError(f"Expected Int/Real sort, got {sort}")


def one_for_sort(sort: SortRef) -> ArithRef:
    if is_real_sort(sort):
        return RealVal("1")
    if is_int_sort(sort):
        return IntVal(1)
    raise TypeError(f"Expected Int/Real sort, got {sort}")


def coerce_number_for_sort(value: object, sort: SortRef) -> ExprRef:
    if is_expr(value):
        if not value.sort().eq(sort):
            raise TypeError(f"Sort mismatch: expected {sort}, got {value.sort()}")
        return value
    if is_bool_sort(sort):
        return BoolVal(bool(value))
    if is_int_sort(sort):
        return IntVal(int(value))
    if is_real_sort(sort):
        if isinstance(value, Fraction):
            return RealVal(f"{value.numerator}/{value.denominator}")
        return RealVal(str(value))
    if is_bv_sort(sort):
        return BitVecVal(int(value), sort.size())
    if sort.kind() == Z3_STRING_SORT:
        return StringVal(str(value))
    raise TypeError(f"Cannot coerce {value!r} into sort {sort}")


# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class Candidate:
    vals: Assignment
    source: str
    template_id: int = 0
    uid: int = -1
    coverage: float = 0.0
    novelty: float = 1.0
    score: float = 0.0
    signature: Tuple[int, ...] = field(default_factory=tuple)


@dataclass
class Attack:
    rep: Assignment
    guard: BoolRef
    source: str
    uid: int = -1
    power: float = 0.0
    novelty: float = 1.0
    score: float = 0.0
    signature: Tuple[int, ...] = field(default_factory=tuple)


@dataclass
class ExistsForallResult:
    status: str
    witness: Optional[Dict[str, object]]
    iterations: int
    message: str = ""


# =============================================================================
# Generic dual-memory CEGIS base
# =============================================================================

class DualMemoryCEGISBase:
    """
    Solves:
        exists X . domain_x(X) and forall Y . domain_y(Y) -> predicate(X, Y)

    Soundness:
      - returns a witness only after exact certification:
            SAT?  domain_y(Y) and Not(predicate(x, Y))
        If UNSAT, x is valid.

    Subclasses customize:
      - exact-check solver choice
      - region generalization
      - novelty objectives
      - optional arithmetic margin objectives
    """

    def __init__(
        self,
        *,
        x_vars: Sequence[ExprRef],
        y_vars: Sequence[ExprRef],
        predicate: BoolRef,
        domain_x: Union[BoolRef, bool] = True,
        domain_y: Union[BoolRef, bool] = True,
        x_templates: Optional[Sequence[Union[BoolRef, bool]]] = None,
        y_region_seeds: Optional[Sequence[Union[BoolRef, bool]]] = None,
        region_basis: Optional[Sequence[Union[BoolRef, bool]]] = None,
        initial_x: Optional[Sequence[AssignmentInput]] = None,
        initial_y: Optional[Sequence[AssignmentInput]] = None,
        max_iters: int = 40,
        max_x_memory: int = 24,
        max_y_memory: int = 24,
        init_x_budget: int = 4,
        init_y_budget: int = 6,
        attack_bundle_size: int = 6,
        cluster_size: int = 6,
        x_synth_per_iter: int = 3,
        y_synth_per_iter: int = 3,
        certify_top_k: int = 4,
        promising_threshold: float = 0.95,
        max_region_atoms: int = 16,
        timeout_ms: int = 5000,
        verbose: bool = True,
    ):
        self.x_vars = list(x_vars)
        self.y_vars = list(y_vars)
        self.predicate = self._normalize_formula(predicate, "predicate")
        self.domain_x = self._normalize_formula(domain_x, "domain_x")
        self.domain_y = self._normalize_formula(domain_y, "domain_y")
        self.x_templates = (
            [self._normalize_formula(t, "x_template") for t in x_templates]
            if x_templates else [BoolVal(True)]
        )
        self.y_region_seeds = (
            [self._normalize_formula(g, "y_region_seed") for g in y_region_seeds]
            if y_region_seeds else []
        )
        self.region_basis = (
            [self._normalize_formula(a, "region_basis_atom") for a in region_basis]
            if region_basis else []
        )
        self.initial_x = list(initial_x) if initial_x else []
        self.initial_y = list(initial_y) if initial_y else []

        self.max_iters = max_iters
        self.max_x_memory = max_x_memory
        self.max_y_memory = max_y_memory
        self.init_x_budget = init_x_budget
        self.init_y_budget = init_y_budget
        self.attack_bundle_size = attack_bundle_size
        self.cluster_size = cluster_size
        self.x_synth_per_iter = x_synth_per_iter
        self.y_synth_per_iter = y_synth_per_iter
        self.certify_top_k = certify_top_k
        self.promising_threshold = promising_threshold
        self.max_region_atoms = max_region_atoms
        self.timeout_ms = timeout_ms
        self.verbose = verbose

        self.M_X: List[Candidate] = []
        self.M_Y: List[Attack] = []

        self._x_seen = set()
        self._y_seen = set()
        self._next_x_uid = 0
        self._next_y_uid = 0

    # -------------------------------------------------------------------------
    # Public
    # -------------------------------------------------------------------------

    def solve(self) -> ExistsForallResult:
        x_admissible = self._x_admissibility_formula()
        rx, x0 = self._check_with_model([x_admissible], self.x_vars)
        if rx == unsat:
            return ExistsForallResult(
                status="unsat-domain-x",
                witness=None,
                iterations=0,
                message="domain_x and x_templates are unsatisfiable.",
            )
        if rx != sat:
            return ExistsForallResult(
                status="unknown",
                witness=None,
                iterations=0,
                message="Could not decide domain_x.",
            )

        ry, _ = self._check_with_model([self.domain_y], self.y_vars)
        if ry == unsat:
            return ExistsForallResult(
                status="valid",
                witness=self._named_assignment(self.x_vars, x0),
                iterations=0,
                message="domain_y is unsatisfiable, so the universal condition is vacuous.",
            )
        if ry != sat:
            return ExistsForallResult(
                status="unknown",
                witness=None,
                iterations=0,
                message="Could not decide domain_y.",
            )

        self._initialize_memories(seed_x=x0)
        self._cross_play_update()

        for it in range(1, self.max_iters + 1):
            if self.verbose:
                best_cov = max((c.coverage for c in self.M_X), default=0.0)
                best_pow = max((a.power for a in self.M_Y), default=0.0)
                print(
                    f"[iter {it:02d}] "
                    f"|M_X|={len(self.M_X):02d} "
                    f"|M_Y|={len(self.M_Y):02d} "
                    f"best_cov={best_cov:.3f} "
                    f"best_attack={best_pow:.3f}"
                )

            # 1) Y -> X
            x_new: List[Candidate] = []
            bundles = self._select_attack_bundles()

            for template_id, template in enumerate(self.x_templates):
                for bundle in bundles:
                    if len(x_new) >= self.x_synth_per_iter:
                        break
                    cand = self._synthesize_x(
                        bundle=bundle,
                        template=template,
                        template_id=template_id,
                        blocked=[c.vals for c in x_new],
                    )
                    if cand is not None:
                        x_new.append(cand)
                if len(x_new) >= self.x_synth_per_iter:
                    break

            for cand in x_new:
                self._add_candidate(cand)

            # 2) X -> Y
            y_new: List[Attack] = []
            clusters = self._select_x_clusters()
            focus_guards = self._select_focus_guards()

            for idx, cluster in enumerate(clusters):
                if len(y_new) >= self.y_synth_per_iter:
                    break
                focus = focus_guards[idx % len(focus_guards)] if focus_guards else BoolVal(True)
                atk = self._synthesize_y(
                    cluster=cluster,
                    focus_guard=focus,
                    blocked=[a.rep for a in y_new],
                )
                if atk is not None:
                    y_new.append(atk)

            for atk in y_new:
                self._add_attack(atk)

            # 3) Cross-play
            self._cross_play_update()

            # 4) Exact certification
            promising = self._promising_candidates()
            added_cex = False

            for cand in promising:
                status, counterattack = self._verify_candidate(cand)
                if status == "valid":
                    return ExistsForallResult(
                        status="valid",
                        witness=self._named_assignment(self.x_vars, cand.vals),
                        iterations=it,
                        message="Certified witness found.",
                    )
                if status == "counterexample" and counterattack is not None:
                    if self._add_attack(counterattack):
                        added_cex = True

            if added_cex:
                self._cross_play_update()

        return ExistsForallResult(
            status="budget-exhausted",
            witness=None,
            iterations=self.max_iters,
            message="No certified witness found within the iteration budget.",
        )

    # -------------------------------------------------------------------------
    # Solver factories / subclass hooks
    # -------------------------------------------------------------------------

    def _new_solver(self) -> Solver:
        s = Solver()
        s.set(timeout=self.timeout_ms)
        return s

    def _new_optimize(self) -> Optimize:
        o = Optimize()
        o.set(timeout=self.timeout_ms)
        return o

    def _add_x_novelty_soft(self, opt: Optimize, recent: Sequence[Candidate]) -> None:
        for cand in recent:
            for xv, val in zip(self.x_vars, cand.vals):
                opt.add_soft(xv != val, weight="1", id="novel-x")

    def _add_y_novelty_soft(self, opt: Optimize, recent: Sequence[Attack]) -> None:
        for atk in recent:
            for yv, val in zip(self.y_vars, atk.rep):
                opt.add_soft(yv != val, weight="1", id="novel-y")

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

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def _initialize_memories(self, seed_x: Optional[Assignment]) -> None:
        for item in self.initial_x:
            vals = self._normalize_assignment_input(self.x_vars, item)
            self._add_candidate(Candidate(vals=vals, source="initial-x"))

        if seed_x is not None:
            self._add_candidate(Candidate(vals=seed_x, source="domain-x-seed"))

        for item in self.initial_y:
            vals = self._normalize_assignment_input(self.y_vars, item)
            self._add_attack(Attack(rep=vals, guard=BoolVal(True), source="initial-y"))

        for region in self.y_region_seeds:
            r, vals = self._check_with_model([self.domain_y, region], self.y_vars)
            if r == sat and vals is not None:
                self._add_attack(Attack(rep=vals, guard=region, source="seed-region"))

        self._bootstrap_y()
        self._bootstrap_x()

    def _bootstrap_y(self) -> None:
        if len(self.M_Y) >= self.init_y_budget:
            return

        if self.region_basis:
            y_only_basis = [
                a for a in self.region_basis
                if (not expr_mentions_any(a, self.x_vars)) and expr_mentions_any(a, self.y_vars)
            ]
        else:
            y_only_basis = extract_atoms(self.domain_y)

        for atom in y_only_basis:
            if len(self.M_Y) >= self.init_y_budget:
                break
            for lit in (atom, Not(atom)):
                if len(self.M_Y) >= self.init_y_budget:
                    break
                r, vals = self._check_with_model([self.domain_y, lit], self.y_vars)
                if r == sat and vals is not None:
                    self._add_attack(Attack(rep=vals, guard=simplify(lit), source="bootstrap-basis"))

        while len(self.M_Y) < self.init_y_budget:
            vals = self._diverse_model(self.domain_y, self.y_vars, [a.rep for a in self.M_Y])
            if vals is None:
                break
            self._add_attack(Attack(rep=vals, guard=BoolVal(True), source="bootstrap-y"))

    def _bootstrap_x(self) -> None:
        if len(self.M_X) >= self.init_x_budget:
            return

        if self.M_Y:
            bundles = self._select_attack_bundles()
            for template_id, template in enumerate(self.x_templates):
                for bundle in bundles:
                    if len(self.M_X) >= self.init_x_budget:
                        break
                    cand = self._synthesize_x(
                        bundle=bundle,
                        template=template,
                        template_id=template_id,
                        blocked=[],
                    )
                    if cand is not None:
                        self._add_candidate(cand)
                if len(self.M_X) >= self.init_x_budget:
                    break

        for template_id, template in enumerate(self.x_templates):
            while len(self.M_X) < self.init_x_budget:
                vals = self._diverse_model(
                    mk_and([self.domain_x, template]),
                    self.x_vars,
                    [c.vals for c in self.M_X],
                )
                if vals is None:
                    break
                self._add_candidate(Candidate(vals=vals, source="bootstrap-x", template_id=template_id))
            if len(self.M_X) >= self.init_x_budget:
                break

    # -------------------------------------------------------------------------
    # Selection
    # -------------------------------------------------------------------------

    def _select_attack_bundles(self) -> List[List[Attack]]:
        if not self.M_Y:
            return [[]]

        attacks = sorted(self.M_Y, key=lambda a: (-a.score, -a.power, -a.uid))
        k = min(self.attack_bundle_size, len(attacks))
        bundles: List[List[Attack]] = []

        candidates = [
            attacks[:k],
            attacks[-k:],
            attacks[::max(1, len(attacks) // max(1, k))][:k],
        ]

        seen = set()
        for bundle in candidates:
            ids = tuple(a.uid for a in bundle)
            if bundle and ids not in seen:
                bundles.append(bundle)
                seen.add(ids)

        return bundles[:max(1, self.x_synth_per_iter)] or [[]]

    def _select_x_clusters(self) -> List[List[Candidate]]:
        if not self.M_X:
            return []

        groups: Dict[Tuple[int, ...], List[Candidate]] = {}
        for c in self.M_X:
            groups.setdefault(c.signature if c.signature else (-1,), []).append(c)

        clusters = list(groups.values())
        clusters.sort(
            key=lambda grp: (
                -len(grp),
                -(sum(c.coverage for c in grp) / len(grp)),
                -max(c.uid for c in grp),
            )
        )

        out: List[List[Candidate]] = []
        used = set()
        for grp in clusters:
            cluster = sorted(grp, key=lambda c: (-c.score, -c.coverage, -c.uid))[:self.cluster_size]
            ids = tuple(c.uid for c in cluster)
            if ids not in used:
                out.append(cluster)
                used.add(ids)
            if len(out) >= self.y_synth_per_iter:
                break

        if len(out) < self.y_synth_per_iter:
            top = sorted(self.M_X, key=lambda c: (-c.score, -c.coverage, -c.uid))[:self.cluster_size]
            ids = tuple(c.uid for c in top)
            if top and ids not in used:
                out.append(top)

        return out[:self.y_synth_per_iter]

    def _select_focus_guards(self) -> List[BoolRef]:
        guards: List[BoolRef] = []
        for atk in sorted(self.M_Y, key=lambda a: (-a.power, -a.score, -a.uid)):
            g = simplify(atk.guard)
            if not is_true(g):
                guards.append(g)
            if len(guards) >= max(1, self.y_synth_per_iter):
                break
        return guards

    def _promising_candidates(self) -> List[Candidate]:
        ranked = sorted(self.M_X, key=lambda c: (-c.score, -c.coverage, -c.uid))
        strong = [c for c in ranked if c.coverage >= self.promising_threshold]
        return (strong if strong else ranked)[:self.certify_top_k]

    def _best_candidate(self) -> Optional[Candidate]:
        if not self.M_X:
            return None
        return max(self.M_X, key=lambda c: (c.score, c.coverage, c.uid))

    # -------------------------------------------------------------------------
    # Synthesis
    # -------------------------------------------------------------------------

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

        for atk in bundle:
            phi = self._instantiate(self.predicate, self.y_vars, atk.rep)
            weight = max(1, int(round(1 + 9 * atk.score)))
            opt.add_soft(phi, weight=str(weight), id="cover")

        self._add_x_novelty_soft(opt, self.M_X[-4:])

        res = opt.check()
        if res == sat:
            vals = self._assignment_from_model(opt.model(), self.x_vars)
            return Candidate(vals=vals, source="y->x:maxsat", template_id=template_id)

        # fallback: make top attacks hard
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

        for cand in cluster:
            bad = Not(self._instantiate(self.predicate, self.x_vars, cand.vals))
            weight = max(1, int(round(1 + 9 * cand.coverage)))
            opt.add_soft(bad, weight=str(weight), id="break")

        self._add_y_novelty_soft(opt, self.M_Y[-4:])

        if opt.check() == sat:
            y_vals = self._assignment_from_model(opt.model(), self.y_vars)
            broken = [c for c in cluster if not self._predicate_holds(c.vals, y_vals)]
            if broken:
                pivot = max(broken, key=lambda c: (c.score, c.coverage, c.uid))
                guard = self._generalize_failure(pivot, y_vals)
                return Attack(rep=y_vals, guard=guard, source="x->y:maxsat")

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

    # -------------------------------------------------------------------------
    # Cross-play and certification
    # -------------------------------------------------------------------------

    def _cross_play_update(self) -> None:
        matrix: Dict[Tuple[int, int], bool] = {}

        for i, cand in enumerate(self.M_X):
            sig = []
            for j, atk in enumerate(self.M_Y):
                ok = self._predicate_holds(cand.vals, atk.rep)
                matrix[(i, j)] = ok
                sig.append(1 if ok else 0)
            cand.signature = tuple(sig)
            cand.coverage = (sum(sig) / len(sig)) if sig else 1.0

        for j, atk in enumerate(self.M_Y):
            sig = []
            for i, _cand in enumerate(self.M_X):
                ok = matrix.get((i, j), True)
                sig.append(0 if ok else 1)
            atk.signature = tuple(sig)
            atk.power = (sum(sig) / len(sig)) if sig else 1.0

        for cand in self.M_X:
            others = [c.signature for c in self.M_X if c.uid != cand.uid]
            cand.novelty = self._signature_novelty(cand.signature, others)
            cand.score = 0.80 * cand.coverage + 0.20 * cand.novelty

        for atk in self.M_Y:
            others = [a.signature for a in self.M_Y if a.uid != atk.uid]
            atk.novelty = self._signature_novelty(atk.signature, others)
            atk.score = 0.80 * atk.power + 0.20 * atk.novelty

        self.M_X = sorted(self.M_X, key=lambda c: (-c.score, -c.coverage, -c.uid))[:self.max_x_memory]
        self.M_Y = sorted(self.M_Y, key=lambda a: (-a.score, -a.power, -a.uid))[:self.max_y_memory]

    def _verify_candidate(self, cand: Candidate) -> Tuple[str, Optional[Attack]]:
        if not self._candidate_admissible(cand.vals):
            return "invalid-domain", None

        pred_x = self._instantiate(self.predicate, self.x_vars, cand.vals)
        s = self._new_solver()
        s.add(self.domain_y, Not(pred_x))
        res = s.check()
        if res == unsat:
            return "valid", None
        if res == sat:
            y_vals = self._assignment_from_model(s.model(), self.y_vars)
            guard = self._generalize_failure(cand, y_vals)
            return "counterexample", Attack(rep=y_vals, guard=guard, source="certify")
        return "unknown", None

    # -------------------------------------------------------------------------
    # Generalization
    # -------------------------------------------------------------------------

    def _generalize_failure(self, cand: Candidate, y_vals: Assignment) -> BoolRef:
        pred_x = self._instantiate(self.predicate, self.x_vars, cand.vals)
        fail_formula = mk_and([self.domain_y, Not(pred_x)])

        basis = self._basis_for_generalization(cand, y_vals, fail_formula)
        if not basis:
            return self._point_guard(self.y_vars, y_vals)

        lits: List[BoolRef] = []
        for atom in basis:
            ground = self._instantiate(atom, self.y_vars, y_vals)
            truth = self._ground_holds(ground)
            lits.append(simplify(atom if truth else Not(atom)))

        if not self._guard_implies_failure(lits, pred_x):
            return self._point_guard(self.y_vars, y_vals)

        minimized = list(lits)
        i = 0
        while i < len(minimized):
            trial = minimized[:i] + minimized[i + 1 :]
            if self._guard_implies_failure(trial, pred_x):
                minimized = trial
            else:
                i += 1

        return simplify(mk_and(minimized))

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _normalize_formula(self, f: Union[BoolRef, bool], name: str) -> BoolRef:
        if isinstance(f, bool):
            return BoolVal(f)
        if not is_expr(f) or not is_bool(f):
            raise TypeError(f"{name} must be a Z3 BoolRef or bool.")
        return f

    def _normalize_assignment_input(
        self,
        vars_: Sequence[ExprRef],
        item: AssignmentInput,
    ) -> Assignment:
        if isinstance(item, dict):
            vals = []
            for v in vars_:
                if v in item:
                    raw = item[v]
                elif str(v) in item:
                    raw = item[str(v)]
                else:
                    raise KeyError(f"Missing value for variable {v}")
                vals.append(coerce_number_for_sort(raw, v.sort()))
            return tuple(vals)

        if isinstance(item, (list, tuple)):
            if len(item) != len(vars_):
                raise ValueError(f"Expected {len(vars_)} values, got {len(item)}")
            return tuple(coerce_number_for_sort(raw, v.sort()) for raw, v in zip(item, vars_))

        raise TypeError("Assignment must be a list/tuple or a dict.")

    def _assignment_from_model(self, model: ModelRef, vars_: Sequence[ExprRef]) -> Assignment:
        return tuple(model.eval(v, model_completion=True) for v in vars_)

    def _instantiate(self, expr: BoolRef, vars_: Sequence[ExprRef], vals: Assignment) -> BoolRef:
        if not vars_:
            return expr
        return substitute(expr, *[(v, val) for v, val in zip(vars_, vals)])

    def _named_assignment(self, vars_: Sequence[ExprRef], vals: Assignment) -> Dict[str, object]:
        return {str(v): z3_value_to_python(val) for v, val in zip(vars_, vals)}

    def _assignment_key(self, vals: Assignment) -> Tuple[str, ...]:
        return tuple(v.sexpr() for v in vals)

    def _assignment_neq_clause(self, vars_: Sequence[ExprRef], vals: Assignment) -> BoolRef:
        if not vars_:
            return BoolVal(False)
        return mk_or(v != val for v, val in zip(vars_, vals))

    def _assignment_satisfies(
        self,
        vars_: Sequence[ExprRef],
        vals: Assignment,
        formula: BoolRef,
    ) -> bool:
        return self._ground_holds(self._instantiate(formula, vars_, vals))

    def _point_guard(self, vars_: Sequence[ExprRef], vals: Assignment) -> BoolRef:
        return mk_and(v == val for v, val in zip(vars_, vals))

    def _guard_implies_failure(self, guard_lits: Sequence[BoolRef], pred_x: BoolRef) -> bool:
        r, _ = self._check_with_model([self.domain_y, *guard_lits, pred_x], self.y_vars)
        return r == unsat

    def _x_admissibility_formula(self) -> BoolRef:
        return mk_and([self.domain_x, mk_or(self.x_templates)])

    def _candidate_admissible(self, vals: Assignment) -> bool:
        return self._assignment_satisfies(self.x_vars, vals, self._x_admissibility_formula())

    def _ground_holds(self, ground_formula: BoolRef) -> bool:
        g = simplify(ground_formula)
        if is_true(g):
            return True
        if is_false(g):
            return False
        s = self._new_solver()
        s.add(Not(g))
        res = s.check()
        if res == unsat:
            return True
        if res == sat:
            return False
        return False

    def _predicate_holds(self, x_vals: Assignment, y_vals: Assignment) -> bool:
        phi = self._instantiate(self.predicate, self.x_vars, x_vals)
        phi = self._instantiate(phi, self.y_vars, y_vals)
        return self._ground_holds(phi)

    def _check_with_model(
        self,
        constraints: Sequence[BoolRef],
        vars_: Sequence[ExprRef],
    ) -> Tuple[CheckSatResult, Optional[Assignment]]:
        s = self._new_solver()
        s.add(*constraints)
        res = s.check()
        if res == sat:
            return res, self._assignment_from_model(s.model(), vars_)
        return res, None

    def _diverse_model(
        self,
        formula: BoolRef,
        vars_: Sequence[ExprRef],
        existing: Sequence[Assignment],
    ) -> Optional[Assignment]:
        if not vars_ and existing:
            return None

        s = self._new_solver()
        s.add(formula)
        for vals in existing:
            neq = self._assignment_neq_clause(vars_, vals)
            if not is_false(simplify(neq)):
                s.add(neq)
        if s.check() == sat:
            return self._assignment_from_model(s.model(), vars_)
        return None

    def _signature_novelty(
        self,
        sig: Tuple[int, ...],
        others: Sequence[Tuple[int, ...]],
    ) -> float:
        if not others or not sig:
            return 1.0
        dists = []
        for other in others:
            if len(other) != len(sig):
                continue
            dists.append(sum(1 for a, b in zip(sig, other) if a != b) / len(sig))
        return sum(dists) / len(dists) if dists else 1.0

    def _add_candidate(self, cand: Candidate) -> bool:
        if not self._candidate_admissible(cand.vals):
            return False
        key = self._assignment_key(cand.vals)
        if key in self._x_seen:
            return False
        self._x_seen.add(key)
        cand.uid = self._next_x_uid
        self._next_x_uid += 1
        self.M_X.append(cand)
        return True

    def _add_attack(self, atk: Attack) -> bool:
        if not self._assignment_satisfies(self.y_vars, atk.rep, self.domain_y):
            return False
        key = self._assignment_key(atk.rep)
        if key in self._y_seen:
            return False
        self._y_seen.add(key)
        atk.uid = self._next_y_uid
        self._next_y_uid += 1
        self.M_Y.append(atk)
        return True
