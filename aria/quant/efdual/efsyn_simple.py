"""
ARC-CEGIS for ∃X ∀Y . P(X,Y)

Memories:
  M_X = witness pool of concrete X assignments + templates + coverage signatures
  M_Y = attack pool of concrete Y representatives + generalized region guards

repeat:
  1. Region/attack-bundle -> witness synthesis
     - choose a high-value bundle A of attacks from M_Y
     - solve a weighted MaxSMT problem over X:
         hard:   domain_X(X) ∧ chosen_template(X)
         soft:   P(X, y_a) for reps y_a in A, weighted by attack power
         soft:   coordinate-level novelty away from recent witnesses
     - add synthesized X candidates to M_X

  2. Witness-cluster -> attack synthesis
     - cluster M_X by failure signature on M_Y
     - for a selected cluster B, solve a weighted MaxSMT problem over Y:
         hard:   domain_Y(Y) ∧ optional focus_guard(Y)
         soft:   ¬P(x_b, Y) for x_b in B, weighted by witness coverage
         soft:   coordinate-level novelty away from recent attacks
     - generalize the resulting failing Y into a reusable region guard
     - add new attacks to M_Y

  3. Cross-play
     - evaluate all X in M_X against all Y reps in M_Y
     - update witness coverage / attack power / signature novelty

  4. Prune
     - keep the best-scoring diverse witnesses and attacks

  5. Exact certification
     - for each promising x in M_X:
         check SAT of domain_Y(Y) ∧ ¬P(x,Y)
         if UNSAT: return x
         if SAT:   add counterexample Y and its generalized region to M_Y
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

from z3 import *
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union


Assignment = Tuple[ExprRef, ...]
AssignmentInput = Union[Sequence[object], Dict[object, object]]


def mk_and(parts: Iterable[BoolRef]) -> BoolRef:
    parts = list(parts)
    return And(*parts) if parts else BoolVal(True)


def mk_or(parts: Iterable[BoolRef]) -> BoolRef:
    parts = list(parts)
    return Or(*parts) if parts else BoolVal(False)


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


class ArcCEGISExistsForallSolver:
    """
    Solves:

        exists X . domain_x(X) and forall Y . domain_y(Y) -> predicate(X, Y)

    Set domain_x = True and domain_y = True to recover the plain problem:

        exists X forall Y . predicate(X, Y)

    Inputs:
      - x_vars, y_vars: lists of Z3 constants
      - predicate: BoolRef over x_vars + y_vars
      - domain_x: BoolRef over x_vars
      - domain_y: BoolRef over y_vars
      - x_templates: optional list of BoolRef constraints over x_vars
      - y_region_seeds: optional list of BoolRef constraints over y_vars
      - region_basis: optional list of BoolRef "atoms" used for region generalization;
                      they may mention X and Y, because X will be substituted during
                      counterexample generalization

    Notes:
      - x_vars and y_vars should be constant symbols, not quantifiers.
      - The solver is sound: it only returns a witness after exact verification.
      - If some Z3 subproblem returns unknown, the implementation conservatively
        skips that substep or falls back to a simpler query.
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
        max_iters: Optional[int] = None,
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
        max_region_atoms: int = 12,
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
            if x_templates
            else [BoolVal(True)]
        )
        self.y_region_seeds = (
            [self._normalize_formula(r, "y_region_seed") for r in y_region_seeds]
            if y_region_seeds
            else []
        )
        self.region_basis = (
            [self._normalize_formula(a, "region_basis_atom") for a in region_basis]
            if region_basis
            else []
        )
        self.initial_x = list(initial_x) if initial_x else []
        self.initial_y = list(initial_y) if initial_y else []

        self.max_iters = None if max_iters is None else max(0, int(max_iters))
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
    # Public API
    # -------------------------------------------------------------------------

    def solve(self) -> ExistsForallResult:
        # Check existential domain first.
        x_admissible = self._x_admissibility_formula()
        r_x, x0 = self._check_with_model([x_admissible], self.x_vars)
        if r_x == unsat:
            return ExistsForallResult(
                status="unsat-domain-x",
                witness=None,
                iterations=0,
                message="domain_x and x_templates are unsatisfiable, so no witness X exists.",
            )
        if r_x != sat:
            return ExistsForallResult(
                status="unknown",
                witness=None,
                iterations=0,
                message="Could not decide satisfiability of domain_x.",
            )

        # If Y-domain is empty, the universal condition is vacuous.
        r_y, _ = self._check_with_model([self.domain_y], self.y_vars)
        if r_y == unsat:
            return ExistsForallResult(
                status="valid",
                witness=self._named_assignment(self.x_vars, x0),
                iterations=0,
                message="domain_y is unsatisfiable, so forall Y is vacuously true.",
            )
        if r_y != sat:
            return ExistsForallResult(
                status="unknown",
                witness=None,
                iterations=0,
                message="Could not decide satisfiability of domain_y.",
            )

        self._initialize_memories(seed_x=x0)
        self._cross_play_update()

        if self._finite_attacks_refute_all_x():
            return ExistsForallResult(
                status="unsat-finite-attacks",
                witness=None,
                iterations=0,
                message="Known concrete Y attacks rule out every admissible X.",
            )

        for it in self._iteration_numbers():
            if self.verbose:
                best_cov = max((c.coverage for c in self.M_X), default=0.0)
                best_pow = max((a.power for a in self.M_Y), default=0.0)
                print(
                    f"[iter {it:02d}] "
                    f"|M_X|={len(self.M_X):02d} "
                    f"|M_Y|={len(self.M_Y):02d} "
                    f"best_cov={best_cov:.3f} "
                    f"best_attack_power={best_pow:.3f}"
                )

            # 1) Y -> X : synthesize new witnesses from attack bundles.
            x_new: List[Candidate] = []
            bundles = self._select_attack_bundles()
            template_order = list(enumerate(self.x_templates))

            for template_id, template in template_order:
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

            # 2) X -> Y : synthesize new attacks from X clusters.
            y_new: List[Attack] = []
            clusters = self._select_x_clusters()
            focus_guards = self._select_focus_guards()

            for idx, cluster in enumerate(clusters):
                if len(y_new) >= self.y_synth_per_iter:
                    break
                focus_guard = (
                    focus_guards[idx % len(focus_guards)]
                    if focus_guards
                    else BoolVal(True)
                )
                atk = self._synthesize_y(
                    cluster=cluster,
                    focus_guard=focus_guard,
                    blocked=[a.rep for a in y_new],
                )
                if atk is not None:
                    y_new.append(atk)

            for atk in y_new:
                self._add_attack(atk)

            # 3) Cross-play
            self._cross_play_update()

            if self._finite_attacks_refute_all_x():
                return ExistsForallResult(
                    status="unsat-finite-attacks",
                    witness=None,
                    iterations=it,
                    message="Known concrete Y attacks rule out every admissible X.",
                )

            # 4) Certification
            promising = self._promising_candidates()
            added_counterexample = False

            for cand in promising:
                status, counterattack = self._verify_candidate(cand)
                if status == "valid":
                    witness = self._named_assignment(self.x_vars, cand.vals)
                    return ExistsForallResult(
                        status="valid",
                        witness=witness,
                        iterations=it,
                        message="Certified witness found.",
                    )
                if status == "counterexample" and counterattack is not None:
                    if self._add_attack(counterattack):
                        added_counterexample = True

            if added_counterexample:
                self._cross_play_update()
                if self._finite_attacks_refute_all_x():
                    return ExistsForallResult(
                        status="unsat-finite-attacks",
                        witness=None,
                        iterations=it,
                        message="Known concrete Y attacks rule out every admissible X.",
                    )

        return ExistsForallResult(
            status="budget-exhausted",
            witness=None,
            iterations=self.max_iters,
            message="No certified witness found within the iteration budget.",
        )

    def _iteration_numbers(self) -> Iterable[int]:
        it = 1
        while self.max_iters is None or it <= self.max_iters:
            yield it
            it += 1

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def _initialize_memories(self, seed_x: Optional[Assignment]) -> None:
        # Initial X seeds.
        for item in self.initial_x:
            vals = self._normalize_assignment_input(self.x_vars, item)
            self._add_candidate(Candidate(vals=vals, source="initial-x"))
        if seed_x is not None:
            self._add_candidate(Candidate(vals=seed_x, source="domain-x-seed"))

        # Initial Y points.
        for item in self.initial_y:
            vals = self._normalize_assignment_input(self.y_vars, item)
            self._add_attack(Attack(rep=vals, guard=BoolVal(True), source="initial-y"))

        # Initial Y regions.
        for region in self.y_region_seeds:
            r, vals = self._check_with_model([self.domain_y, region], self.y_vars)
            if r == sat and vals is not None:
                self._add_attack(Attack(rep=vals, guard=region, source="seed-region"))

        # Bootstrap Y from basis / domain.
        self._bootstrap_y()

        # Bootstrap X from bundles / domain.
        self._bootstrap_x()

    def _bootstrap_y(self) -> None:
        if len(self.M_Y) >= self.init_y_budget:
            return

        # Use Y-only basis atoms if available.
        y_only_basis = []
        if self.region_basis:
            for atom in self.region_basis:
                if (not self._mentions_any(atom, self.x_vars)) and self._mentions_any(atom, self.y_vars):
                    y_only_basis.append(atom)
        else:
            y_only_basis = self._extract_atoms(self.domain_y)

        for atom in y_only_basis:
            if len(self.M_Y) >= self.init_y_budget:
                break
            for lit in (atom, Not(atom)):
                if len(self.M_Y) >= self.init_y_budget:
                    break
                r, vals = self._check_with_model([self.domain_y, lit], self.y_vars)
                if r == sat and vals is not None:
                    self._add_attack(
                        Attack(rep=vals, guard=simplify(lit), source="bootstrap-basis")
                    )

        # Fallback: just sample diverse Y points from domain_y.
        while len(self.M_Y) < self.init_y_budget:
            vals = self._diverse_model(self.domain_y, self.y_vars, [a.rep for a in self.M_Y])
            if vals is None:
                break
            self._add_attack(Attack(rep=vals, guard=BoolVal(True), source="bootstrap-y"))

    def _bootstrap_x(self) -> None:
        if len(self.M_X) >= self.init_x_budget:
            return

        # If we already have attacks, try solving coverage bundles first.
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

        # Fallback: sample X directly from domain_x and templates.
        for template_id, template in enumerate(self.x_templates):
            while len(self.M_X) < self.init_x_budget:
                vals = self._diverse_model(
                    mk_and([self.domain_x, template]),
                    self.x_vars,
                    [c.vals for c in self.M_X],
                )
                if vals is None:
                    break
                self._add_candidate(
                    Candidate(vals=vals, source="bootstrap-x", template_id=template_id)
                )
            if len(self.M_X) >= self.init_x_budget:
                break

    # -------------------------------------------------------------------------
    # Selection
    # -------------------------------------------------------------------------

    def _select_attack_bundles(self) -> List[List[Attack]]:
        if not self.M_Y:
            return [[]]

        attacks = sorted(
            self.M_Y,
            key=lambda a: (-a.score, -a.power, -a.uid),
        )

        k = min(self.attack_bundle_size, len(attacks))
        candidates = []

        # hardest prefix
        candidates.append(attacks[:k])

        # recent/hard suffix
        candidates.append(attacks[-k:])

        # sparse diverse scan
        stride = max(1, len(attacks) // max(1, k))
        candidates.append(attacks[::stride][:k])

        bundles: List[List[Attack]] = []
        seen = set()
        for bundle in candidates:
            ids = tuple(a.uid for a in bundle)
            if ids not in seen and bundle:
                bundles.append(bundle)
                seen.add(ids)

        return bundles[: max(1, self.x_synth_per_iter)] or [[]]

    def _select_x_clusters(self) -> List[List[Candidate]]:
        if not self.M_X:
            return []

        groups: Dict[Tuple[int, ...], List[Candidate]] = {}
        for c in self.M_X:
            key = c.signature if c.signature else (-1,)
            groups.setdefault(key, []).append(c)

        clusters = list(groups.values())
        clusters.sort(
            key=lambda grp: (
                -len(grp),
                -sum(c.coverage for c in grp) / len(grp),
                -max(c.uid for c in grp),
            )
        )

        out = []
        used = set()

        for grp in clusters:
            grp_sorted = sorted(grp, key=lambda c: (-c.score, -c.coverage, -c.uid))
            cluster = grp_sorted[: self.cluster_size]
            ids = tuple(c.uid for c in cluster)
            if ids not in used:
                out.append(cluster)
                used.add(ids)
            if len(out) >= self.y_synth_per_iter:
                break

        # Also add a top-score cluster if not already included.
        if len(out) < self.y_synth_per_iter:
            top = sorted(self.M_X, key=lambda c: (-c.score, -c.coverage, -c.uid))[
                : self.cluster_size
            ]
            ids = tuple(c.uid for c in top)
            if top and ids not in used:
                out.append(top)

        return out[: self.y_synth_per_iter]

    def _select_focus_guards(self) -> List[BoolRef]:
        guards = []
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
        return (strong if strong else ranked)[: self.certify_top_k]

    def _best_candidate(self) -> Optional[Candidate]:
        if not self.M_X:
            return None
        return max(self.M_X, key=lambda c: (c.score, c.coverage, c.uid))

    # -------------------------------------------------------------------------
    # Synthesis: Y -> X
    # -------------------------------------------------------------------------

    def _synthesize_x(
        self,
        *,
        bundle: List[Attack],
        template: BoolRef,
        template_id: int,
        blocked: List[Assignment],
    ) -> Optional[Candidate]:
        s = Solver()
        s.set(timeout=self.timeout_ms)
        s.add(self.domain_x, template)

        # Hard-block exact duplicates.
        for vals in [c.vals for c in self.M_X] + list(blocked):
            neq = self._assignment_neq_clause(self.x_vars, vals)
            if not is_false(simplify(neq)):
                s.add(neq)

        for atk in bundle:
            phi = self._instantiate(self.predicate, self.y_vars, atk.rep)
            s.add(phi)

        if s.check() == sat:
            vals = self._assignment_from_model(s.model(), self.x_vars)
            return Candidate(vals=vals, source="y->x:smt", template_id=template_id)

        return None

    # -------------------------------------------------------------------------
    # Synthesis: X -> Y
    # -------------------------------------------------------------------------

    def _synthesize_y(
        self,
        *,
        cluster: List[Candidate],
        focus_guard: BoolRef,
        blocked: List[Assignment],
    ) -> Optional[Attack]:
        if not cluster:
            return None

        # First attempt: focused optimize search.
        attack = self._optimize_y_attack(cluster, focus_guard, blocked)
        if attack is not None:
            return attack

        # Second attempt: remove region focus.
        if not is_true(simplify(focus_guard)):
            attack = self._optimize_y_attack(cluster, BoolVal(True), blocked)
            if attack is not None:
                return attack

        # Final fallback: directly break the strongest candidate in the cluster.
        strongest = max(cluster, key=lambda c: (c.score, c.coverage, c.uid))
        s = Solver()
        s.set(timeout=self.timeout_ms)
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

    def _optimize_y_attack(
        self,
        cluster: List[Candidate],
        focus_guard: BoolRef,
        blocked: List[Assignment],
    ) -> Optional[Attack]:
        opt = Optimize()
        opt.set(timeout=self.timeout_ms)
        opt.add(self.domain_y)
        if not is_true(simplify(focus_guard)):
            opt.add(focus_guard)

        # Hard-block exact duplicates.
        for vals in [a.rep for a in self.M_Y] + list(blocked):
            neq = self._assignment_neq_clause(self.y_vars, vals)
            if not is_false(simplify(neq)):
                opt.add(neq)

        # Breaking objective: falsify as many strong candidates as possible.
        for cand in cluster:
            bad = Not(self._instantiate(self.predicate, self.x_vars, cand.vals))
            weight = max(1, int(round(1 + 9 * cand.coverage)))
            opt.add_soft(bad, weight=str(weight), id="break")

        # Coordinate-level novelty wrt recent attacks.
        for atk in self.M_Y[-4:]:
            for yv, val in zip(self.y_vars, atk.rep):
                opt.add_soft(yv != val, weight="1", id="novel")

        res = opt.check()
        if res != sat:
            return None

        y_vals = self._assignment_from_model(opt.model(), self.y_vars)

        broken = [cand for cand in cluster if not self._predicate_holds(cand.vals, y_vals)]
        if not broken:
            return None

        pivot = max(broken, key=lambda c: (c.score, c.coverage, c.uid))
        guard = self._generalize_failure(pivot, y_vals)
        return Attack(rep=y_vals, guard=guard, source="x->y:maxsat")

    # -------------------------------------------------------------------------
    # Cross-play + memory updates
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
            for i, cand in enumerate(self.M_X):
                ok = matrix.get((i, j), True)
                sig.append(0 if ok else 1)
            atk.signature = tuple(sig)
            atk.power = (sum(sig) / len(sig)) if sig else 1.0

        # Novelty on signatures.
        for cand in self.M_X:
            others = [c.signature for c in self.M_X if c.uid != cand.uid]
            cand.novelty = self._signature_novelty(cand.signature, others)
            cand.score = 0.80 * cand.coverage + 0.20 * cand.novelty

        for atk in self.M_Y:
            others = [a.signature for a in self.M_Y if a.uid != atk.uid]
            atk.novelty = self._signature_novelty(atk.signature, others)
            atk.score = 0.80 * atk.power + 0.20 * atk.novelty

        self.M_X = sorted(
            self.M_X,
            key=lambda c: (-c.score, -c.coverage, -c.uid),
        )[: self.max_x_memory]

        self.M_Y = sorted(
            self.M_Y,
            key=lambda a: (-a.score, -a.power, -a.uid),
        )[: self.max_y_memory]

    # -------------------------------------------------------------------------
    # Certification
    # -------------------------------------------------------------------------

    def _verify_candidate(self, cand: Candidate) -> Tuple[str, Optional[Attack]]:
        if not self._candidate_admissible(cand.vals):
            return "invalid-domain", None

        pred_x = self._instantiate(self.predicate, self.x_vars, cand.vals)

        s = Solver()
        s.set(timeout=self.timeout_ms)
        s.add(self.domain_y)
        s.add(Not(pred_x))

        res = s.check()
        if res == unsat:
            return "valid", None
        if res == sat:
            y_vals = self._assignment_from_model(s.model(), self.y_vars)
            guard = self._generalize_failure(cand, y_vals)
            return "counterexample", Attack(rep=y_vals, guard=guard, source="certify")
        return "unknown", None

    def _finite_attacks_refute_all_x(self) -> bool:
        if not self.M_Y:
            return False

        s = Solver()
        s.set(timeout=self.timeout_ms)
        s.add(self._x_admissibility_formula())
        for atk in self.M_Y:
            s.add(self._instantiate(self.predicate, self.y_vars, atk.rep))
        return s.check() == unsat

    # -------------------------------------------------------------------------
    # Region generalization
    # -------------------------------------------------------------------------

    def _generalize_failure(self, cand: Candidate, y_vals: Assignment) -> BoolRef:
        pred_x = self._instantiate(self.predicate, self.x_vars, cand.vals)
        fail_formula = mk_and([self.domain_y, Not(pred_x)])

        if self.region_basis:
            raw_basis = [self._instantiate(atom, self.x_vars, cand.vals) for atom in self.region_basis]
        else:
            raw_basis = self._extract_atoms(fail_formula)

        basis = []
        seen = set()
        for atom in raw_basis:
            atom = simplify(atom)
            if not is_bool(atom):
                continue
            if self._mentions_any(atom, self.x_vars):
                continue
            if not self._mentions_any(atom, self.y_vars):
                continue
            k = atom.sexpr()
            if k not in seen:
                basis.append(atom)
                seen.add(k)
            if len(basis) >= self.max_region_atoms:
                break

        if not basis:
            return self._point_guard(self.y_vars, y_vals)

        lits = []
        for atom in basis:
            truth = self._ground_holds(self._instantiate(atom, self.y_vars, y_vals))
            lits.append(simplify(atom if truth else Not(atom)))

        if not self._guard_implies_failure(lits, pred_x):
            return self._point_guard(self.y_vars, y_vals)

        # Greedy minimization: drop a literal only if the remaining guard still
        # implies failure for this candidate over domain_y.
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
        if not is_expr(f):
            raise TypeError(f"{name} must be a Z3 BoolRef or a Python bool.")
        if not is_bool(f):
            raise TypeError(f"{name} must have Bool sort.")
        return f

    def _coerce_to_sort(self, value: object, sort: SortRef) -> ExprRef:
        if is_expr(value):
            if not value.sort().eq(sort):
                raise TypeError(f"Sort mismatch: expected {sort}, got {value.sort()}")
            return value

        k = sort.kind()
        if k == Z3_BOOL_SORT:
            return BoolVal(bool(value))
        if k == Z3_INT_SORT:
            return IntVal(int(value))
        if k == Z3_REAL_SORT:
            if isinstance(value, Fraction):
                return RealVal(f"{value.numerator}/{value.denominator}")
            return RealVal(str(value))
        if k == Z3_BV_SORT:
            return BitVecVal(int(value), sort.size())
        if k == Z3_STRING_SORT:
            return StringVal(str(value))

        raise TypeError(
            f"Cannot coerce Python value {value!r} into sort {sort}. "
            "For uninterpreted / custom sorts, pass a Z3 value directly."
        )

    def _normalize_assignment_input(
        self, vars_: Sequence[ExprRef], item: AssignmentInput
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
                vals.append(self._coerce_to_sort(raw, v.sort()))
            return tuple(vals)

        if isinstance(item, (list, tuple)):
            if len(item) != len(vars_):
                raise ValueError(f"Expected {len(vars_)} values, got {len(item)}")
            return tuple(
                self._coerce_to_sort(raw, v.sort())
                for raw, v in zip(item, vars_)
            )

        raise TypeError("Assignment must be a sequence or a dict.")

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

        s = Solver()
        s.set(timeout=self.timeout_ms)
        s.add(Not(g))
        res = s.check()
        if res == unsat:
            return True
        if res == sat:
            return False

        # Conservative fallback.
        return False

    def _predicate_holds(self, x_vals: Assignment, y_vals: Assignment) -> bool:
        phi = self._instantiate(self.predicate, self.x_vars, x_vals)
        phi = self._instantiate(phi, self.y_vars, y_vals)
        return self._ground_holds(phi)

    def _check_with_model(
        self, constraints: Sequence[BoolRef], vars_: Sequence[ExprRef]
    ) -> Tuple[CheckSatResult, Optional[Assignment]]:
        s = Solver()
        s.set(timeout=self.timeout_ms)
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

        s = Solver()
        s.set(timeout=self.timeout_ms)
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
        if not others:
            return 1.0
        if not sig:
            return 1.0

        dists = []
        for other in others:
            if len(other) != len(sig) or len(sig) == 0:
                continue
            dist = sum(1 for a, b in zip(sig, other) if a != b) / len(sig)
            dists.append(dist)
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

    def _mentions_any(self, expr: ExprRef, vars_: Sequence[ExprRef]) -> bool:
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

    def _extract_atoms(self, expr: BoolRef) -> List[BoolRef]:
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

            if is_true(e) or is_false(e):
                return
            if is_quantifier(e):
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


# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Example:
    #   exists x_lo, x_hi .
    #       0 <= x_lo <= x_hi <= 10
    #   and forall y in [0, 10] .
    #       x_lo <= y <= x_hi
    #
    # One valid witness is x_lo = 0, x_hi = 10.

    x_lo, x_hi = Ints("x_lo x_hi")
    y = Int("y")

    solver = ArcCEGISExistsForallSolver(
        x_vars=[x_lo, x_hi],
        y_vars=[y],
        predicate=And(x_lo <= y, y <= x_hi),
        domain_x=And(0 <= x_lo, x_lo <= x_hi, x_hi <= 10),
        domain_y=And(0 <= y, y <= 10),
        max_iters=20,
        max_x_memory=20,
        max_y_memory=20,
        attack_bundle_size=5,
        cluster_size=5,
        x_synth_per_iter=3,
        y_synth_per_iter=3,
        certify_top_k=4,
        verbose=True,
    )

    result = solver.solve()
    print("\nRESULT")
    print("status :", result.status)
    print("witness:", result.witness)
    print("iters  :", result.iterations)
    print("msg    :", result.message)
