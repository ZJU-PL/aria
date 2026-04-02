# sic_smt/solver.py -----------------------------------------------------------
from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Set, Tuple, cast

from z3 import (  # type: ignore
    And,
    AstVector,
    BoolRef,
    BoolVal,
    ExprRef,
    ForAll,
    ModelRef,
    Not,
    Or,
    QuantifierRef,
    Solver,
    is_false,
    is_quantifier,
    is_true,
    sat,
    simplify,
    substitute,
    substitute_vars,
    unsat,
)

from aria.utils.z3_expr_utils import get_variables

from .taint import infer_sic_and_wic, infer_sic_candidates
from .util import fresh_const, project_model

logger = logging.getLogger(__name__)


class QuantSolver:
    """
    Solver for prenex exists-forall formulas with a quantifier-free matrix,
    based on taint-generated SICs.

    Supported fragment:
        exists X . forall Y . P(X, Y)

    Free variables are treated as existentially quantified parameters. Any
    nested or alternating quantifier structure outside this fragment returns
    ``unknown``.

    By default the solver runs a counterexample-guided refinement loop around
    the taint engine. The legacy one-shot path is still available via
    ``refine_sic=False``.
    """

    def __init__(
        self,
        *,
        timeout_ms: Optional[int] = None,
        confirm_unsat: bool = True,
        simplify_sic: bool = True,
        verify_wic: bool = False,
        refine_sic: bool = True,
        max_refinement_rounds: int = 8,
        max_quantifier_witness_attempts: int = 4,
    ) -> None:
        self.timeout_ms = timeout_ms
        self.confirm_unsat = confirm_unsat
        self.simplify_sic = simplify_sic
        self.verify_wic = verify_wic
        self.refine_sic = refine_sic
        self.max_refinement_rounds = max_refinement_rounds
        self.max_quantifier_witness_attempts = max_quantifier_witness_attempts

    # ------------------------------------------------------------------ core
    def solve(self, formula: ExprRef) -> Tuple[str, Optional[ModelRef]]:
        """
        Return (result, model):

        * result ∈ {"sat","unsat","unknown"}
        * model  : model of the **original** formula (free symbols only)
        """
        if isinstance(formula, AstVector):
            items = [formula[i] for i in range(len(formula))]
            formula = (
                cast(ExprRef, items[0])
                if len(items) == 1
                else cast(ExprRef, And(*items))
            )
        elif isinstance(formula, (list, tuple)):
            formula = (
                cast(ExprRef, formula[0])
                if len(formula) == 1
                else cast(ExprRef, And(*formula))
            )
        work = formula
        keep_vars = self._sort_consts(get_variables(work))

        parsed = self._parse_exists_forall_prefix(work)
        if parsed is None:
            return "unknown", None
        matrix, universal_consts = parsed

        if not universal_consts:
            return self._solve_qf(matrix, keep_vars)

        existential_consts = self._collect_existential_consts(matrix, universal_consts)
        if self.refine_sic:
            return self._solve_with_refinement(
                matrix,
                existential_consts,
                universal_consts,
                keep_vars,
            )
        return self._solve_with_initial_sic(
            matrix,
            existential_consts,
            universal_consts,
            keep_vars,
        )

    # ---------------------------------------------------------------- helpers
    def _solve_qf(
        self, formula: ExprRef, keep_vars: Sequence[ExprRef]
    ) -> Tuple[str, Optional[ModelRef]]:
        if self._contains_quantifier(formula):
            return "unknown", None

        s = Solver()
        if self.timeout_ms:
            s.set("timeout", self.timeout_ms)
        s.add(formula)
        r = s.check()
        if r == sat:
            return "sat", project_model(s.model(), keep_vars)
        if r == unsat:
            return "unsat", None
        return "unknown", None

    def _solve_with_initial_sic(
        self,
        matrix: ExprRef,
        existential_consts: Sequence[ExprRef],
        universal_consts: Sequence[ExprRef],
        keep_vars: Sequence[ExprRef],
    ) -> Tuple[str, Optional[ModelRef]]:
        """Run the original one-shot taint reduction without refinement."""
        sic, _ = infer_sic_and_wic(
            matrix,
            set(universal_consts),
            do_simplify=self.simplify_sic,
            verify_wic=self.verify_wic,
        )
        reduced = self._simplify_expr(cast(ExprRef, And(matrix, sic)))

        soundness_status, _ = self._soundness_status(
            matrix,
            universal_consts,
            sic,
            existential_consts,
        )
        if soundness_status != "valid":
            return "unknown", None

        result, model = self._solve_qf(reduced, keep_vars)
        if result == "sat":
            return result, model
        if result == "unknown":
            return result, model

        if not self.confirm_unsat:
            return "unsat", None

        completeness_status, _ = self._completeness_status(
            matrix,
            existential_consts,
            universal_consts,
            sic,
        )
        if completeness_status == "complete":
            return "unsat", None
        return "unknown", None

    def _solve_with_refinement(
        self,
        matrix: ExprRef,
        existential_consts: Sequence[ExprRef],
        universal_consts: Sequence[ExprRef],
        keep_vars: Sequence[ExprRef],
    ) -> Tuple[str, Optional[ModelRef]]:
        """
        Learn a better SIC from soundness/completeness counterexamples.

        The loop maintains samples over the existential block only:
        - negative samples violate soundness and must be excluded by the SIC;
        - positive samples satisfy ``forall Y. P(X, Y)`` and should be
          admitted by the SIC so that unsat is not concluded prematurely.
        """
        base_sic, _, initial_candidates = infer_sic_candidates(
            matrix,
            set(universal_consts),
            do_simplify=self.simplify_sic,
            verify_wic=self.verify_wic,
        )
        current_sic = base_sic
        candidate_guards = self._merge_candidates(initial_candidates, [base_sic])

        positive_samples: List[ModelRef] = []
        negative_samples: List[ModelRef] = []
        positive_signatures: Set[Tuple[Tuple[str, str], ...]] = set()
        negative_signatures: Set[Tuple[Tuple[str, str], ...]] = set()
        seen_sics: Set[str] = set()

        for _ in range(self.max_refinement_rounds + 1):
            current_sic = self._simplify_expr(current_sic)
            sic_key = current_sic.sexpr()
            if sic_key in seen_sics:
                return "unknown", None
            seen_sics.add(sic_key)

            soundness_status, bad_sample = self._soundness_status(
                matrix,
                universal_consts,
                current_sic,
                existential_consts,
            )
            if soundness_status == "unknown":
                return "unknown", None
            if soundness_status == "counterexample":
                assert bad_sample is not None
                self._append_sample(
                    negative_samples,
                    negative_signatures,
                    bad_sample,
                    existential_consts,
                )
                next_sic = self._strengthen_sic(
                    current_sic,
                    candidate_guards,
                    positive_samples,
                    negative_samples,
                    existential_consts,
                )
                if next_sic is None or next_sic.eq(current_sic):
                    return "unknown", None
                candidate_guards = self._merge_candidates(
                    candidate_guards, [current_sic, next_sic]
                )
                current_sic = next_sic
                continue

            # Once the current SIC is sound, solving ``matrix /\ sic`` is enough
            # for SAT: any model for the reduced formula is a valid existential
            # witness for the original quantified problem.
            reduced = self._simplify_expr(cast(ExprRef, And(matrix, current_sic)))
            result, model = self._solve_qf(reduced, keep_vars)
            if result == "sat":
                return result, model
            if result == "unknown":
                return result, model

            if not self.confirm_unsat:
                return "unsat", None

            # UNSAT is harder: an unsat reduced problem is only conclusive when
            # every truly satisfying existential assignment already lies inside
            # the current SIC.
            completeness_status, good_sample = self._completeness_status(
                matrix,
                existential_consts,
                universal_consts,
                current_sic,
            )
            if completeness_status == "complete":
                return "unsat", None
            if completeness_status == "unknown":
                return "unknown", None

            assert good_sample is not None
            self._append_sample(
                positive_samples,
                positive_signatures,
                good_sample,
                existential_consts,
            )
            next_sic = self._weaken_sic(
                current_sic,
                candidate_guards,
                positive_samples,
                negative_samples,
                existential_consts,
            )
            if next_sic is None or next_sic.eq(current_sic):
                return "unknown", None
            candidate_guards = self._merge_candidates(
                candidate_guards, [current_sic, next_sic]
            )
            current_sic = next_sic

        return "unknown", None

    def _parse_exists_forall_prefix(
        self, expr: ExprRef
    ) -> Optional[Tuple[ExprRef, List[ExprRef]]]:
        universal_consts: List[ExprRef] = []
        cur = expr

        while is_quantifier(cur) and cast(QuantifierRef, cur).is_exists():
            _, cur = self._open_quantifier(cast(QuantifierRef, cur))

        while is_quantifier(cur) and cast(QuantifierRef, cur).is_forall():
            consts, cur = self._open_quantifier(cast(QuantifierRef, cur))
            universal_consts.extend(consts)

        if is_quantifier(cur):
            return None

        cur = self._simplify_expr(cur)
        if self._contains_quantifier(cur):
            return None
        return cur, self._sort_consts(universal_consts)

    def _soundness_status(
        self,
        matrix: ExprRef,
        targets: Sequence[ExprRef],
        sic: ExprRef,
        existential_consts: Sequence[ExprRef],
    ) -> Tuple[str, Optional[ModelRef]]:
        """
        Check ``P(X,Y) /\\ sic(X) -> P(X,Y')``.

        A satisfying model gives a bad existential assignment: one that passes
        the current SIC but still depends on the universal block.
        """
        other_targets = [fresh_const(t.sort(), "y_sound") for t in targets]
        other_matrix = substitute(matrix, *list(zip(targets, other_targets)))
        solver = Solver()
        if self.timeout_ms:
            solver.set("timeout", self.timeout_ms)
        solver.add(matrix, sic, Not(other_matrix))
        result = solver.check()
        if result == unsat:
            return "valid", None
        if result == sat:
            return "counterexample", project_model(
                solver.model(), existential_consts
            )
        return "unknown", None

    def _completeness_status(
        self,
        matrix: ExprRef,
        existential_consts: Sequence[ExprRef],
        targets: Sequence[ExprRef],
        sic: ExprRef,
    ) -> Tuple[str, Optional[ModelRef]]:
        """
        Check whether ``forall Y. P(X, Y)`` can hold outside the current SIC.

        A satisfying model for this query is only a candidate witness because it
        comes from a quantified formula. We therefore re-check it in a separate
        quantifier-free query before using it as a positive sample.
        """
        other_targets = [fresh_const(t.sort(), "y_complete") for t in targets]
        universally_valid_matrix = ForAll(
            list(other_targets),
            substitute(matrix, *list(zip(targets, other_targets))),
        )
        solver = Solver()
        if self.timeout_ms:
            solver.set("timeout", self.timeout_ms)
        solver.add(universally_valid_matrix, Not(sic))
        result = solver.check()
        if result == unsat:
            return "complete", None
        if result != sat:
            return "unknown", None

        attempts = 0
        while attempts < self.max_quantifier_witness_attempts:
            attempts += 1
            sample = project_model(solver.model(), existential_consts)
            if self._sample_is_universally_valid(
                matrix, existential_consts, sample
            ):
                return "counterexample", sample
            solver.add(Not(self._sample_cube(sample, existential_consts)))
            result = solver.check()
            if result != sat:
                break
        return "unknown", None

    def _sample_is_universally_valid(
        self,
        matrix: ExprRef,
        existential_consts: Sequence[ExprRef],
        sample: ModelRef,
    ) -> bool:
        substitutions = [
            (var, sample.eval(var, model_completion=True))
            for var in existential_consts
        ]
        instantiated = substitute(matrix, *substitutions)
        solver = Solver()
        if self.timeout_ms:
            solver.set("timeout", self.timeout_ms)
        solver.add(Not(instantiated))
        return solver.check() == unsat

    def _strengthen_sic(
        self,
        current_sic: ExprRef,
        candidate_guards: Sequence[BoolRef],
        positive_samples: Sequence[ModelRef],
        negative_samples: Sequence[ModelRef],
        existential_consts: Sequence[ExprRef],
    ) -> Optional[ExprRef]:
        """
        Exclude the latest bad sample while trying to preserve prior good ones.

        The preferred path relearns the SIC from the accumulated sample set. If
        that fails, we fall back to a local blocker derived from the newest bad
        sample.
        """
        learned = self._learn_sic_from_samples(
            candidate_guards,
            positive_samples,
            negative_samples,
            existential_consts,
        )
        if learned is not None and not learned.eq(current_sic):
            return learned

        if not negative_samples:
            return None
        blocker = self._generalize_negative_sample(
            negative_samples[-1], candidate_guards, existential_consts
        )
        next_sic = self._simplify_expr(cast(ExprRef, And(current_sic, blocker)))
        if next_sic.eq(current_sic):
            return None
        return next_sic

    def _weaken_sic(
        self,
        current_sic: ExprRef,
        candidate_guards: Sequence[BoolRef],
        positive_samples: Sequence[ModelRef],
        negative_samples: Sequence[ModelRef],
        existential_consts: Sequence[ExprRef],
    ) -> Optional[ExprRef]:
        """
        Admit the latest good sample without re-admitting known bad ones.

        The learned update prefers a compact union of regions. If no useful
        generalization is available, we add a single region covering the newest
        positive sample.
        """
        learned = self._learn_sic_from_samples(
            candidate_guards,
            positive_samples,
            negative_samples,
            existential_consts,
        )
        if learned is not None and not learned.eq(current_sic):
            return learned

        if not positive_samples:
            return None
        region = self._generalize_positive_sample(
            positive_samples[-1],
            negative_samples,
            candidate_guards,
            existential_consts,
        )
        next_sic = self._simplify_expr(cast(ExprRef, Or(current_sic, region)))
        if next_sic.eq(current_sic):
            return None
        return next_sic

    def _learn_sic_from_samples(
        self,
        candidate_guards: Sequence[BoolRef],
        positive_samples: Sequence[ModelRef],
        negative_samples: Sequence[ModelRef],
        existential_consts: Sequence[ExprRef],
    ) -> Optional[ExprRef]:
        """
        Build a DNF-style SIC from sample data.

        Each cube is synthesized from one positive sample and must exclude every
        known negative sample. The outer loop greedily covers the remaining
        positive samples with the smallest available set of cubes.
        """
        if not positive_samples:
            return None

        guards = self._merge_candidates(candidate_guards, [])
        uncovered = list(range(len(positive_samples)))
        cubes: List[ExprRef] = []

        while uncovered:
            best_cube: Optional[ExprRef] = None
            best_cover: List[int] = []
            for index in uncovered:
                cube = self._generalize_positive_sample(
                    positive_samples[index],
                    negative_samples,
                    guards,
                    existential_consts,
                )
                if any(
                    self._holds_on_sample(cube, sample)
                    for sample in negative_samples
                ):
                    continue
                cover = [
                    pos_index
                    for pos_index in uncovered
                    if self._holds_on_sample(cube, positive_samples[pos_index])
                ]
                if len(cover) > len(best_cover):
                    best_cube = cube
                    best_cover = cover
                elif (
                    len(cover) == len(best_cover)
                    and best_cube is not None
                    and self._formula_complexity(cube)
                    < self._formula_complexity(best_cube)
                ):
                    best_cube = cube
                    best_cover = cover

            if best_cube is None or not best_cover:
                return None

            cubes.append(best_cube)
            uncovered = [index for index in uncovered if index not in best_cover]

        if not cubes:
            return None
        if len(cubes) == 1:
            return self._simplify_expr(cubes[0])
        return self._simplify_expr(cast(ExprRef, Or(*cubes)))

    def _generalize_positive_sample(
        self,
        sample: ModelRef,
        negative_samples: Sequence[ModelRef],
        candidate_guards: Sequence[BoolRef],
        existential_consts: Sequence[ExprRef],
    ) -> ExprRef:
        """
        Synthesize one conjunctive region containing ``sample``.

        Literals are chosen greedily from the candidate-guard pool to exclude as
        many negative samples as possible. Falling back to the exact sample cube
        preserves soundness when no useful abstraction is found.
        """
        if not negative_samples:
            true_guards = [
                guard
                for guard in candidate_guards
                if not is_true(guard) and self._holds_on_sample(guard, sample)
            ]
            if true_guards:
                return min(true_guards, key=self._formula_complexity)
            return self._sample_cube(sample, existential_consts)

        remaining = set(range(len(negative_samples)))
        chosen: List[ExprRef] = []
        used: Set[str] = set()

        while remaining:
            best_literal: Optional[ExprRef] = None
            best_excluded: Set[int] = set()

            for guard in candidate_guards:
                if is_true(guard) or is_false(guard):
                    continue
                literal = (
                    cast(ExprRef, guard)
                    if self._holds_on_sample(guard, sample)
                    else cast(ExprRef, Not(guard))
                )
                literal_key = literal.sexpr()
                if literal_key in used:
                    continue
                excluded = {
                    index
                    for index in remaining
                    if not self._holds_on_sample(literal, negative_samples[index])
                }
                if len(excluded) > len(best_excluded):
                    best_literal = literal
                    best_excluded = excluded
                elif (
                    len(excluded) == len(best_excluded)
                    and best_literal is not None
                    and self._formula_complexity(literal)
                    < self._formula_complexity(best_literal)
                ):
                    best_literal = literal
                    best_excluded = excluded

            if best_literal is None or not best_excluded:
                return self._sample_cube(sample, existential_consts)

            chosen.append(best_literal)
            used.add(best_literal.sexpr())
            remaining -= best_excluded

        if len(chosen) == 1:
            return self._simplify_expr(chosen[0])
        return self._simplify_expr(cast(ExprRef, And(*chosen)))

    def _generalize_negative_sample(
        self,
        sample: ModelRef,
        candidate_guards: Sequence[BoolRef],
        existential_consts: Sequence[ExprRef],
    ) -> ExprRef:
        """Pick a short guard that is false on a bad sample, or block it exactly."""
        false_guards = [
            guard
            for guard in candidate_guards
            if not is_true(guard)
            and not is_false(guard)
            and not self._holds_on_sample(guard, sample)
        ]
        if false_guards:
            return min(false_guards, key=self._formula_complexity)
        return cast(ExprRef, Not(self._sample_cube(sample, existential_consts)))

    def _merge_candidates(
        self,
        candidate_guards: Sequence[BoolRef],
        extras: Sequence[ExprRef],
    ) -> List[BoolRef]:
        """Normalize, deduplicate, and sort candidate guards by increasing size."""
        merged: List[BoolRef] = []
        seen: Set[str] = set()
        for expr in list(candidate_guards) + list(extras):
            expr = self._simplify_expr(expr)
            if self._contains_quantifier(expr) or not expr.sort().eq(BoolVal(True).sort()):
                continue
            if is_true(expr) or is_false(expr):
                continue
            key = expr.sexpr()
            if key in seen:
                continue
            seen.add(key)
            merged.append(cast(BoolRef, expr))
        merged.sort(key=lambda expr: (self._formula_complexity(expr), expr.sexpr()))
        return merged

    def _holds_on_sample(self, expr: ExprRef, sample: ModelRef) -> bool:
        value = sample.eval(expr, model_completion=True)
        value = self._simplify_expr(value)
        if is_true(value):
            return True
        if is_false(value):
            return False
        raise ValueError(f"sample does not ground expression: {expr}")

    def _sample_cube(
        self, sample: ModelRef, existential_consts: Sequence[ExprRef]
    ) -> BoolRef:
        """Exact description of one existential assignment."""
        equalities = [
            cast(BoolRef, var == sample.eval(var, model_completion=True))
            for var in existential_consts
        ]
        if not equalities:
            return BoolVal(True)
        return self._simplify_expr(cast(ExprRef, And(*equalities)))

    def _append_sample(
        self,
        samples: List[ModelRef],
        signatures: Set[Tuple[Tuple[str, str], ...]],
        sample: ModelRef,
        existential_consts: Sequence[ExprRef],
    ) -> None:
        signature = self._sample_signature(sample, existential_consts)
        if signature in signatures:
            return
        signatures.add(signature)
        samples.append(sample)

    @staticmethod
    def _sample_signature(
        sample: ModelRef, existential_consts: Sequence[ExprRef]
    ) -> Tuple[Tuple[str, str], ...]:
        return tuple(
            (
                var.decl().name(),
                sample.eval(var, model_completion=True).sexpr(),
            )
            for var in existential_consts
        )

    @staticmethod
    def _formula_complexity(expr: ExprRef) -> int:
        return len(expr.sexpr())

    def _collect_existential_consts(
        self, matrix: ExprRef, universal_consts: Sequence[ExprRef]
    ) -> List[ExprRef]:
        """Free constants in the matrix that are not part of the universal block."""
        universals = self._sort_consts(universal_consts)
        return self._sort_consts(
            [
                symbol
                for symbol in get_variables(matrix)
                if not self._is_member(symbol, universals)
            ]
        )

    @staticmethod
    def _is_member(symbol: ExprRef, symbols: Sequence[ExprRef]) -> bool:
        return any(symbol.eq(other) for other in symbols)

    @staticmethod
    def _sort_consts(symbols: Sequence[ExprRef]) -> List[ExprRef]:
        return sorted(
            symbols,
            key=lambda expr: (expr.decl().name(), expr.sort().kind(), expr.sexpr()),
        )

    @staticmethod
    def _open_quantifier(q: QuantifierRef) -> Tuple[List[ExprRef], ExprRef]:
        consts = [fresh_const(q.var_sort(i), "q") for i in range(q.num_vars())]
        body = q.body()
        if consts:
            body = substitute_vars(body, *reversed(consts))
        return consts, body

    @staticmethod
    def _simplify_expr(expr: ExprRef) -> ExprRef:
        return cast(ExprRef, simplify(expr))

    @staticmethod
    def _contains_quantifier(expr: ExprRef) -> bool:
        stack = [expr]
        while stack:
            node = stack.pop()
            if is_quantifier(node):
                return True
            stack.extend(node.children())
        return False

    # ------------------------------------------------------------------ file


def solve_file(path: str, *, timeout_ms: Optional[int] = None) -> None:
    from z3 import parse_smt2_file  # late import

    f = parse_smt2_file(path)
    if isinstance(f, list):
        f = cast(ExprRef, And(*f))
    solver = QuantSolver(timeout_ms=timeout_ms)
    res, model = solver.solve(cast(ExprRef, f))
    print(res)
    if model is not None:
        print(model)
