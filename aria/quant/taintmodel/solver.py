from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, cast

from z3 import (  # type: ignore
    And,
    AstVector,
    BoolVal,
    ExprRef,
    ModelRef,
    Not,
    QuantifierRef,
    Solver,
    is_quantifier,
    is_true,
    sat,
    simplify,
    substitute,
    substitute_vars,
    unsat,
)

from aria.utils.z3.expr import get_variables

from .compression import (
    CounterexampleCompression,
    certify_compression,
    propose_joint_compression,
)
from .taint import infer_sic_and_wic
from .util import fresh_const, project_model


class QuantSolver:
    """Solve prenex ``exists X. forall Y. P(X, Y)`` formulas.

    The default path is counterexample-guided and sends only quantifier-free
    assertions to the backend. It maintains an over-approximation ``A(X)`` of
    the solution set, verifies candidate assignments by searching for a
    universal counterexample, and refines ``A`` with ordinary or symbolic
    instances of ``P``.

    A certified compression may replace ``Y`` by a smaller residual tuple
    ``Z`` during candidate verification. The identity compression is always
    available, so failed or inconclusive analysis never blocks base progress.

    Passing ``refine_sic=False`` selects the legacy one-shot SIC reduction. It
    is retained as an experimental SAT fast path; it does not use a quantified
    completeness oracle and therefore returns ``unknown`` when its reduced
    problem is unsatisfiable.
    """

    def __init__(
        self,
        *,
        timeout_ms: Optional[int] = None,
        confirm_unsat: bool = True,
        simplify_sic: bool = True,
        verify_wic: bool = False,
        refine_sic: bool = True,
        max_refinement_rounds: Optional[int] = None,
        max_quantifier_witness_attempts: int = 4,
        max_cegis_rounds: Optional[int] = None,
        enable_compression: bool = True,
    ) -> None:
        self.timeout_ms = timeout_ms
        # Retained for source compatibility. CEGIS establishes UNSAT directly;
        # the one-shot SIC path never reports UNSAT from an incomplete region.
        self.confirm_unsat = confirm_unsat
        self.simplify_sic = simplify_sic
        self.verify_wic = verify_wic
        self.refine_sic = refine_sic
        self.max_quantifier_witness_attempts = max_quantifier_witness_attempts
        self.max_cegis_rounds = (
            max_cegis_rounds if max_cegis_rounds is not None else max_refinement_rounds
        )
        self.enable_compression = enable_compression
        self.last_statistics: Dict[str, int] = {}

    def solve(self, formula: ExprRef) -> Tuple[str, Optional[ModelRef]]:
        """Return ``(status, model)`` for the supported exists-forall fragment."""
        if isinstance(formula, AstVector):
            items = [formula[i] for i in range(len(formula))]
            formula = cast(ExprRef, items[0] if len(items) == 1 else And(*items))
        elif isinstance(formula, (list, tuple)):
            formula = cast(
                ExprRef,
                formula[0] if len(formula) == 1 else And(*formula),
            )

        keep_vars = self._sort_consts(get_variables(formula))
        parsed = self._parse_exists_forall_prefix(formula)
        if parsed is None:
            return "unknown", None
        matrix, universal_consts = parsed
        if not universal_consts:
            return self._solve_qf(matrix, keep_vars)

        existential_consts = self._collect_existential_consts(matrix, universal_consts)
        if not self.refine_sic:
            return self._solve_with_initial_sic(
                matrix,
                existential_consts,
                universal_consts,
                keep_vars,
            )
        return self._solve_with_cegis(
            matrix,
            existential_consts,
            universal_consts,
            keep_vars,
        )

    # --------------------------------------------------------------- QF CEGIS
    def _solve_with_cegis(
        self,
        matrix: ExprRef,
        existential_consts: Sequence[ExprRef],
        universal_consts: Sequence[ExprRef],
        keep_vars: Sequence[ExprRef],
    ) -> Tuple[str, Optional[ModelRef]]:
        candidate_space: ExprRef = BoolVal(True)
        identity = CounterexampleCompression.identity(universal_consts)
        compressions = [identity]
        analysis_attempted = False
        refinements = 0

        self.last_statistics = {
            "candidate_iterations": 0,
            "refinements": 0,
            "compressions_proposed": 0,
            "compressions_certified": 0,
            "compressed_verifications": 0,
        }

        while True:
            candidate_status, candidate = self._check_qf(candidate_space)
            if candidate_status == "unsat":
                return "unsat", None
            if candidate_status != "sat" or candidate is None:
                return "unknown", None

            # A round cap is only a resource limit. We still solve A first so
            # that the final refinement can establish UNSAT conclusively.
            if (
                self.max_cegis_rounds is not None
                and refinements >= self.max_cegis_rounds
            ):
                return "unknown", None

            self.last_statistics["candidate_iterations"] += 1
            compression = self._choose_compression(compressions, candidate)
            verifier_status, counterexample = self._verify_candidate(
                matrix,
                existential_consts,
                candidate,
                compression,
            )

            if verifier_status == "unknown" and compression is not identity:
                compression = identity
                verifier_status, counterexample = self._verify_candidate(
                    matrix,
                    existential_consts,
                    candidate,
                    identity,
                )

            if compression is not identity:
                self.last_statistics["compressed_verifications"] += 1

            if verifier_status == "unsat":
                return "sat", project_model(candidate, keep_vars)
            if verifier_status != "sat" or counterexample is None:
                return "unknown", None

            instance = self._counterexample_instance(
                matrix,
                compression,
                counterexample,
            )
            candidate_space = self._simplify_expr(
                cast(ExprRef, And(candidate_space, instance))
            )
            refinements += 1
            self.last_statistics["refinements"] = refinements

            # The first concrete failure triggers bounded, opportunistic
            # dependency discovery. A bad proposal is harmless because it is
            # checked against the entire matrix before use.
            if self.enable_compression and not analysis_attempted:
                analysis_attempted = True
                proposal = propose_joint_compression(
                    matrix,
                    existential_consts,
                    universal_consts,
                    timeout_ms=self.timeout_ms,
                )
                if proposal is not None:
                    self.last_statistics["compressions_proposed"] += 1
                    certificate_status, _ = certify_compression(
                        matrix,
                        existential_consts,
                        proposal,
                        timeout_ms=self.timeout_ms,
                    )
                    if certificate_status == "valid":
                        compressions.append(proposal)
                        self.last_statistics["compressions_certified"] += 1

    def _verify_candidate(
        self,
        matrix: ExprRef,
        existential_consts: Sequence[ExprRef],
        candidate: ModelRef,
        compression: CounterexampleCompression,
    ) -> Tuple[str, Optional[ModelRef]]:
        compressed_matrix = compression.compressed_matrix(matrix)
        substitutions = [
            (variable, candidate.eval(variable, model_completion=True))
            for variable in existential_consts
        ]
        grounded = cast(ExprRef, substitute(compressed_matrix, *substitutions))
        return self._check_qf(cast(ExprRef, Not(grounded)))

    def _counterexample_instance(
        self,
        matrix: ExprRef,
        compression: CounterexampleCompression,
        counterexample: ModelRef,
    ) -> ExprRef:
        residual_values = [
            (residual, counterexample.eval(residual, model_completion=True))
            for residual in compression.residuals
        ]
        symbolic_input = [
            cast(ExprRef, substitute(term, *residual_values))
            for term in compression.reconstruction
        ]
        return cast(
            ExprRef,
            substitute(matrix, *list(zip(compression.universals, symbolic_input))),
        )

    def _choose_compression(
        self,
        compressions: Sequence[CounterexampleCompression],
        candidate: ModelRef,
    ) -> CounterexampleCompression:
        applicable: List[CounterexampleCompression] = []
        for compression in compressions:
            value = self._simplify_expr(
                candidate.eval(compression.guard, model_completion=True)
            )
            if is_true(value):
                applicable.append(compression)
        assert applicable  # The identity guard is true.
        return min(
            applicable,
            key=lambda item: (
                (
                    item.residual_bit_count
                    if item.residual_bit_count is not None
                    else 1 << 60
                ),
                len(item.residuals),
                item.name,
            ),
        )

    # ---------------------------------------------------------- legacy SIC path
    def _solve_with_initial_sic(
        self,
        matrix: ExprRef,
        existential_consts: Sequence[ExprRef],
        universal_consts: Sequence[ExprRef],
        keep_vars: Sequence[ExprRef],
    ) -> Tuple[str, Optional[ModelRef]]:
        """Run the original one-shot taint reduction as a SAT-only fast path."""
        sic, _ = infer_sic_and_wic(
            matrix,
            set(universal_consts),
            do_simplify=self.simplify_sic,
            verify_wic=self.verify_wic,
        )
        soundness_status, _ = self._soundness_status(
            matrix,
            universal_consts,
            sic,
            existential_consts,
        )
        if soundness_status != "valid":
            return "unknown", None

        reduced = self._simplify_expr(cast(ExprRef, And(matrix, sic)))
        result, model = self._solve_qf(reduced, keep_vars)
        if result == "sat":
            return result, model
        # Reduced UNSAT says only that this sufficient region is empty.
        return "unknown", None

    def _soundness_status(
        self,
        matrix: ExprRef,
        targets: Sequence[ExprRef],
        sic: ExprRef,
        existential_consts: Sequence[ExprRef],
    ) -> Tuple[str, Optional[ModelRef]]:
        other_targets = [fresh_const(target.sort(), "y_sound") for target in targets]
        other_matrix = substitute(matrix, *list(zip(targets, other_targets)))
        status, model = self._check_qf(
            cast(ExprRef, And(matrix, sic, Not(other_matrix)))
        )
        if status == "unsat":
            return "valid", None
        if status == "sat" and model is not None:
            return "counterexample", project_model(model, existential_consts)
        return "unknown", None

    # --------------------------------------------------------------- utilities
    def _check_qf(self, formula: ExprRef) -> Tuple[str, Optional[ModelRef]]:
        if self._contains_quantifier(formula):
            return "unknown", None
        solver = Solver()
        if self.timeout_ms:
            solver.set("timeout", self.timeout_ms)
        solver.add(formula)
        result = solver.check()
        if result == sat:
            return "sat", solver.model()
        if result == unsat:
            return "unsat", None
        return "unknown", None

    def _solve_qf(
        self,
        formula: ExprRef,
        keep_vars: Sequence[ExprRef],
    ) -> Tuple[str, Optional[ModelRef]]:
        status, model = self._check_qf(formula)
        if status == "sat" and model is not None:
            return status, project_model(model, keep_vars)
        return status, None

    def _parse_exists_forall_prefix(
        self,
        expr: ExprRef,
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
        if self._contains_quantifier(cur):
            return None
        return cur, self._sort_consts(universal_consts)

    def _collect_existential_consts(
        self,
        matrix: ExprRef,
        universal_consts: Sequence[ExprRef],
    ) -> List[ExprRef]:
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


def solve_file(path: str, *, timeout_ms: Optional[int] = None) -> None:
    from z3 import parse_smt2_file

    formula = parse_smt2_file(path)
    if isinstance(formula, list):
        formula = cast(ExprRef, And(*formula))
    result, model = QuantSolver(timeout_ms=timeout_ms).solve(formula)
    print(result)
    if model is not None:
        print(model)
