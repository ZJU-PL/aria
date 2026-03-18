"""SMT-enhanced Programming by Example solver."""

from typing import Any, Dict, List, Optional

from .expressions import Expression, Theory
from .pbe_solver import PBESolver, SynthesisResult
from .smt_verifier import SMTVerifier


class SMTPBESolver(PBESolver):
    """PBE solver with SMT-backed verification and distinguishing inputs."""

    def __init__(
        self,
        max_expression_depth: int = 3,
        timeout: float = 30.0,
        max_counterexamples: int = 10,
        use_smt: bool = True,
        theory_hint: Optional[Theory] = None,
        max_candidates: int = 2000,
        bitwidth: int = 32,
    ):
        super().__init__(
            max_expression_depth=max_expression_depth,
            timeout=timeout,
            max_counterexamples=max_counterexamples,
            theory_hint=theory_hint,
            max_candidates=max_candidates,
            bitwidth=bitwidth,
        )
        self.use_smt = use_smt
        self.smt_verifier = SMTVerifier() if use_smt else None

    def synthesize(
        self, examples: List[Dict[str, Any]], theory: Optional[Theory] = None
    ) -> SynthesisResult:
        """Synthesize a program from input-output examples using SMT validation."""
        result = super().synthesize(examples, theory=theory)
        if not self.use_smt or not self.smt_verifier or not result.success:
            return result

        if result.expression and not self.smt_verifier.verify_expression(
            result.expression, examples
        ):
            return SynthesisResult(
                False,
                message="Best-ranked candidate failed SMT verification",
                statistics=result.statistics,
                task=result.task,
            )

        if result.version_space and len(result.version_space) > 1:
            smt_counterexample = self._find_refinement_counterexample(result)
            if smt_counterexample is not None:
                result.distinguishing_inputs = [smt_counterexample] + [
                    item
                    for item in result.distinguishing_inputs
                    if item != smt_counterexample
                ]

        result.statistics = dict(result.statistics)
        result.statistics["smt_integration"] = "enabled"
        return result

    def verify_with_smt(
        self, expression: Expression, examples: List[Dict[str, Any]]
    ) -> bool:
        """Verify an expression using SMT."""
        if not self.smt_verifier:
            return self.verify(expression, examples)
        return self.smt_verifier.verify_expression(expression, examples)

    def prove_equivalence_with_smt(self, expr1: Expression, expr2: Expression) -> bool:
        """Prove equivalence of two expressions using SMT."""
        if not self.smt_verifier:
            return False
        return self.smt_verifier.prove_equivalence(expr1, expr2)

    def get_smt_formula(self, expression: Expression) -> str:
        """Get SMT-LIB format for an expression."""
        if not self.smt_verifier:
            return str(expression)
        return self.smt_verifier.get_smt_formula(expression)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Return SMT solver status information."""
        return {"smt_integration": "enabled" if self.use_smt else "disabled"}

    def enable_smt_integration(self) -> None:
        """Enable SMT integration."""
        self.use_smt = True
        self.smt_verifier = SMTVerifier()

    def disable_smt_integration(self) -> None:
        """Disable SMT integration."""
        self.use_smt = False
        self.smt_verifier = None

    def _find_refinement_counterexample(
        self, result: SynthesisResult
    ) -> Optional[Dict[str, Any]]:
        if (
            self.use_smt
            and self.smt_verifier is not None
            and result.version_space is not None
            and result.task is not None
        ):
            smt_counterexample = self.smt_verifier.find_counterexample(
                list(result.version_space.expressions),
                result.task.as_examples(),
                task=result.task,
            )
            if smt_counterexample is not None:
                return smt_counterexample

        return super()._find_refinement_counterexample(result)
