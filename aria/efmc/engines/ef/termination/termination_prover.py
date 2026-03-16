"""
Main termination prover class that orchestrates ranking function and recurrence set synthesis.
"""

import logging
import time
from typing import Optional
import z3

from aria.efmc.sts import TransitionSystem
from aria.efmc.engines.ef.templates import (
    BitVecLinearRankingTemplate,
    BitVecLexicographicRankingTemplate,
    BitVecConditionalRankingTemplate,
    BitVecLinearRecurrenceTemplate,
    BitVecIntervalRecurrenceTemplate,
    BitVecDisjunctiveRecurrenceTemplate,
)
from .result_types import TerminationResult, NonTerminationResult
from .vc_generators import RankingVCGenerator, RecurrenceVCGenerator
from .solvers import RankingSolver, RecurrenceSolver
from .validators import RankingFunctionValidator, RecurrenceSetValidator

logger = logging.getLogger(__name__)


class TerminationProver:
    """
    Template-based termination verification using ranking function synthesis
    and recurrence set based non-termination verification.
    """

    def __init__(self, sts: TransitionSystem, **kwargs):
        self.sts = sts
        self.ranking_template = None
        self.recurrence_template = None
        self.print_vc = kwargs.get("print_vc", False)

        # Configuration options
        self.solver = kwargs.get("solver", "z3api")
        self.validate_ranking_function = kwargs.get("validate_ranking_function", False)
        self.validate_recurrence_set = kwargs.get("validate_recurrence_set", False)
        self.num_components = kwargs.get("num_components", 2)
        self.num_branches = kwargs.get("num_branches", 2)

        # Initialize components
        self.ranking_vc_generator = RankingVCGenerator(sts)
        self.recurrence_vc_generator = RecurrenceVCGenerator(sts)
        self.ranking_solver = RankingSolver("BV", self.solver)
        self.recurrence_solver = RecurrenceSolver("BV", self.solver)
        self.ranking_validator = RankingFunctionValidator(sts)
        self.recurrence_validator = RecurrenceSetValidator(sts)

    def set_ranking_template(self, template_name: str, **kwargs):
        """Set the ranking function template to use."""
        if not self.sts.has_bv:
            raise NotImplementedError(
                "Currently only bit-vector programs are supported"
            )

        template_map = {
            "bv_linear_ranking": BitVecLinearRankingTemplate,
            "bv_lexicographic_ranking": lambda sts, **kw: BitVecLexicographicRankingTemplate(
                sts, num_components=self.num_components, **kw
            ),
            "bv_conditional_ranking": lambda sts, **kw: BitVecConditionalRankingTemplate(
                sts, num_branches=self.num_branches, **kw
            ),
        }

        if template_name not in template_map:
            raise NotImplementedError(
                f"Ranking template '{template_name}' not implemented"
            )

        self.ranking_template = template_map[template_name](self.sts, **kwargs)

    def set_recurrence_template(self, template_name: str, **kwargs):
        """Set the recurrence set template to use."""
        if not self.sts.has_bv:
            raise NotImplementedError(
                "Currently only bit-vector programs are supported"
            )

        template_map = {
            "bv_linear_recurrence": BitVecLinearRecurrenceTemplate,
            "bv_interval_recurrence": BitVecIntervalRecurrenceTemplate,
            "bv_disjunctive_recurrence": BitVecDisjunctiveRecurrenceTemplate,
        }

        if template_name not in template_map:
            raise NotImplementedError(
                f"Recurrence template '{template_name}' not implemented"
            )

        self.recurrence_template = template_map[template_name](self.sts, **kwargs)

    def prove_termination(self, timeout: Optional[int] = None) -> TerminationResult:
        """Prove termination by synthesizing a ranking function."""
        if self.ranking_template is None:
            raise ValueError(
                "No ranking template set. Call set_ranking_template() first."
            )

        start_time = time.time()

        try:
            # Generate and solve verification condition
            vc = self.ranking_vc_generator.generate_vc(self.ranking_template)

            if self.print_vc:
                print("Ranking function verification condition:")
                print(vc)

            result = self.ranking_solver.solve(vc, self.ranking_template, timeout)

            # Validate if requested and successful
            if result.result and self.validate_ranking_function:
                ranking_function = self._extract_ranking_function()
                if (
                    ranking_function
                    and not self.ranking_validator.validate_ranking_function(
                        ranking_function, self.ranking_template.signedness
                    )
                ):
                    logger.warning("Synthesized ranking function failed validation")
                    result = TerminationResult(
                        result=False, error="Ranking function validation failed"
                    )

            result.time = time.time() - start_time
            return result

        except Exception as e:
            logger.error("Error in termination proving: %s", e)
            return TerminationResult(
                result=False, time=time.time() - start_time, error=str(e)
            )

    def prove_non_termination(
        self, timeout: Optional[int] = None
    ) -> NonTerminationResult:
        """Prove non-termination by synthesizing a recurrence set."""
        if self.recurrence_template is None:
            raise ValueError(
                "No recurrence template set. Call set_recurrence_template() first."
            )

        start_time = time.time()

        try:
            # Generate and solve verification condition
            vc = self.recurrence_vc_generator.generate_vc(self.recurrence_template)

            if self.print_vc:
                print("Recurrence set verification condition:")
                print(vc)

            result = self.recurrence_solver.solve(vc, self.recurrence_template, timeout)

            # Validate if requested and successful
            if result.result and self.validate_recurrence_set:
                recurrence_set = self._extract_recurrence_set()
                if (
                    recurrence_set
                    and not self.recurrence_validator.validate_recurrence_set(
                        recurrence_set
                    )
                ):
                    logger.warning("Synthesized recurrence set failed validation")
                    result = NonTerminationResult(
                        result=False, error="Recurrence set validation failed"
                    )
                else:
                    result.recurrence_set = recurrence_set

            result.time = time.time() - start_time
            return result

        except Exception as e:
            logger.error("Error in non-termination proving: %s", e)
            return NonTerminationResult(
                result=False, time=time.time() - start_time, error=str(e)
            )

    def _extract_ranking_function(self) -> Optional[z3.ExprRef]:
        """Extract ranking function from template."""
        if hasattr(self.ranking_template, "build_invariant_expr"):
            dummy_model = self._create_dummy_model()
            if dummy_model:
                return self.ranking_template.build_invariant_expr(
                    dummy_model, use_prime_variables=False
                )
        return None

    def _extract_recurrence_set(self) -> Optional[z3.ExprRef]:
        """Extract recurrence set from template."""
        if hasattr(self.recurrence_template, "build_invariant_expr"):
            dummy_model = self._create_dummy_model()
            if dummy_model:
                return self.recurrence_template.build_invariant_expr(
                    dummy_model, use_prime_variables=False
                )
        return None

    def _create_dummy_model(self):
        """Create a dummy model for testing purposes."""
        solver = z3.Solver()
        solver.add(z3.BoolVal(True))
        return solver.model() if solver.check() == z3.sat else None
