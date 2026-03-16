import logging
import time
from typing import Optional, Dict, Any, List, Tuple
import z3
from aria.efmc.sts import TransitionSystem
from aria.efmc.utils.verification_utils import verify_invariant_with_counterexamples
from aria.efmc.engines.llm4inv.bv.llm_interface import LLMInterface
from aria.efmc.engines.llm4inv.bv.prompt_manager import InvariantPromptManager

logger = logging.getLogger(__name__)


class LLMInvariantCEGIS:
    def __init__(self, sts: TransitionSystem, **kwargs):
        self.sts = sts
        self.max_iterations = kwargs.get("max_iterations", 10)
        self.max_candidates_per_iter = kwargs.get("max_candidates_per_iteration", 5)
        self.timeout = kwargs.get("timeout", 600)
        self.llm_interface = LLMInterface(sts, **kwargs)
        self.prompt_manager = InvariantPromptManager(sts, kwargs.get("bit_width"))
        self.failed_candidates: List[str] = []
        self.counterexamples: List = []
        self.stats: Dict[str, Any] = {"iterations": 0, "candidates": 0, "successful": 0}

    def synthesize_invariant(self, program_code: str = "") -> Optional[z3.ExprRef]:
        start_time = time.time()
        try:
            for _ in range(self.max_iterations):
                self.stats["iterations"] += 1
                if time.time() - start_time > self.timeout:
                    break

                result = self._cegis_iteration(program_code)
                if result is not None:
                    self.stats["successful"] += 1
                    return result

                if self.counterexamples:
                    self._update_approach()
            return None
        finally:
            self.stats["total_time"] = time.time() - start_time

    def _cegis_iteration(self, program_code: str) -> Optional[z3.ExprRef]:
        prompt_gen = self.prompt_manager.create_prompt_generator(
            program_code=program_code,
            failed_candidates=self.failed_candidates[-5:],
            max_candidates=self.max_candidates_per_iter,
        )
        candidates = self.llm_interface.generate_candidates(prompt_gen, self.prompt_manager.parse_invariant_response)
        self.stats["candidates"] += len(candidates)

        for cand_str, cand_expr in candidates:
            is_correct, cex = verify_invariant_with_counterexamples(self.sts, cand_expr)
            if is_correct:
                return cand_expr
            self.failed_candidates.append(cand_str)
            self.counterexamples.extend(cex)
        return None

    def _update_approach(self):
        self.counterexamples = self.counterexamples[-10:]
        self.failed_candidates = self.failed_candidates[-10:]

    def get_statistics(self) -> Dict[str, Any]:
        stats = self.stats.copy()
        if self.stats["iterations"] > 0:
            stats["success_rate"] = self.stats["successful"] / self.stats["iterations"]
        return stats

    def reset(self):
        self.failed_candidates.clear()
        self.counterexamples.clear()
        for key in self.stats:
            self.stats[key] = 0
