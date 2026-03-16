import logging
from typing import Optional, Dict, Any
from aria.efmc.sts import TransitionSystem
from aria.efmc.utils.verification_utils import VerificationResult
from aria.efmc.engines.llm4inv.bv.cegis_loop import LLMInvariantCEGIS
from aria.efmc.engines.llm4inv.bv.prompt_manager import extract_bit_width_from_sts

logger = logging.getLogger(__name__)


class LLM4InvProver:
    def __init__(self, sts: TransitionSystem, **kwargs):
        self.sts = sts
        self.timeout = kwargs.get("timeout", 600)
        self.max_iterations = kwargs.get("max_iterations", 10)
        self.bit_width = kwargs.get("bit_width") or extract_bit_width_from_sts(sts)
        self.llm_model = kwargs.get("llm_model", "deepseek-v3")
        self.cegis = LLMInvariantCEGIS(sts, **kwargs)
        self.result: Optional[VerificationResult] = None

    def solve(self, timeout: Optional[int] = None) -> VerificationResult:
        if timeout is not None:
            self.timeout = timeout
            self.cegis.timeout = timeout

        logger.info("Starting LLM4Inv synthesis")
        program_desc = self._generate_program_description()
        invariant = self.cegis.synthesize_invariant(program_desc)

        if invariant is not None:
            self.result = VerificationResult(True, invariant)
            logger.info("Found invariant: %s", invariant)
        else:
            self.result = VerificationResult(False, None)
            logger.warning("Synthesis failed")

        return self.result

    def _generate_program_description(self) -> str:
        vars_str = ", ".join(str(v) for v in self.sts.variables)
        return f"Variables: {vars_str}\nBit width: {self.bit_width}\nProgram: initial + transition + safety"

    def get_statistics(self) -> Dict[str, Any]:
        stats = {"solve_time": self.cegis.stats.get("total_time", 0.0),
                 "timeout": self.timeout, "max_iterations": self.max_iterations,
                 "bit_width": self.bit_width, "llm_model": self.llm_model,
                 "success": self.result.is_safe if self.result else False}
        stats.update(self.cegis.get_statistics())
        return stats

    def set_timeout(self, timeout: int):
        self.timeout = timeout
        self.cegis.timeout = timeout

    def reset(self):
        self.result = None
        self.cegis.reset()

    def __repr__(self) -> str:
        return f"LLM4InvProver(model={self.llm_model}, timeout={self.timeout})"
