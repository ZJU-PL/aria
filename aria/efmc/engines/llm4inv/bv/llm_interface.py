import logging
from typing import Dict, Any, Optional, Tuple, Callable, List
import z3
from aria.llmtools import LLM
from aria.llmtools.core.logger import Logger
from aria.llmtools.local_client import LLMLocal
from aria.efmc.sts import TransitionSystem
from aria.efmc.engines.llm4inv.bv.prompt_manager import extract_bit_width_from_sts

logger = logging.getLogger(__name__)


class LLMInterface:
    def __init__(self, sts: TransitionSystem, model_name: str = "deepseek-v3", **kwargs):
        self.sts = sts
        self.model_name = model_name
        self.temperature = kwargs.get("temperature", 0.1)
        self.llm_provider = kwargs.get("llm_provider", "local")
        self.bit_width = kwargs.get("bit_width") or extract_bit_width_from_sts(sts)

        self.logger = Logger("llm4inv.log")

        if self.llm_provider == "local":
            local_base_url: Optional[str] = kwargs.get("local_base_url")
            local_api_key: Optional[str] = kwargs.get("local_api_key")
            self.llm = LLMLocal(
                offline_model_name=kwargs.get("llm_model", "qwen/qwen3-coder-30b"),
                logger=self.logger,
                temperature=self.temperature,
                max_output_length=kwargs.get("max_output_length", 4096),
                measure_cost=kwargs.get("measure_cost", False),
                provider=kwargs.get("local_provider", "lm-studio"),
                base_url=local_base_url,
                api_key=local_api_key,
                max_retries=kwargs.get("local_max_retries", 3),
            )
        else:
            self.llm = LLM(model_name, self.logger, temperature=self.temperature)

        self.stats = {"total_llm_calls": 0, "successful_generations": 0,
                      "total_candidates_generated": 0, "refinement_calls": 0}

    def generate_candidates(self, prompt_generator: Callable[[], str],
                           response_parser: Callable[[str], List[Tuple[str, z3.ExprRef]]],
                           context: Optional[Dict[str, Any]] = None) -> List[Tuple[str, z3.ExprRef]]:
        prompt = prompt_generator()
        self.stats["total_llm_calls"] += 1
        resp, _, _ = self.llm.infer(prompt, is_measure_cost=False)

        try:
            pairs = response_parser(resp)
            if pairs:
                self.stats["successful_generations"] += 1
                self.stats["total_candidates_generated"] += len(pairs)
            return pairs
        except (ValueError, AttributeError, TypeError, KeyError) as e:
            logger.error("Parse failed: %s", e)
            return []

    def call_llm(self, prompt: str, **kwargs) -> str:
        self.stats["total_llm_calls"] += 1
        resp, _, _ = self.llm.infer(prompt, is_measure_cost=kwargs.get("measure_cost", False))
        return resp

    def get_statistics(self) -> Dict[str, Any]:
        return self.stats.copy()
