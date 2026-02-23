# pylint: disable=invalid-name
"""Local LLM client for offline providers."""

from __future__ import annotations

from typing import Tuple

from aria.llmtools.core.base import InferenceResult
from aria.llmtools.core.retry import retry_with_backoff
from aria.llmtools.core.token_counter import TokenCounter
from aria.llmtools.logger import Logger
from aria.llmtools.registry import get_local_provider


class LLMLocal:
    """Local LLM inference: vLLM, SGLang, LM Studio, CLI providers."""

    def __init__(
        self,
        offline_model_name: str,
        logger: Logger,
        temperature: float = 0.0,
        system_role: str = (
            "You are an experienced programmer and good at understanding "
            "programs written in mainstream programming languages."
        ),
        max_output_length: int = 4096,
        provider: str = "lm-studio",
    ) -> None:
        self.offline_model_name = offline_model_name
        self.temperature = temperature
        self.systemRole = system_role
        self.logger = logger
        self.max_output_length = max_output_length
        self.provider = provider
        self.token_counter = TokenCounter()

    def infer(
        self, message: str, is_measure_cost: bool = False
    ) -> Tuple[str, int, int]:
        """Backward-compatible tuple API."""
        result = self.infer_response(message, is_measure_cost=is_measure_cost)
        content = result.content
        if result.error:
            content = "[LLM ERROR] {0}".format(result.error)
        return content, result.input_tokens, result.output_tokens

    def infer_response(
        self, message: str, is_measure_cost: bool = False
    ) -> InferenceResult:
        """Structured inference API with explicit error information."""
        self.logger.print_log(self.offline_model_name, "is running")

        provider = get_local_provider(
            self.offline_model_name, self.provider, self.logger, self.temperature
        )
        if provider is None:
            return InferenceResult(
                content="",
                input_tokens=0,
                output_tokens=0,
                finish_reason="error",
                error="Unsupported provider: {0}".format(self.provider),
            )

        timeout = 300 if self.provider in ["vllm", "sglang", "lm-studio"] else 120

        def call_func() -> InferenceResult:
            return provider.infer(
                message=message,
                system_role=self.systemRole,
                temperature=self.temperature,
                max_output_length=self.max_output_length,
            )

        result = retry_with_backoff(call_func, self.logger, timeout=timeout)

        input_tokens, output_tokens = self.token_counter.compute_costs(
            message=message,
            content=result.content,
            system_role=self.systemRole,
            usage=result.usage,
            is_measure_cost=is_measure_cost,
        )

        result.input_tokens = input_tokens
        result.output_tokens = output_tokens

        return result
