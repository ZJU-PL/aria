# pylint: disable=invalid-name
"""Shared client execution flow for LLM providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from aria.llmtools.core.base import BaseProvider, InferenceResult
from aria.llmtools.core.logger import Logger
from aria.llmtools.core.retry import retry_with_backoff
from aria.llmtools.core.token_counter import TokenCounter


@dataclass(frozen=True)
class ProviderResolution:
    """Resolved provider configuration for a client request."""

    provider: BaseProvider
    timeout: int


ProviderResolver = Callable[[], Optional[ProviderResolution]]
UnsupportedErrorResolver = Callable[[], str]


class BaseLLMClient:
    """Shared inference client for all provider families."""

    def __init__(
        self,
        model_name: str,
        logger: Logger,
        resolver: ProviderResolver,
        unsupported_error: UnsupportedErrorResolver,
        temperature: float = 0.0,
        system_role: str = "You are an experienced programmer.",
        max_output_length: int = 4096,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.systemRole = system_role
        self.logger = logger
        self.max_output_length = max_output_length
        self._resolver = resolver
        self._unsupported_error = unsupported_error
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
        self.logger.print_log(self.model_name, "is running")

        resolution = self._resolver()
        if resolution is None:
            return InferenceResult(
                content="",
                input_tokens=0,
                output_tokens=0,
                finish_reason="error",
                error=self.get_unsupported_error(),
            )

        def call_func() -> InferenceResult:
            return resolution.provider.infer(
                message=message,
                system_role=self.systemRole,
                temperature=self.temperature,
                max_output_length=self.max_output_length,
                model_name=self.model_name,
            )

        result = retry_with_backoff(call_func, self.logger, timeout=resolution.timeout)

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

    def get_unsupported_error(self) -> str:
        """Return the error when the provider cannot be resolved."""
        return self._unsupported_error()
