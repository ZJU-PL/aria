# pylint: disable=invalid-name
"""Main routed LLM client."""

from __future__ import annotations

from typing import Optional

from aria.llmtools.core.client import BaseLLMClient
from aria.llmtools.core.logger import Logger
from aria.llmtools.routing import resolve_provider


class LLM(BaseLLMClient):
    """Routed LLM inference client across online and local providers."""

    def __init__(
        self,
        model_name: str,
        logger: Logger,
        temperature: float = 0.0,
        system_role: str = "You are an experienced programmer.",
        max_output_length: int = 4096,
        provider: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.provider = provider
        super().__init__(
            model_name=model_name,
            logger=logger,
            resolver=lambda: resolve_provider(
                self.model_name,
                self.provider,
                self.logger,
                self.temperature,
            ),
            unsupported_error=self._unsupported_error_message,
            temperature=temperature,
            system_role=system_role,
            max_output_length=max_output_length,
        )

    def _unsupported_error_message(self) -> str:
        """Return the compatibility error message for unsupported routing."""
        if self.provider is not None:
            return "Unsupported provider: {0}".format(self.provider)
        return "Unsupported model name: {0}".format(self.model_name)
