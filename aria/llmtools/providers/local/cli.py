# pylint: disable=invalid-name
"""Local CLI provider wrapper."""

from __future__ import annotations

from typing import Optional

from aria.llmtools.core.base import BaseProvider, InferenceResult
from aria.llmtools.logger import Logger
from aria.llmtools.providers.cli.local_cli import LLMCli


class LocalCLIProvider(BaseProvider):
    """Local CLI provider wrapper (kilo-cli, opencode-cli)."""

    def __init__(self, model_name: str, logger: Logger, temperature: float):
        self.cli_provider = LLMCli(
            model_name=model_name,
            logger=logger,
            temperature=temperature,
            system_role="",
            measure_cost=False,
        )

    def infer(
        self,
        message: str,
        system_role: str,
        temperature: float,
        max_output_length: int,
    ) -> InferenceResult:
        """Run inference with local CLI."""
        output, _, _ = self.cli_provider.infer(message)

        if not output:
            return InferenceResult(
                content="",
                input_tokens=0,
                output_tokens=0,
                finish_reason="error",
                error="Empty CLI response",
            )

        return InferenceResult(
            content=output,
            input_tokens=0,
            output_tokens=0,
            finish_reason="stop",
        )
