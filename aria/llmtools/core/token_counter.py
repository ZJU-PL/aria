"""Token counting utilities."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

try:
    import tiktoken  # pylint: disable=import-error
except ImportError:
    tiktoken = None  # type: ignore


class TokenCounter:
    """Token counter using tiktoken."""

    def __init__(self) -> None:
        self.encoding = None
        if tiktoken is not None:
            self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-0125")

    def compute_costs(
        self,
        message: str,
        content: str,
        system_role: str,
        usage: Dict[str, int],
        is_measure_cost: bool,
    ) -> Tuple[int, int]:
        """
        Compute token costs for input and output.

        Args:
            message: User message
            content: LLM response content
            system_role: System role prompt
            usage: Usage dict from provider (may contain prompt_tokens, completion_tokens)
            is_measure_cost: Whether to measure costs

        Returns:
            Tuple of (input_tokens, output_tokens)
        """
        if not is_measure_cost:
            return 0, 0

        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if prompt_tokens is not None and completion_tokens is not None:
            return int(prompt_tokens), int(completion_tokens)

        if self.encoding is None:
            return 0, 0

        input_tokens = len(self.encoding.encode(system_role)) + len(
            self.encoding.encode(message)
        )
        output_tokens = len(self.encoding.encode(content)) if content else 0
        return input_tokens, output_tokens
