"""Abstract base classes for LLM-backed tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional

from aria.llmtools import LLM, Logger


class LLMToolInput(ABC):
    """Abstract base class for LLM tool input."""

    @abstractmethod
    def __hash__(self):
        raise NotImplementedError

    def __eq__(self, value):
        return self is value


class LLMToolOutput(ABC):
    """Abstract base class for LLM tool output."""


class LLMTool(ABC):
    """Abstract base class for LLM-based tools."""

    def __init__(
        self,
        model_name: str,
        temperature: float,
        language: str,
        max_query_num: int,
        logger: Logger,
    ) -> None:
        self.language = language
        self.model_name = model_name
        self.temperature = temperature
        self.max_query_num = max_query_num
        self.logger = logger
        self.model = LLM(model_name, self.logger, temperature)
        self.cache: Dict[LLMToolInput, LLMToolOutput] = {}
        self.input_token_cost = 0
        self.output_token_cost = 0
        self.total_query_num = 0

    def invoke(self, tool_input: LLMToolInput) -> Optional[LLMToolOutput]:
        """Invoke the tool with caching and bounded retries."""
        class_name = type(self).__name__
        self.logger.print_console(f"The LLM Tool {class_name} is invoked.")
        if tool_input in self.cache:
            self.logger.print_log("Cache hit.")
            return self.cache[tool_input]

        prompt = self._get_prompt(tool_input)
        self.logger.print_log("Prompt:", "\n", prompt)

        single_query_num = 0
        output: Optional[LLMToolOutput] = None
        while single_query_num < self.max_query_num:
            single_query_num += 1
            response, input_token_cost, output_token_cost = self.model.infer(prompt, True)
            self.logger.print_log("Response:", "\n", response)
            self.input_token_cost += input_token_cost
            self.output_token_cost += output_token_cost
            output = self._parse_response(response, tool_input)
            if output is not None:
                break

        self.total_query_num += single_query_num
        if output is not None:
            self.cache[tool_input] = output
        return output

    @abstractmethod
    def _get_prompt(self, tool_input: LLMToolInput) -> str:
        raise NotImplementedError

    @abstractmethod
    def _parse_response(
        self, response: str, tool_input: Optional[LLMToolInput] = None
    ) -> Optional[LLMToolOutput]:
        raise NotImplementedError
