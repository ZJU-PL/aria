# pylint: disable=invalid-name
"""Base classes for prompt-driven LLM tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional

from aria.llmtools.client import LLM
from aria.llmtools.logger import Logger


class LLMToolInput(ABC):
    """Input payload for an LLMTool."""

    @abstractmethod
    def __hash__(self) -> int:
        """Subclasses must provide a stable hash."""
        raise NotImplementedError

    def __eq__(self, value: object) -> bool:
        if self is value:
            return True
        if value is None or type(self) is not type(value):
            return False
        other = value
        if getattr(self, "__dict__", None) and getattr(other, "__dict__", None):
            return self.__dict__ == other.__dict__
        return hash(self) == hash(other)


class LLMToolOutput(ABC):
    """Output payload for an LLMTool."""


class LLMTool(ABC):
    """Base class for tools that generate/parse LLM prompts."""

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

    def invoke(self, input_data: LLMToolInput) -> Optional[LLMToolOutput]:
        class_name = type(self).__name__
        self.logger.print_console("The LLM Tool {0} is invoked.".format(class_name))

        if input_data in self.cache:
            self.logger.print_log("Cache hit.")
            return self.cache[input_data]

        prompt = self._get_prompt(input_data)
        self.logger.print_log("Prompt:\n{0}".format(prompt))

        output = None
        single_query_num = 0
        for single_query_num in range(1, self.max_query_num + 1):
            response, input_token_cost, output_token_cost = self.model.infer(
                prompt,
                True,
            )
            self.logger.print_log("Response:\n{0}".format(response))

            self.input_token_cost += input_token_cost
            self.output_token_cost += output_token_cost
            output = self._parse_response(response, input_data)
            if output is not None:
                break

        self.total_query_num += single_query_num

        if output is not None:
            self.cache[input_data] = output
        return output

    @abstractmethod
    def _get_prompt(self, input_data: LLMToolInput) -> str:
        raise NotImplementedError

    @abstractmethod
    def _parse_response(
        self,
        response: str,
        input_data: Optional[LLMToolInput] = None,
    ) -> Optional[LLMToolOutput]:
        raise NotImplementedError
