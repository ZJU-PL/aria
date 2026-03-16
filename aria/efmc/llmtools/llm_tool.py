"""LLM Tool base classes for implementing LLM-based tools."""

from abc import ABC, abstractmethod
from typing import Dict
from aria.efmc.llmtools.logger import Logger
from aria.efmc.llmtools.llm_utils import LLM


class LLMToolInput(ABC):
    """Abstract base class for LLM tool input."""

    def __init__(self):
        pass

    @abstractmethod
    def __hash__(self):
        pass

    def __eq__(self, value):
        return self is value


class LLMToolOutput(ABC):
    """Abstract base class for LLM tool output."""

    def __init__(self):
        pass


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
        """
        Initialize LLM Tool.

        Args:
            model_name: Name of the LLM model to use
            temperature: Temperature for model inference
            language: Programming language for the tool
            max_query_num: Maximum number of queries per invocation
            logger: Logger instance for logging
        """
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

    def invoke(self, tool_input: LLMToolInput) -> LLMToolOutput:
        """
        Invoke the LLM tool with given input.

        Args:
            tool_input: Input to the LLM tool

        Returns:
            Output from the LLM tool
        """
        class_name = type(self).__name__
        self.logger.print_console(f"The LLM Tool {class_name} is invoked.")
        if tool_input in self.cache:
            self.logger.print_log("Cache hit.")
            return self.cache[tool_input]

        prompt = self._get_prompt(tool_input)
        self.logger.print_log("Prompt:", "\n", prompt)

        single_query_num = 0
        output = None
        while single_query_num < self.max_query_num:
            single_query_num += 1
            response, input_token_cost, output_token_cost = self.model.infer(
                prompt, True
            )
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
        """Generate prompt from input."""
        pass

    @abstractmethod
    def _parse_response(
        self, response: str, tool_input: LLMToolInput = None
    ) -> LLMToolOutput:
        """Parse response from LLM into output."""
        pass
