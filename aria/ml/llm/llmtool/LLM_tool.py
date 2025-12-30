# pylint: disable=invalid-name
from abc import ABC, abstractmethod
from typing import Dict
from aria.ml.llm.llmtool.LLM_utils import *
from aria.ml.llm.llmtool.logger import Logger


class LLMToolInput(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __hash__(self):
        pass

    def __eq__(self, value):
        return self.__hash__() == value.__hash__()


class LLMToolOutput(ABC):
    def __init__(self):
        pass


class LLMTool(ABC):
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
        self.language = language
        self.max_query_num = max_query_num
        self.logger = logger

        self.model = LLM(model_name, self.logger, temperature)
        self.cache: Dict[LLMToolInput, LLMToolOutput] = {}

        self.input_token_cost = 0
        self.output_token_cost = 0
        self.total_query_num = 0

    def invoke(self, input_data: LLMToolInput) -> LLMToolOutput:  # pylint: disable=redefined-builtin
        class_name = type(self).__name__
        self.logger.print_console(f"The LLM Tool {class_name} is invoked.")
        if input_data in self.cache:
            self.logger.print_log("Cache hit.")
            return self.cache[input_data]

        prompt = self._get_prompt(input_data)
        self.logger.print_log("Prompt:", "\n", prompt)

        single_query_num = 0
        output = None
        while True:
            if single_query_num > self.max_query_num:
                break
            single_query_num += 1
            response, input_token_cost, output_token_cost = self.model.infer(
                prompt, True
            )
            self.logger.print_log("Response:", "\n", response)
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
    def _get_prompt(self, input_data: LLMToolInput) -> str:  # pylint: disable=redefined-builtin
        pass

    @abstractmethod
    def _parse_response(
        self, response: str, input_data: LLMToolInput = None  # pylint: disable=redefined-builtin
    ) -> LLMToolOutput:
        pass
