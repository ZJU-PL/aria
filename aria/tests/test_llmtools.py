"""Tests for aria.llmtools robustness and routing behavior."""

import tempfile
from typing import Optional

from aria.llmtools.providers.cli.base import LLMResponse
from aria.llmtools.local_client import LLMLocal
from aria.llmtools.tool import LLMTool, LLMToolInput, LLMToolOutput
from aria.llmtools.client import LLM
from aria.llmtools.logger import Logger
from aria.tests import TestCase, main


class _DummyInput(LLMToolInput):
    def __init__(self, value: str, fake_hash: int = 0) -> None:
        self.value = value
        self.fake_hash = fake_hash

    def __hash__(self) -> int:
        return self.fake_hash


class _DummyOutput(LLMToolOutput):
    def __init__(self, value: str) -> None:
        self.value = value


class _StubModel:
    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.calls = 0

    def infer(self, prompt: str, measure_cost: bool):
        del prompt, measure_cost
        self.calls += 1
        if not self.responses:
            return "", 0, 0
        return self.responses.pop(0), 0, 0


class _DummyTool(LLMTool):
    def _get_prompt(self, input_data: LLMToolInput) -> str:
        del input_data
        return "prompt"

    def _parse_response(
        self,
        response: str,
        input_data: Optional[LLMToolInput] = None,
    ) -> Optional[LLMToolOutput]:
        del input_data
        if response == "ok":
            return _DummyOutput(response)
        return None


class TestLlmTools(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._tmpdir = tempfile.TemporaryDirectory()
        self.logger = Logger("{0}/llmtools_test.log".format(self._tmpdir.name))

    def tearDown(self) -> None:
        self._tmpdir.cleanup()
        super().tearDown()

    def test_tool_input_equality_not_hash_only(self) -> None:
        left = _DummyInput("a", fake_hash=1)
        right = _DummyInput("b", fake_hash=1)
        self.assertNotEqual(left, right)

    def test_tool_retry_respects_max_query_num(self) -> None:
        tool = _DummyTool(
            model_name="unsupported-model",
            temperature=0.0,
            language="en",
            max_query_num=2,
            logger=self.logger,
        )
        tool.model = _StubModel(["", "", "ok"])

        output = tool.invoke(_DummyInput("x", fake_hash=3))

        self.assertIsNone(output)
        self.assertEqual(tool.model.calls, 2)

    def test_llm_returns_explicit_error_for_unsupported_model(self) -> None:
        llm = LLM(online_model_name="unknown-model", logger=self.logger)

        response, _, _ = llm.infer("hello", is_measure_cost=False)

        self.assertIn("[LLM ERROR]", response)
        self.assertIn("Unsupported model name", response)

    def test_llm_local_returns_explicit_error_for_unsupported_provider(self) -> None:
        llm = LLMLocal(
            offline_model_name="local-model",
            logger=self.logger,
            provider="unsupported",
        )

        response, _, _ = llm.infer("hello", is_measure_cost=False)

        self.assertIn("[LLM ERROR]", response)
        self.assertIn("Unsupported provider", response)

    def test_o3_model_routes_to_openai_provider(self) -> None:
        llm = LLM(online_model_name="o3-mini", logger=self.logger)

        called = {"value": False}

        from aria.llmtools.core.base import InferenceResult

        def _fake_get_provider(_model_name: str):
            called["value"] = True
            from aria.llmtools.providers.online.openai import OpenAIProvider

            class FakeProvider(OpenAIProvider):
                def infer(self, message, system_role, temperature, max_output_length):
                    return InferenceResult(
                        content="ok", input_tokens=0, output_tokens=0, finish_reason="stop"
                    )

            return FakeProvider()

        import aria.llmtools.client as client_module

        original_get_provider = client_module.get_online_provider
        client_module.get_online_provider = _fake_get_provider  # type: ignore[assignment]

        try:
            result = llm.infer_response("test", is_measure_cost=False)
            self.assertTrue(called["value"])
            self.assertEqual(result.content, "ok")
        finally:
            client_module.get_online_provider = original_get_provider


if __name__ == "__main__":
    main()
