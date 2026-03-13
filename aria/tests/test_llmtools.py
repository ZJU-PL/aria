"""Tests for aria.llmtools robustness and routing behavior."""

import tempfile

from aria.llmtools.core.client import ProviderResolution
from aria.llmtools import LLM, Logger
from aria.llmtools.routing import resolve_provider
from aria.tests import TestCase, main


class TestLlmTools(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._tmpdir = tempfile.TemporaryDirectory()
        self.logger = Logger("{0}/llmtools_test.log".format(self._tmpdir.name))

    def tearDown(self) -> None:
        self._tmpdir.cleanup()
        super().tearDown()

    def test_llm_returns_explicit_error_for_unsupported_model(self) -> None:
        llm = LLM(model_name="unknown-model", logger=self.logger)

        response, _, _ = llm.infer("hello", is_measure_cost=False)

        self.assertIn("[LLM ERROR]", response)
        self.assertIn("Unsupported model name", response)

    def test_llm_returns_explicit_error_for_unsupported_provider(self) -> None:
        llm = LLM(model_name="local-model", logger=self.logger, provider="unsupported")

        response, _, _ = llm.infer("hello", is_measure_cost=False)

        self.assertIn("[LLM ERROR]", response)
        self.assertIn("Unsupported provider", response)

    def test_o3_model_routes_to_openai_provider(self) -> None:
        llm = LLM(model_name="o3-mini", logger=self.logger)

        called = {"value": False}
        captured = {"model_name": None}

        from aria.llmtools.core.base import InferenceResult

        def _fake_resolve_provider(
            _model_name: str,
            _provider=None,
            _logger=None,
            _temperature: float = 0.0,
        ):
            called["value"] = True
            from aria.llmtools.providers.online.openai import OpenAIProvider

            class FakeProvider(OpenAIProvider):
                def infer(
                    self,
                    message,
                    system_role,
                    temperature,
                    max_output_length,
                    model_name=None,
                ):
                    del message, system_role, temperature, max_output_length
                    captured["model_name"] = model_name
                    return InferenceResult(
                        content="ok", input_tokens=0, output_tokens=0, finish_reason="stop"
                    )

            return ProviderResolution(provider=FakeProvider(), timeout=100)

        import aria.llmtools.client as client_module

        original_resolve_provider = client_module.resolve_provider
        client_module.resolve_provider = _fake_resolve_provider  # type: ignore[assignment]

        try:
            result = llm.infer_response("test", is_measure_cost=False)
            self.assertTrue(called["value"])
            self.assertEqual(result.content, "ok")
            self.assertEqual(captured["model_name"], "o3-mini")
        finally:
            client_module.resolve_provider = original_resolve_provider

    def test_resolve_provider_returns_none_for_unknown_model(self) -> None:
        self.assertIsNone(resolve_provider("unknown-model", None))


if __name__ == "__main__":
    main()
