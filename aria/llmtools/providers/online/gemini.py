# pylint: disable=invalid-name
"""Google Gemini provider."""

from __future__ import annotations

from typing import Optional

from aria.llmtools.core.base import BaseProvider, InferenceResult
from aria.llmtools.core.results import error_result, result_from_usage

try:
    import google.generativeai as genai  # pylint: disable=import-error

    GENAI_AVAILABLE = True
except ImportError:
    genai = None  # type: ignore
    GENAI_AVAILABLE = False


class GeminiProvider(BaseProvider):
    """Google Gemini provider."""

    def infer(
        self,
        message: str,
        system_role: str,
        temperature: float,
        max_output_length: int,
        model_name: Optional[str] = None,
    ) -> InferenceResult:
        """Run inference with Gemini."""
        if not GENAI_AVAILABLE:
            return error_result("Gemini SDK not installed")

        del max_output_length
        return self._call_api(message, system_role, temperature, model_name)

    def _call_api(
        self,
        message: str,
        system_role: str,
        temperature: float,
        model_name: Optional[str] = None,
    ) -> InferenceResult:
        """Call Gemini API."""
        model = genai.GenerativeModel(model_name or "gemini-2.0-flash")

        response = model.generate_content(
            "{0}\n{1}".format(system_role, message),
            safety_settings=[
                {"category": "HARM_CATEGORY_DANGEROUS", "threshold": "BLOCK_NONE"}
            ],
            generation_config=genai.types.GenerationConfig(temperature=temperature),
        )

        return result_from_usage(content=response.text or "", finish_reason="stop")
