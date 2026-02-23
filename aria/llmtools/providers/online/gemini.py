# pylint: disable=invalid-name
"""Google Gemini provider."""

from __future__ import annotations

from typing import Optional

from aria.llmtools.core.base import BaseProvider, InferenceResult

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
    ) -> InferenceResult:
        """Run inference with Gemini."""
        if not GENAI_AVAILABLE:
            return InferenceResult(
                content="",
                input_tokens=0,
                output_tokens=0,
                finish_reason="error",
                error="Gemini SDK not installed",
            )

        return self._call_api(message, system_role, temperature)

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

        return InferenceResult(
            content=response.text or "",
            input_tokens=0,
            output_tokens=0,
            finish_reason="stop",
        )
