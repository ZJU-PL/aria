"""GLM provider backed by ZhipuAI."""

from __future__ import annotations

from aria.llmtools.core.base import BaseProvider, InferenceResult
from aria.llmtools.core.results import error_result, result_from_usage


class GLMProvider(BaseProvider):
    """Provider for GLM models through the `zhipuai` SDK."""

    def infer(
        self,
        message: str,
        system_role: str,
        temperature: float,
        max_output_length: int,
        model_name: str | None = None,
    ) -> InferenceResult:
        del max_output_length
        try:
            from zhipuai import ZhipuAI  # pylint: disable=import-outside-toplevel
        except ImportError:
            return error_result(
                "zhipuai package not installed. Please install it to use GLM models."
            )

        import os  # pylint: disable=import-outside-toplevel

        api_key = os.environ.get("GLM_API_KEY")
        if not api_key:
            return error_result("GLM_API_KEY is not set")

        try:
            client = ZhipuAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model_name or "glm-4-flash",
                messages=[
                    {"role": "system", "content": system_role},
                    {"role": "user", "content": message},
                ],
                temperature=temperature,
            )
            usage = getattr(response, "usage", None)
            usage_dict = {}
            if usage is not None:
                usage_dict = {
                    "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                    "completion_tokens": int(
                        getattr(usage, "completion_tokens", 0) or 0
                    ),
                    "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
                }
            content = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason or "stop"
            return result_from_usage(content, finish_reason, usage_dict)
        except Exception as exc:  # pragma: no cover - network path
            return error_result(f"GLM error: {exc}")
