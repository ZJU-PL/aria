"""OpenAI Codex Responses provider."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, List, Optional, Tuple

import httpx
from loguru import logger
from oauth_cli_kit import get_token as get_codex_token  # type: ignore[import-untyped]

from aria.llmtools.providers.cli.base import (
    LLMProvider,
    LLMResponse,
    ToolCallRequest,
    error_response,
)

DEFAULT_CODEX_URL = "https://chatgpt.com/backend-api/codex/responses"
DEFAULT_ORIGINATOR = "aria"


class OpenAICodexProvider(LLMProvider):
    """Use Codex OAuth to call the Responses API."""

    def __init__(self, default_model: str = "openai-codex/gpt-5.1-codex"):
        super().__init__(api_key=None, api_base=None)
        self.default_model = default_model

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        on_retry: Optional[Callable[[int, int, float], Awaitable[None]]] = None,
    ) -> LLMResponse:
        del temperature

        model_name = model or self.default_model
        system_prompt, input_items = _convert_messages(messages)

        token = await asyncio.to_thread(get_codex_token)
        headers = _build_headers(token.account_id, token.access)

        body: Dict[str, Any] = {
            "model": _strip_model_prefix(model_name),
            "store": False,
            "stream": True,
            "instructions": system_prompt,
            "input": input_items,
            "text": {"verbosity": "medium"},
            "include": ["reasoning.encrypted_content"],
            "max_output_tokens": max_tokens,
            "prompt_cache_key": _prompt_cache_key(messages),
            "tool_choice": "auto",
            "parallel_tool_calls": True,
        }
        if tools:
            body["tools"] = _convert_tools(tools)

        verify = True
        max_attempts = 2
        for attempt in range(max_attempts):
            try:
                content, tool_calls, finish_reason = await _request_codex(
                    DEFAULT_CODEX_URL,
                    headers,
                    body,
                    verify=verify,
                )
                return LLMResponse(
                    content=content,
                    tool_calls=tool_calls,
                    finish_reason=finish_reason,
                )
            except Exception as exc:
                may_retry_insecure = (
                    "CERTIFICATE_VERIFY_FAILED" in str(exc)
                    and os.environ.get("ARIA_CODEX_ALLOW_INSECURE_SSL") == "1"
                )
                if may_retry_insecure and attempt == 0:
                    verify = False
                    delay = 0.0
                    if on_retry is not None:
                        await on_retry(attempt + 1, max_attempts, delay)
                    logger.warning(
                        "Codex SSL verification failed; retrying with verify=False "
                        "because ARIA_CODEX_ALLOW_INSECURE_SSL=1"
                    )
                    continue
                return LLMResponse(
                    content="Error calling Codex: {0}".format(exc),
                    finish_reason="error",
                )

        return LLMResponse(content="Error calling Codex", finish_reason="error")

    def get_default_model(self) -> str:
        return self.default_model


def _strip_model_prefix(model: str) -> str:
    if model.startswith("openai-codex/") or model.startswith("openai_codex/"):
        return model.split("/", 1)[1]
    return model


def _build_headers(account_id: str, token: str) -> Dict[str, str]:
    return {
        "Authorization": "Bearer {0}".format(token),
        "chatgpt-account-id": account_id,
        "OpenAI-Beta": "responses=experimental",
        "originator": DEFAULT_ORIGINATOR,
        "User-Agent": "aria (python)",
        "accept": "text/event-stream",
        "content-type": "application/json",
    }


async def _request_codex(
    url: str,
    headers: Dict[str, str],
    body: Dict[str, Any],
    verify: bool,
) -> Tuple[str, List[ToolCallRequest], str]:
    async with httpx.AsyncClient(timeout=60.0, verify=verify) as client:
        async with client.stream("POST", url, headers=headers, json=body) as response:
            if response.status_code != 200:
                text = await response.aread()
                raise RuntimeError(
                    _friendly_error(
                        response.status_code,
                        text.decode("utf-8", "ignore"),
                    )
                )
            return await _consume_sse(response)


def _convert_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    converted: List[Dict[str, Any]] = []
    for tool in tools:
        fn = (tool.get("function") or {}) if tool.get("type") == "function" else tool
        name = fn.get("name")
        if not name:
            continue
        params = fn.get("parameters") or {}
        converted.append(
            {
                "type": "function",
                "name": name,
                "description": fn.get("description") or "",
                "parameters": params if isinstance(params, dict) else {},
            }
        )
    return converted


def _convert_messages(
    messages: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    system_prompt = ""
    input_items: List[Dict[str, Any]] = []

    for idx, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content")

        if role == "system":
            system_prompt = content if isinstance(content, str) else ""
            continue

        if role == "user":
            input_items.append(_convert_user_message(content))
            continue

        if role == "assistant":
            if isinstance(content, str) and content:
                input_items.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": content}],
                        "status": "completed",
                        "id": "msg_{0}".format(idx),
                    }
                )
            for tool_call in msg.get("tool_calls", []) or []:
                fn = tool_call.get("function") or {}
                call_id, item_id = _split_tool_call_id(tool_call.get("id"))
                call_id = call_id or "call_{0}".format(idx)
                item_id = item_id or "fc_{0}".format(idx)
                input_items.append(
                    {
                        "type": "function_call",
                        "id": item_id,
                        "call_id": call_id,
                        "name": fn.get("name"),
                        "arguments": fn.get("arguments") or "{}",
                    }
                )
            continue

        if role == "tool":
            call_id, _ = _split_tool_call_id(msg.get("tool_call_id"))
            output_text = (
                content
                if isinstance(content, str)
                else json.dumps(content, ensure_ascii=False)
            )
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output_text,
                }
            )

    return system_prompt, input_items


def _convert_user_message(content: Any) -> Dict[str, Any]:
    if isinstance(content, str):
        return {"role": "user", "content": [{"type": "input_text", "text": content}]}

    if isinstance(content, list):
        converted: List[Dict[str, Any]] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                converted.append({"type": "input_text", "text": item.get("text", "")})
            elif item.get("type") == "image_url":
                url = (item.get("image_url") or {}).get("url")
                if url:
                    converted.append(
                        {"type": "input_image", "image_url": url, "detail": "auto"}
                    )
        if converted:
            return {"role": "user", "content": converted}

    return {"role": "user", "content": [{"type": "input_text", "text": ""}]}


def _split_tool_call_id(tool_call_id: Any) -> Tuple[str, Optional[str]]:
    if isinstance(tool_call_id, str) and tool_call_id:
        if "|" in tool_call_id:
            call_id, item_id = tool_call_id.split("|", 1)
            return call_id, item_id or None
        return tool_call_id, None
    return "call_0", None


def _prompt_cache_key(messages: List[Dict[str, Any]]) -> str:
    raw = json.dumps(messages, ensure_ascii=True, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def _iter_sse(response: httpx.Response) -> AsyncGenerator[Dict[str, Any], None]:
    buffer: List[str] = []
    async for line in response.aiter_lines():
        if line == "":
            if buffer:
                data_lines = [
                    item[5:].strip() for item in buffer if item.startswith("data:")
                ]
                buffer = []
                if not data_lines:
                    continue
                data = "\n".join(data_lines).strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    yield json.loads(data)
                except Exception:
                    continue
            continue
        buffer.append(line)


async def _consume_sse(
    response: httpx.Response,
) -> Tuple[str, List[ToolCallRequest], str]:
    content = ""
    tool_calls: List[ToolCallRequest] = []
    tool_call_buffers: Dict[str, Dict[str, Any]] = {}
    finish_reason = "stop"

    async for event in _iter_sse(response):
        event_type = event.get("type")
        if event_type == "response.output_item.added":
            item = event.get("item") or {}
            if item.get("type") == "function_call":
                call_id = item.get("call_id")
                if call_id:
                    tool_call_buffers[call_id] = {
                        "id": item.get("id") or "fc_0",
                        "name": item.get("name"),
                        "arguments": item.get("arguments") or "",
                    }
        elif event_type == "response.output_text.delta":
            content += event.get("delta") or ""
        elif event_type == "response.function_call_arguments.delta":
            call_id = event.get("call_id")
            if call_id and call_id in tool_call_buffers:
                tool_call_buffers[call_id]["arguments"] += event.get("delta") or ""
        elif event_type == "response.function_call_arguments.done":
            call_id = event.get("call_id")
            if call_id and call_id in tool_call_buffers:
                tool_call_buffers[call_id]["arguments"] = event.get("arguments") or ""
        elif event_type == "response.output_item.done":
            item = event.get("item") or {}
            if item.get("type") == "function_call":
                call_id = item.get("call_id")
                if not call_id:
                    continue
                buf = tool_call_buffers.get(call_id) or {}
                args_raw = buf.get("arguments") or item.get("arguments") or "{}"
                try:
                    args = json.loads(args_raw)
                except Exception:
                    args = {"raw": args_raw}
                if not isinstance(args, dict):
                    args = {"value": args}
                tool_calls.append(
                    ToolCallRequest(
                        id="{0}|{1}".format(
                            call_id,
                            buf.get("id") or item.get("id") or "fc_0",
                        ),
                        name=str(buf.get("name") or item.get("name") or ""),
                        arguments=args,
                    )
                )
        elif event_type == "response.completed":
            status = (event.get("response") or {}).get("status")
            finish_reason = _map_finish_reason(status)
        elif event_type in {"error", "response.failed"}:
            raise RuntimeError("Codex response failed")

    return content, tool_calls, finish_reason


_FINISH_REASON_MAP = {
    "completed": "stop",
    "incomplete": "length",
    "failed": "error",
    "cancelled": "error",
}


def _map_finish_reason(status: Optional[str]) -> str:
    return _FINISH_REASON_MAP.get(status or "completed", "stop")


def _friendly_error(status_code: int, raw: str) -> str:
    if status_code == 429:
        return "ChatGPT quota exceeded or rate limit triggered. Try again later."
    return "HTTP {0}: {1}".format(status_code, raw)
