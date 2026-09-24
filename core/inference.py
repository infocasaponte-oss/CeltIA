# Copyright (c) 2026 Luis Manuel Cousido Hermida. All rights reserved.
import json
import logging
from typing import Any
import httpx

from core import diagnostics
from core.config import settings

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 3
TOOL_SCHEMA_TOKEN_OVERHEAD = 120
RESERVED_OUTPUT_MARGIN = 64
MIN_OUTPUT_TOKENS = 220


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def _fit_to_context(messages, tools, desired_max_tokens, context_limit):
    """Drop oldest non-system messages and clamp max_tokens so prompt + output
    never exceeds the model's context window."""
    tool_overhead = TOOL_SCHEMA_TOKEN_OVERHEAD * len(tools) if tools else 0
    working = list(messages)

    def prompt_tokens():
        return sum(_estimate_tokens(m.get("content") or "") for m in working) + tool_overhead

    while len(working) > 1 and prompt_tokens() + MIN_OUTPUT_TOKENS > context_limit:
        drop_index = 1 if working[0].get("role") == "system" else 0
        if len(working) <= drop_index + 1:
            break
        working.pop(drop_index)

    remaining = context_limit - prompt_tokens() - RESERVED_OUTPUT_MARGIN
    max_tokens = max(MIN_OUTPUT_TOKENS, min(desired_max_tokens, remaining))
    return working, max_tokens


class VLLMClient:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def _chat_streaming(self, payload, timeout, on_token):
        payload = {**payload, "stream": True, "stream_options": {"include_usage": True}}
        content_parts: list[str] = []
        calls: dict[int, dict] = {}
        usage = None
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", f"{self.base_url}/chat/completions", json=payload) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    body = line[5:].strip()
                    if body == "[DONE]":
                        break
                    chunk = json.loads(body)
                    usage = chunk.get("usage") or usage
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        if delta.get("content"):
                            content_parts.append(delta["content"])
                            on_token(delta["content"])
                        for tc in delta.get("tool_calls") or []:
                            slot = calls.setdefault(tc.get("index", 0), {
                                "id": tc.get("id"), "type": "function", "function": {"name": "", "arguments": ""}})
                            if tc.get("id"):
                                slot["id"] = tc["id"]
                            fn = tc.get("function") or {}
                            if fn.get("name"):
                                slot["function"]["name"] += fn["name"]
                            if fn.get("arguments"):
                                slot["function"]["arguments"] += fn["arguments"]
        tool_calls = [calls[i] for i in sorted(calls)]
        if tool_calls and content_parts:
            on_token(None)
        message = {"role": "assistant", "content": "".join(content_parts), "tool_calls": tool_calls}
        out = {"choices": [{"message": message}]}
        if usage:
            out["usage"] = usage
        return out

    async def chat(self, messages: list[dict[str, Any]], *, thinking: bool,
                   max_tokens: int, tools=None, temperature=None, on_token=None):
        """`on_token(delta)` streams content as it is generated; `on_token(None)` signals that
        the text streamed so far was preamble to a tool call and should be discarded."""
        fitted_messages, fitted_max_tokens = _fit_to_context(messages, tools, max_tokens, settings.model_context)
        payload = {
            "model": self.model,
            "messages": fitted_messages,
            "max_tokens": fitted_max_tokens,
            "temperature": temperature if temperature is not None else (1.0 if thinking else 0.7),
            "top_p": 0.95,
            "chat_template_kwargs": {"enable_thinking": thinking},
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        timeout = max(60, fitted_max_tokens * 1.0)
        try:
            if on_token is not None:
                return await self._chat_streaming(payload, timeout, on_token)
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(f"{self.base_url}/chat/completions", json=payload)
                r.raise_for_status()
                return r.json()
        except Exception as exc:
            logger.exception("vLLM backend call failed, falling back to demo mode")
            diagnostics.record_error("vllm_backend_call", f"{type(exc).__name__}: {exc}")
            user_prompt = messages[-1].get("content", "") if isinstance(messages, list) and messages else ""
            reply = (
                "Soy CeltIA V4 y estoy en modo demo porque el backend local no está disponible. "
                f"Tu mensaje fue: {user_prompt[:200]}"
            )
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": reply,
                        "tool_calls": [],
                    }
                }],
                "meta": {"offline_fallback": True, "model": self.model},
            }
