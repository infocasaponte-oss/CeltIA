# Copyright (c) 2026 Luis Manuel Cousido Hermida. All rights reserved.
import json
import logging
import re
import time
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
    """OpenAI-compatible chat client: local vLLM/Ollama, or a hosted API (xAI Grok) when `api_key` is set."""

    def __init__(self, base_url: str, model: str, api_key: str = "", *, context: int | None = None,
                 reasoning_model: str | None = None, local: bool = True, timeout_floor: int = 60):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.context = context
        self.reasoning_model = reasoning_model
        self.local = local
        self.timeout_floor = timeout_floor

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    async def _chat_streaming(self, payload, timeout, on_token):
        payload = {**payload, "stream": True, "stream_options": {"include_usage": True}}
        content_parts: list[str] = []
        calls: dict[int, dict] = {}
        usage = None
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", f"{self.base_url}/chat/completions", json=payload, headers=self._headers()) as r:
                if r.status_code >= 400:
                    await r.aread()  # keep the error body available for diagnosis
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

    async def raw_chat(self, messages: list[dict[str, Any]], *, thinking: bool, max_tokens: int, tools=None,
                       temperature=None, on_token=None):
        """One completion; raises on any failure (the caller decides how to degrade).
        `on_token(delta)` streams content as it is generated; `on_token(None)` signals that the text streamed so
        far was preamble to a tool call and should be discarded."""
        context = self.context or settings.model_context
        fitted_messages, fitted_max_tokens = _fit_to_context(messages, tools, max_tokens, context)
        # The (slower, pricier) reasoning model is only used for plain answers; tool-calling turns use the fast one.
        model = self.reasoning_model if (thinking and self.reasoning_model and not tools) else self.model
        payload = {
            "model": model,
            "messages": fitted_messages,
            "max_tokens": fitted_max_tokens,
            "temperature": temperature if temperature is not None else (1.0 if thinking and self.local else 0.7),
            "top_p": 0.95,
        }
        if self.local:
            payload["chat_template_kwargs"] = {"enable_thinking": thinking}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        timeout = max(self.timeout_floor, fitted_max_tokens * 1.0) if self.local else self.timeout_floor
        if on_token is not None:
            out = await self._chat_streaming(payload, timeout, on_token)
        else:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(f"{self.base_url}/chat/completions", json=payload, headers=self._headers())
                r.raise_for_status()
                out = r.json()
        out.setdefault("meta", {})["model"] = model
        return out

    async def chat(self, messages: list[dict[str, Any]], *, thinking: bool,
                   max_tokens: int, tools=None, temperature=None, on_token=None):
        try:
            return await self.raw_chat(messages, thinking=thinking, max_tokens=max_tokens, tools=tools,
                                       temperature=temperature, on_token=on_token)
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


class FailoverClient:
    """Primary (cloud) model with automatic fallback to the local model.

    Availability failures (auth, credit, rate limit, 5xx, timeouts, network) put the primary on a short cooldown so
    requests go straight to the local model instead of waiting on a dead service; request-specific 4xx errors only
    fall back for that one request."""

    AVAILABILITY_STATUS = {401, 402, 403, 404, 408, 429}
    ACCOUNT_ERROR_HINT = re.compile(r"api key|credit|billing|quota|permission|authenticat|unauthori|rate limit|suspended|blocked", re.I)

    def __init__(self, primary: VLLMClient, fallback: VLLMClient, cooldown_seconds: int = 60):
        self.primary = primary
        self.fallback = fallback
        self.cooldown_seconds = cooldown_seconds
        self._down_until = 0.0

    @property
    def model(self) -> str:
        return self.primary.model

    def primary_available(self) -> bool:
        return time.monotonic() >= self._down_until

    def _is_availability_error(self, exc: Exception) -> bool:
        if isinstance(exc, httpx.HTTPStatusError):
            code = exc.response.status_code
            if code >= 500 or code in self.AVAILABILITY_STATUS:
                return True
            # xAI answers 400 (not 401) for an invalid key, and 4xx for exhausted credit or blocked teams.
            try:
                body = exc.response.text
            except Exception:
                body = ""
            return bool(self.ACCOUNT_ERROR_HINT.search(body))
        return True  # timeouts, connection errors, ...

    async def chat(self, messages, *, thinking: bool, max_tokens: int, tools=None, temperature=None, on_token=None):
        if self.primary_available():
            streamed = {"any": False}
            wrapped = None
            if on_token is not None:
                def wrapped(delta):
                    if delta is not None:
                        streamed["any"] = True
                    on_token(delta)
            try:
                return await self.primary.raw_chat(messages, thinking=thinking, max_tokens=max_tokens, tools=tools,
                                                   temperature=temperature, on_token=wrapped)
            except Exception as exc:
                logger.warning("primary LLM failed (%s: %s); using local fallback", type(exc).__name__, exc)
                diagnostics.record_error("llm_primary", f"{type(exc).__name__}: {exc}")
                if self._is_availability_error(exc):
                    self._down_until = time.monotonic() + self.cooldown_seconds
                if streamed["any"] and on_token is not None:
                    on_token(None)  # discard the partial text already shown
        return await self.fallback.chat(messages, thinking=thinking, max_tokens=max_tokens, tools=tools,
                                        temperature=temperature, on_token=on_token)


def build_llm():
    """Grok (xAI) as primary chat model when a key is configured, with the local model as fallback."""
    local = VLLMClient(settings.vllm_base_url, settings.model_serve_name)
    key = settings.primary_llm_key
    if not (settings.llm_primary_enabled and key):
        return local
    primary = VLLMClient(
        settings.llm_primary_base_url, settings.llm_primary_model, api_key=key, context=settings.llm_primary_context,
        reasoning_model=settings.llm_primary_reasoning_model or None, local=False,
        timeout_floor=settings.llm_primary_timeout_seconds,
    )
    return FailoverClient(primary, local, cooldown_seconds=settings.llm_primary_cooldown_seconds)
