# Copyright (c) 2026 Luis Manuel Cousido Hermida. All rights reserved.
from __future__ import annotations

import json

from celtia.decision.async_engine import AsyncDecisionEngine
from celtia.decision.llm_scorer import AsyncLLMDecisionScorer
from celtia.decision.schema import DecisionQuestion, DecisionRequest, DecisionType


class CeltIADecisionRuntime:
    MAX_CONTEXT_CHARS = 50000
    MAX_QUESTIONS = 32

    """Bridge between the independent CDE core and CeltIA's existing local LLM client."""

    def __init__(self, llm, *, abstain_below: float = 0.55, temperature: float = 1.0, reject_suspected_ood: bool = True, ood_entropy_threshold: float = 0.90, ood_margin_threshold: float = 0.10):
        self.llm = llm
        self.engine_options = {
            "abstain_below": abstain_below,
            "temperature": temperature,
            "reject_suspected_ood": reject_suspected_ood,
            "ood_entropy_threshold": ood_entropy_threshold,
            "ood_margin_threshold": ood_margin_threshold,
        }

    def _request(self, context, questions) -> DecisionRequest:
        try:
            serialized_context = json.dumps(context, ensure_ascii=False)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("context must be JSON-serializable") from exc
        if len(serialized_context) > self.MAX_CONTEXT_CHARS:
            raise ValueError("decision context exceeds 50000 serialized characters")
        if not questions or len(questions) > self.MAX_QUESTIONS:
            raise ValueError("questions must contain between 1 and 32 items")
        ids = [q.get("id") for q in questions]
        if len(ids) != len(set(ids)):
            raise ValueError("question ids must be unique")
        return DecisionRequest(
            context=context,
            questions=tuple(
                DecisionQuestion(
                    id=q["id"],
                    prompt=q["prompt"],
                    type=DecisionType(q["type"]),
                    options=tuple(q.get("options") or ()),
                    minimum=q.get("minimum"),
                    maximum=q.get("maximum"),
                )
                for q in questions
            ),
        )

    async def decide_with_usage(self, context, questions):
        request = self._request(context, questions)
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        async def chat(messages: list[dict]) -> str:
            response = await self.llm.chat(
                messages,
                thinking=False,
                max_tokens=1024,
                temperature=0.0,
            )
            meta = response.get("meta") or {}
            if meta.get("offline_fallback"):
                raise RuntimeError("decision backend unavailable")
            choices = response.get("choices") or []
            if not choices:
                raise RuntimeError("decision backend returned no choices")
            content = (choices[0].get("message") or {}).get("content")
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError("decision backend returned empty content")

            raw_usage = response.get("usage") or {}

            def token_count(name: str, fallback: int) -> int:
                raw = raw_usage.get(name)
                if raw is None or isinstance(raw, bool):
                    return fallback
                try:
                    value = int(raw)
                except (TypeError, ValueError, OverflowError):
                    return fallback
                return value if value >= 0 else fallback

            prompt_tokens = token_count(
                "prompt_tokens",
                max(1, len(json.dumps(messages, ensure_ascii=False)) // 3),
            )
            completion_tokens = token_count(
                "completion_tokens",
                max(1, len(content) // 3),
            )
            total_tokens = prompt_tokens + completion_tokens
            usage["prompt_tokens"] += prompt_tokens
            usage["completion_tokens"] += completion_tokens
            usage["total_tokens"] += total_tokens
            return content.strip()

        engine = AsyncDecisionEngine(AsyncLLMDecisionScorer(chat), **self.engine_options)
        results = await engine.decide(request)
        return results, usage

    async def decide(self, context, questions):
        results, _ = await self.decide_with_usage(context, questions)
        return results
