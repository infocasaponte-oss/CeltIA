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
        async def chat(messages: list[dict]) -> str:
            response = await llm.chat(
                messages,
                thinking=False,
                max_tokens=384,
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
            return content.strip()

        self.engine = AsyncDecisionEngine(
            AsyncLLMDecisionScorer(chat),
            abstain_below=abstain_below,
            temperature=temperature,
            reject_suspected_ood=reject_suspected_ood,
            ood_entropy_threshold=ood_entropy_threshold,
            ood_margin_threshold=ood_margin_threshold,
        )

    async def decide(self, context, questions):
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
        request = DecisionRequest(
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
        return await self.engine.decide(request)
