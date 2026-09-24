from __future__ import annotations
import json
import math
from collections.abc import Awaitable, Callable, Sequence
from .schema import DecisionQuestion

class AsyncLLMDecisionScorer:
    """Model adapter using observable structured output, not hidden reasoning.

    The callable receives messages and must return assistant text. This keeps the
    adapter independent of CeltIA HTTP/vLLM internals and easy to test.
    """
    def __init__(self, chat: Callable[[list[dict]], Awaitable[str]]): self.chat = chat

    async def score(self, context: object, question: DecisionQuestion, candidates: Sequence[str]) -> list[float]:
        prompt = ("Score every candidate for the decision. Return JSON only: "
                  "{\\\"scores\\\":{\\\"candidate\\\": number}}. Do not add candidates. "
                  "Scores are relative logits, not probabilities.\\nContext: " + json.dumps(context, ensure_ascii=False, default=str) +
                  "\\nQuestion: " + question.prompt + "\\nCandidates: " + json.dumps(list(candidates), ensure_ascii=False))
        text = await self.chat([{"role":"system","content":"You are CeltIA Decision Scorer. Output only the requested JSON; no chain-of-thought."},{"role":"user","content":prompt}])
        try:
            data=json.loads(text)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError("model returned invalid JSON") from exc
        if not isinstance(data, dict) or set(data) != {"scores"}:
            raise ValueError("model returned invalid score envelope")
        scores=data["scores"]
        if not isinstance(scores, dict) or set(scores) != set(candidates):
            raise ValueError("model returned invalid candidate scores")
        values=[]
        for candidate in candidates:
            raw=scores[candidate]
            if isinstance(raw, bool):
                raise ValueError("candidate scores must be finite numbers")
            try:
                value=float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError("candidate scores must be finite numbers") from exc
            if not math.isfinite(value):
                raise ValueError("candidate scores must be finite numbers")
            values.append(value)
        return values
