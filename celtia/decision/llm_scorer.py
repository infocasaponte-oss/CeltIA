from __future__ import annotations
import json
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
        data=json.loads(text); scores=data.get("scores")
        if not isinstance(scores, dict) or set(scores) != set(candidates): raise ValueError("model returned invalid candidate scores")
        return [float(scores[c]) for c in candidates]
