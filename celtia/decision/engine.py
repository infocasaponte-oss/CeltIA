from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence
from .schema import DecisionQuestion, DecisionRequest, DecisionResult

class CandidateScorer(Protocol):
    def score(self, context: object, question: DecisionQuestion, candidates: Sequence[str]) -> Sequence[float]: ...

@dataclass
class DecisionEngine:
    scorer: CandidateScorer
    abstain_below: float = 0.55
    temperature: float = 1.0

    def decide(self, request: DecisionRequest) -> list[DecisionResult]:
        if self.temperature <= 0: raise ValueError("temperature must be positive")
        return [self._decide_one(request.context, q) for q in request.questions]

    def _decide_one(self, context: object, q: DecisionQuestion) -> DecisionResult:
        candidates = q.candidates()
        logits = list(self.scorer.score(context, q, candidates))
        if len(logits) != len(candidates) or not logits: raise ValueError("scorer returned an invalid number of logits")
        probs = self._softmax([float(x) / self.temperature for x in logits])
        pairs = dict(zip(candidates, probs, strict=True))
        best = max(range(len(probs)), key=probs.__getitem__)
        confidence = probs[best]
        abstained = confidence < self.abstain_below
        return DecisionResult(q.id, pairs, None if abstained else candidates[best], confidence, abstained)

    @staticmethod
    def _softmax(values: Sequence[float]) -> list[float]:
        if not all(math.isfinite(v) for v in values): raise ValueError("logits must be finite")
        m = max(values); exps = [math.exp(v - m) for v in values]; z = sum(exps)
        return [v / z for v in exps]
