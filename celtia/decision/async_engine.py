from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Protocol, Sequence
from .schema import DecisionQuestion, DecisionRequest, DecisionResult

class AsyncCandidateScorer(Protocol):
    async def score(self, context: object, question: DecisionQuestion, candidates: Sequence[str]) -> Sequence[float]: ...

@dataclass
class AsyncDecisionEngine:
    scorer: AsyncCandidateScorer
    abstain_below: float = 0.55
    temperature: float = 1.0

    async def decide(self, request: DecisionRequest) -> list[DecisionResult]:
        if self.temperature <= 0: raise ValueError("temperature must be positive")
        results=[]
        for q in request.questions:
            candidates=q.candidates(); logits=list(await self.scorer.score(request.context,q,candidates))
            if len(logits)!=len(candidates) or not logits or not all(math.isfinite(float(v)) for v in logits): raise ValueError("scorer returned invalid logits")
            scaled=[float(v)/self.temperature for v in logits]; m=max(scaled); exps=[math.exp(v-m) for v in scaled]; z=sum(exps); probs=[v/z for v in exps]
            best=max(range(len(probs)),key=probs.__getitem__); confidence=probs[best]; abstained=confidence < self.abstain_below
            results.append(DecisionResult(q.id,dict(zip(candidates,probs,strict=True)),None if abstained else candidates[best],confidence,abstained))
        return results
