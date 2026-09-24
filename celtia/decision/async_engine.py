from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Protocol, Sequence
from .schema import DecisionQuestion, DecisionRequest, DecisionResult
from .robustness import ood_signal

class AsyncCandidateScorer(Protocol):
    async def score(self, context: object, question: DecisionQuestion, candidates: Sequence[str]) -> Sequence[float]: ...

@dataclass
class AsyncDecisionEngine:
    scorer: AsyncCandidateScorer
    abstain_below: float = 0.55
    temperature: float = 1.0
    reject_suspected_ood: bool = True
    ood_entropy_threshold: float = 0.90
    ood_margin_threshold: float = 0.10

    async def decide(self, request: DecisionRequest) -> list[DecisionResult]:
        if self.temperature <= 0: raise ValueError("temperature must be positive")
        results=[]
        for q in request.questions:
            candidates=q.candidates(); logits=list(await self.scorer.score(request.context,q,candidates))
            if len(logits)!=len(candidates) or not logits or not all(math.isfinite(float(v)) for v in logits): raise ValueError("scorer returned invalid logits")
            scaled=[float(v)/self.temperature for v in logits]; m=max(scaled); exps=[math.exp(v-m) for v in scaled]; z=sum(exps); probs=[v/z for v in exps]
            best=max(range(len(probs)),key=probs.__getitem__); confidence=probs[best]
            distribution=dict(zip(candidates,probs,strict=True))
            risk=ood_signal(distribution, entropy_threshold=self.ood_entropy_threshold,
                            margin_threshold=self.ood_margin_threshold)
            low_confidence=confidence < self.abstain_below
            rejected_ood=self.reject_suspected_ood and risk["suspected_ood"]
            abstained=low_confidence or rejected_ood
            reason="low_confidence" if low_confidence else ("suspected_ood" if rejected_ood else None)
            results.append(DecisionResult(
                q.id, distribution, None if abstained else candidates[best], confidence, abstained,
                reason, risk["normalized_entropy"], risk["margin"]
            ))
        return results
