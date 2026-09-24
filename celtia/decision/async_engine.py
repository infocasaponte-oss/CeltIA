from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Sequence
from .schema import DecisionQuestion, DecisionRequest, DecisionResult
from .policy import decision_policy, probability_distribution

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
            candidates=q.candidates()
            logits=list(await self.scorer.score(request.context,q,candidates))
            distribution=probability_distribution(candidates, logits, temperature=self.temperature)
            outcome=decision_policy(
                distribution,
                abstain_below=self.abstain_below,
                reject_suspected_ood=self.reject_suspected_ood,
                ood_entropy_threshold=self.ood_entropy_threshold,
                ood_margin_threshold=self.ood_margin_threshold,
            )
            expected_score = (
                sum(float(candidate) * probability for candidate, probability in distribution.items())
                if q.type.value == "score"
                else None
            )
            results.append(DecisionResult(q.id, distribution, expected_score=expected_score, **outcome))
        return results
