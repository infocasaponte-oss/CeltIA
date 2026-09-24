from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence
from .schema import DecisionQuestion, DecisionRequest, DecisionResult
from .policy import decision_policy, probability_distribution

class CandidateScorer(Protocol):
    def score(self, context: object, question: DecisionQuestion, candidates: Sequence[str]) -> Sequence[float]: ...

@dataclass
class DecisionEngine:
    scorer: CandidateScorer
    abstain_below: float = 0.55
    temperature: float = 1.0
    reject_suspected_ood: bool = True
    ood_entropy_threshold: float = 0.90
    ood_margin_threshold: float = 0.10

    def decide(self, request: DecisionRequest) -> list[DecisionResult]:
        if self.temperature <= 0: raise ValueError("temperature must be positive")
        return [self._decide_one(request.context, q) for q in request.questions]

    def _decide_one(self, context: object, q: DecisionQuestion) -> DecisionResult:
        candidates = q.candidates()
        logits = list(self.scorer.score(context, q, candidates))
        pairs = probability_distribution(candidates, logits, temperature=self.temperature)
        outcome = decision_policy(
            pairs,
            abstain_below=self.abstain_below,
            reject_suspected_ood=self.reject_suspected_ood,
            ood_entropy_threshold=self.ood_entropy_threshold,
            ood_margin_threshold=self.ood_margin_threshold,
        )
        return DecisionResult(q.id, pairs, **outcome)
