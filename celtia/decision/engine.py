from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence
from .schema import DecisionQuestion, DecisionRequest, DecisionResult
from .policy import decision_policy, probability_distribution, validate_policy_settings

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
        settings = validate_policy_settings(
            abstain_below=self.abstain_below,
            temperature=self.temperature,
            reject_suspected_ood=self.reject_suspected_ood,
            ood_entropy_threshold=self.ood_entropy_threshold,
            ood_margin_threshold=self.ood_margin_threshold,
        )
        return [self._decide_one(request.context, q, settings) for q in request.questions]

    def _decide_one(self, context: object, q: DecisionQuestion, settings: dict) -> DecisionResult:
        candidates = q.candidates()
        logits = list(self.scorer.score(context, q, candidates))
        pairs = probability_distribution(candidates, logits, temperature=settings["temperature"])
        outcome = decision_policy(
            pairs,
            abstain_below=settings["abstain_below"],
            reject_suspected_ood=settings["reject_suspected_ood"],
            ood_entropy_threshold=settings["ood_entropy_threshold"],
            ood_margin_threshold=settings["ood_margin_threshold"],
        )
        expected_score = (
            sum(float(candidate) * probability for candidate, probability in pairs.items())
            if q.type.value == "score"
            else None
        )
        return DecisionResult(q.id, pairs, expected_score=expected_score, **outcome)
