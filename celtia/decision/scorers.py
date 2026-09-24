from __future__ import annotations
from collections.abc import Callable, Sequence
from .schema import DecisionQuestion

class CallableScorer:
    def __init__(self, fn: Callable[[object, DecisionQuestion, Sequence[str]], Sequence[float]]): self.fn = fn
    def score(self, context: object, question: DecisionQuestion, candidates: Sequence[str]) -> Sequence[float]: return self.fn(context, question, candidates)

class UniformScorer:
    def score(self, context: object, question: DecisionQuestion, candidates: Sequence[str]) -> Sequence[float]: return [0.0] * len(candidates)
