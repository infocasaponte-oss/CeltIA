import asyncio

from celtia.decision.async_engine import AsyncDecisionEngine
from celtia.decision.engine import DecisionEngine
from celtia.decision.schema import DecisionQuestion, DecisionRequest, DecisionType


class SyncScorer:
    def __init__(self, logits):
        self.logits = logits

    def score(self, context, question, candidates):
        return self.logits


class AsyncScorer:
    def __init__(self, logits):
        self.logits = logits

    async def score(self, context, question, candidates):
        return self.logits


def test_sync_and_async_engines_have_identical_policy_semantics():
    q = DecisionQuestion("route", "choose", DecisionType.CHOICE, ("fast", "think", "agent"))
    req = DecisionRequest({}, (q,))
    kwargs = {
        "abstain_below": 0.4,
        "temperature": 0.8,
        "reject_suspected_ood": True,
        "ood_entropy_threshold": 0.95,
        "ood_margin_threshold": 0.05,
    }
    sync = DecisionEngine(SyncScorer([0.0, 2.0, 0.5]), **kwargs).decide(req)[0]
    async_result = asyncio.run(AsyncDecisionEngine(AsyncScorer([0.0, 2.0, 0.5]), **kwargs).decide(req))[0]
    assert sync == async_result


def test_sync_and_async_score_expected_value_match():
    q = DecisionQuestion("score", "rate", DecisionType.SCORE, minimum=1, maximum=3)
    req = DecisionRequest({}, (q,))
    kwargs = {"abstain_below": 0.0, "reject_suspected_ood": False}
    sync = DecisionEngine(SyncScorer([0.0, 1.0, 2.0]), **kwargs).decide(req)[0]
    async_result = asyncio.run(
        AsyncDecisionEngine(AsyncScorer([0.0, 1.0, 2.0]), **kwargs).decide(req)
    )[0]
    assert sync == async_result
    assert sync.expected_score is not None
