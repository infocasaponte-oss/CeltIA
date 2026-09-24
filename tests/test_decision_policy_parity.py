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


def test_policy_rejects_boolean_logits():
    q = DecisionQuestion("safe", "safe?", DecisionType.BOOLEAN)
    req = DecisionRequest({}, (q,))
    try:
        DecisionEngine(SyncScorer([False, True]), reject_suspected_ood=False).decide(req)
        assert False
    except ValueError as exc:
        assert "invalid logits" in str(exc)


def test_policy_rejects_out_of_range_ood_thresholds():
    q = DecisionQuestion("safe", "safe?", DecisionType.BOOLEAN)
    req = DecisionRequest({}, (q,))
    for kwargs in (
        {"ood_entropy_threshold": -0.1},
        {"ood_entropy_threshold": 1.1},
        {"ood_margin_threshold": -0.1},
        {"ood_margin_threshold": 1.1},
    ):
        try:
            DecisionEngine(
                SyncScorer([0.0, 1.0]),
                reject_suspected_ood=True,
                **kwargs,
            ).decide(req)
            assert False
        except ValueError:
            pass


def test_sync_and_async_engines_reject_invalid_policy_types_consistently():
    q = DecisionQuestion("safe", "safe?", DecisionType.BOOLEAN)
    req = DecisionRequest({}, (q,))
    invalid = (
        {"temperature": True},
        {"temperature": "not-a-number"},
        {"abstain_below": False},
        {"abstain_below": float("nan")},
        {"reject_suspected_ood": 1},
        {"ood_entropy_threshold": float("inf")},
        {"ood_margin_threshold": "bad"},
    )
    for kwargs in invalid:
        try:
            DecisionEngine(SyncScorer([0.0, 1.0]), **kwargs).decide(req)
            assert False, kwargs
        except ValueError:
            pass
        try:
            asyncio.run(AsyncDecisionEngine(AsyncScorer([0.0, 1.0]), **kwargs).decide(req))
            assert False, kwargs
        except ValueError:
            pass


def test_engine_policy_validation_runs_even_for_empty_request():
    req = DecisionRequest({}, ())
    try:
        DecisionEngine(SyncScorer([]), temperature=True).decide(req)
        assert False
    except ValueError:
        pass
    try:
        asyncio.run(AsyncDecisionEngine(AsyncScorer([]), temperature=True).decide(req))
        assert False
    except ValueError:
        pass
