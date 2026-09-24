import asyncio
from types import SimpleNamespace

from core.decision_shadow import evaluate_route_shadow


class FakeRuntime:
    def __init__(self):
        self.calls=[]

    async def decide_with_usage(self, context, questions):
        self.calls.append((context, questions))
        return [
            SimpleNamespace(
                decision="think",
                confidence=.8,
                abstained=False,
                abstention_reason=None,
                suspected_ood=False,
                normalized_entropy=.3,
                margin=.5,
            )
        ], {"prompt_tokens": 11, "completion_tokens": 2, "total_tokens": 13}


def test_shadow_route_returns_usage_without_changing_heuristic():
    runtime=FakeRuntime()
    result=asyncio.run(evaluate_route_shadow(runtime, "analyze carefully", "fast"))
    assert result == {
        "heuristic": "fast",
        "cde": "think",
        "confidence": .8,
        "abstained": False,
        "abstention_reason": None,
        "suspected_ood": False,
        "normalized_entropy": .3,
        "margin": .5,
        "usage": {"prompt_tokens": 11, "completion_tokens": 2, "total_tokens": 13},
    }
    context,questions=runtime.calls[0]
    assert context["heuristic_route"] == "fast"
    assert questions[0]["options"] == ["fast","think","code","agent","long"]


class FailingRuntime:
    async def decide_with_usage(self, context, questions):
        raise RuntimeError("backend unavailable")


def test_shadow_route_fails_closed_to_none():
    assert asyncio.run(evaluate_route_shadow(FailingRuntime(), "x", "fast")) is None
