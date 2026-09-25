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
            ),
            SimpleNamespace(
                probabilities={"false":.9,"true":.1},
            ),
        ], {"prompt_tokens": 22, "completion_tokens": 4, "total_tokens": 26}


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
        "ood_probability": .1,
        "ood_classifier_confidence": .9,
        "ood_classifier_decision": False,
        "usage": {"prompt_tokens": 22, "completion_tokens": 4, "total_tokens": 26},
    }
    context,questions=runtime.calls[0]
    assert "heuristic_route" not in context
    assert context["user_message"] == "analyze carefully"
    assert context["input_chars"] == len("analyze carefully")
    assert context["long_context_chars"] > context["input_chars"]
    assert questions[0]["options"] == ["fast","think","code","agent","long"]
    assert questions[1] == {
        "id":"route_ood",
        "prompt":"Is this input outside the CeltIA routing domain?",
        "type":"boolean",
    }


class FailingRuntime:
    async def decide_with_usage(self, context, questions):
        raise RuntimeError("backend unavailable")


def test_shadow_route_fails_closed_to_none():
    assert asyncio.run(evaluate_route_shadow(FailingRuntime(), "x", "fast")) is None
