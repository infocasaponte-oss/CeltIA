import asyncio
from types import SimpleNamespace

from core.decision_shadow import evaluate_route_shadow


class FakeRuntime:
    def __init__(self):
        self.calls=[]

    async def decide_with_usage(self, context, questions):
        self.calls.append((context, questions))
        q = questions[0]
        if q["id"] == "route":
            return [SimpleNamespace(
                decision="think",
                confidence=.8,
                abstained=False,
                abstention_reason=None,
                suspected_ood=False,
                normalized_entropy=.3,
                margin=.5,
            )], {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15, "models": ["m1"]}
        return [SimpleNamespace(
            probabilities={"false":.9,"true":.1},
        )], {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11, "models": ["m1"]}


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
        "usage": {"prompt_tokens": 22, "completion_tokens": 4, "total_tokens": 26, "models": ["m1"]},
    }
    assert len(runtime.calls) == 2
    contexts = [call[0] for call in runtime.calls]
    assert all("heuristic_route" not in context for context in contexts)
    assert all(context["user_message"] == "analyze carefully" for context in contexts)
    assert all(context["input_chars"] == len("analyze carefully") for context in contexts)
    questions = [call[1][0] for call in runtime.calls]
    by_id = {question["id"]: question for question in questions}
    assert by_id["route"]["options"] == ["fast","think","code","agent","long"]
    assert by_id["route_ood"] == {
        "id":"route_ood",
        "prompt":"Is this input outside the CeltIA routing domain?",
        "type":"boolean",
    }


class FailingRuntime:
    async def decide_with_usage(self, context, questions):
        raise RuntimeError("backend unavailable")


def test_shadow_route_fails_closed_to_none():
    assert asyncio.run(evaluate_route_shadow(FailingRuntime(), "x", "fast")) is None


class ConcurrentRuntime:
    def __init__(self):
        self.active = 0
        self.max_active = 0

    async def decide_with_usage(self, context, questions):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        if questions[0]["id"] == "route":
            return [SimpleNamespace(
                decision="fast",
                confidence=.9,
                abstained=False,
                abstention_reason=None,
                suspected_ood=False,
                normalized_entropy=.1,
                margin=.8,
            )], {"prompt_tokens":1,"completion_tokens":1,"total_tokens":2,"models":[]}
        return [SimpleNamespace(probabilities={"false":.9,"true":.1})], {
            "prompt_tokens":1,"completion_tokens":1,"total_tokens":2,"models":[]
        }


def test_shadow_route_and_ood_are_scored_concurrently():
    runtime = ConcurrentRuntime()
    result = asyncio.run(evaluate_route_shadow(runtime, "hello", "fast"))
    assert result["cde"] == "fast"
    assert runtime.max_active == 2
