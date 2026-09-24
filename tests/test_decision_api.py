import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

from apps.api import main as api_main


class FakeGateway:
    def __init__(self):
        self.keys = []

    @asynccontextmanager
    async def slot(self, key):
        self.keys.append(key)
        yield


class FakeDecisionRuntime:
    async def decide_with_usage(self, context, questions):
        return [
            SimpleNamespace(
                id=questions[0]["id"],
                probabilities={"false": 0.1, "true": 0.9},
                decision="true",
                confidence=0.9,
                abstained=False,
            )
        ], {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}


def test_decide_uses_gateway_slot_and_records_metric(monkeypatch):
    gateway = FakeGateway()
    monkeypatch.setattr(api_main, "gateway", gateway)
    monkeypatch.setattr(api_main, "decision_runtime", FakeDecisionRuntime())
    before = api_main.app.state.metrics["decision_requests"]
    req = api_main.DecisionApiRequest(
        context={"x": 1},
        questions=[api_main.DecisionQuestionInput(id="q", prompt="safe?", type="boolean")],
    )
    key = {"id": None, "role": "user"}
    result = asyncio.run(api_main.decide(req, key))

    assert gateway.keys == [key]
    assert result["data"][0]["decision"] == "true"
    assert api_main.app.state.metrics["decision_requests"] == before + 1
    assert result["usage"]["total_tokens"] == 6
