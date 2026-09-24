import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

from apps.api import main as api_main


class FakeGateway:
    def __init__(self):
        self.keys = []
        self.tokens = []

    @asynccontextmanager
    async def slot(self, key):
        self.keys.append(key)
        yield

    def record_tokens(self, key_id, tokens, client_key_id=None):
        self.tokens.append((key_id, tokens, client_key_id))


class FakeDecisionRuntime:
    async def decide_with_usage(self, context, questions):
        return [
            SimpleNamespace(
                id=questions[0]["id"],
                probabilities={"false": 0.1, "true": 0.9},
                decision="true",
                confidence=0.9,
                abstained=False,
                abstention_reason=None,
                normalized_entropy=0.2,
                margin=0.8,
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
    assert result["data"][0]["abstention_reason"] is None
    assert result["data"][0]["normalized_entropy"] == 0.2
    assert result["data"][0]["margin"] == 0.8


class FakeMemory:
    def __init__(self):
        self.usage = []
        self.decrements = []

    def record_usage(self, key_id, prompt_tokens, completion_tokens, **kwargs):
        self.usage.append((key_id, prompt_tokens, completion_tokens, kwargs))

    def decrement_token_balance(self, key_id, amount):
        self.decrements.append((key_id, amount))


def test_decide_accounts_usage_for_authenticated_key(monkeypatch):
    gateway = FakeGateway()
    memory = FakeMemory()
    monkeypatch.setattr(api_main, "gateway", gateway)
    monkeypatch.setattr(api_main, "memory", memory)
    monkeypatch.setattr(api_main, "decision_runtime", FakeDecisionRuntime())
    req = api_main.DecisionApiRequest(
        context={},
        questions=[api_main.DecisionQuestionInput(id="q", prompt="safe?", type="boolean")],
    )
    key = {
        "id": 7,
        "role": "user",
        "token_balance": 100,
        "client_key_id": 9,
        "stripe_customer_id": None,
    }
    result = asyncio.run(api_main.decide(req, key))

    assert result["usage"]["total_tokens"] == 6
    assert gateway.tokens == [(7, 6, 9)]
    assert memory.decrements == [(7, 6)]
    assert memory.usage[0][:3] == (7, 4, 2)
    assert memory.usage[0][3]["client_key_id"] == 9
