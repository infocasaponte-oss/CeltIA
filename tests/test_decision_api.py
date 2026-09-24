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
                expected_score=None,
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
    assert result["data"][0]["expected_score"] is None


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


def test_decision_api_schema_rejects_invalid_type_and_bounds():
    invalid_payloads = [
        {"context": {}, "questions": []},
        {"context": {}, "questions": [{"id":"","prompt":"x","type":"boolean"}]},
        {"context": {}, "questions": [{"id":"q","prompt":"","type":"boolean"}]},
        {"context": {}, "questions": [{"id":"q","prompt":"x","type":"unknown"}]},
        {"context": {}, "questions": [{
            "id":"q","prompt":"x","type":"choice","options":["x" * 1001, "b"]
        }]},
        {"context": {}, "questions": [
            {"id":f"q{i}","prompt":"x","type":"boolean"} for i in range(33)
        ]},
    ]
    for payload in invalid_payloads:
        try:
            api_main.DecisionApiRequest.model_validate(payload)
            assert False
        except ValueError:
            pass


def test_decision_api_schema_accepts_maximum_supported_sizes():
    req = api_main.DecisionApiRequest.model_validate({
        "context": {},
        "questions": [{
            "id":"i" * 128,
            "prompt":"p" * 8000,
            "type":"choice",
            "options":[str(i) for i in range(64)],
        }],
    })
    assert len(req.questions[0].options) == 64


def test_decide_honors_configured_question_limit(monkeypatch):
    gateway = FakeGateway()
    monkeypatch.setattr(api_main, "gateway", gateway)
    monkeypatch.setattr(api_main.settings, "decision_max_questions", 1)
    req = api_main.DecisionApiRequest(
        context={},
        questions=[
            api_main.DecisionQuestionInput(id="q1", prompt="safe?", type="boolean"),
            api_main.DecisionQuestionInput(id="q2", prompt="safe?", type="boolean"),
        ],
    )
    key = {"id": None, "role": "user"}
    try:
        asyncio.run(api_main.decide(req, key))
        assert False
    except api_main.HTTPException as exc:
        assert exc.status_code == 400
        assert "between 1 and 1" in str(exc.detail)
    assert gateway.keys == []


def test_decision_api_schema_rejects_invalid_choice_and_score_semantics():
    invalid_questions = [
        {"id":"c","prompt":"choice?","type":"choice","options":[]},
        {"id":"c","prompt":"choice?","type":"choice","options":["x"]},
        {"id":"c","prompt":"choice?","type":"choice","options":["x","x"]},
        {"id":"s","prompt":"score?","type":"score"},
        {"id":"s","prompt":"score?","type":"score","minimum":5,"maximum":4},
        {"id":"s","prompt":"score?","type":"score","minimum":0,"maximum":101},
    ]
    for question in invalid_questions:
        try:
            api_main.DecisionApiRequest.model_validate({"context": {}, "questions": [question]})
            assert False
        except ValueError:
            pass


def test_decision_api_schema_accepts_valid_score_range():
    req = api_main.DecisionApiRequest.model_validate({
        "context": {},
        "questions": [{"id":"s","prompt":"score?","type":"score","minimum":0,"maximum":100}],
    })
    assert req.questions[0].minimum == 0
    assert req.questions[0].maximum == 100


def test_decision_api_schema_rejects_cross_type_fields():
    invalid_questions = [
        {"id":"b","prompt":"bool?","type":"boolean","options":["x","y"]},
        {"id":"b","prompt":"bool?","type":"boolean","minimum":0,"maximum":1},
        {"id":"c","prompt":"choice?","type":"choice","options":["x","y"],"minimum":0},
        {"id":"s","prompt":"score?","type":"score","options":["x","y"],"minimum":1,"maximum":5},
    ]
    for question in invalid_questions:
        try:
            api_main.DecisionApiRequest.model_validate({"context": {}, "questions": [question]})
            assert False
        except ValueError:
            pass


class TimeoutDecisionRuntime:
    async def decide_with_usage(self, context, questions):
        raise RuntimeError("decision backend timed out")


def test_decide_maps_backend_timeout_to_503(monkeypatch):
    gateway = FakeGateway()
    monkeypatch.setattr(api_main, "gateway", gateway)
    monkeypatch.setattr(api_main, "decision_runtime", TimeoutDecisionRuntime())
    req = api_main.DecisionApiRequest(
        context={},
        questions=[api_main.DecisionQuestionInput(id="q", prompt="safe?", type="boolean")],
    )
    key = {"id": None, "role": "user"}
    try:
        asyncio.run(api_main.decide(req, key))
        assert False
    except api_main.HTTPException as exc:
        assert exc.status_code == 503
        assert "timed out" in str(exc.detail)
    assert gateway.keys == [key]


class RequestTimeoutDecisionRuntime:
    async def decide_with_usage(self, context, questions):
        raise RuntimeError("decision request timed out")


def test_decide_maps_request_timeout_to_503(monkeypatch):
    gateway = FakeGateway()
    monkeypatch.setattr(api_main, "gateway", gateway)
    monkeypatch.setattr(api_main, "decision_runtime", RequestTimeoutDecisionRuntime())
    req = api_main.DecisionApiRequest(
        context={},
        questions=[api_main.DecisionQuestionInput(id="q", prompt="safe?", type="boolean")],
    )
    key = {"id": None, "role": "user"}
    try:
        asyncio.run(api_main.decide(req, key))
        assert False
    except api_main.HTTPException as exc:
        assert exc.status_code == 503
        assert "request timed out" in str(exc.detail)
    assert gateway.keys == [key]
