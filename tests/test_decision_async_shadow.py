import asyncio
from types import SimpleNamespace

from apps.api import main as api_main


class FastLLM:
    model = "fake"

    def __init__(self):
        self.finished = False

    async def chat(self, *args, **kwargs):
        await asyncio.sleep(0)
        self.finished = True
        return {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }


class MinimalGateway:
    def limits_for(self, role):
        return {"max_tokens_per_request": 4096}

    def record_tokens(self, *args, **kwargs):
        pass


def test_shadow_is_scheduled_only_after_foreground_generation(monkeypatch):
    llm = FastLLM()
    scheduled = []

    def schedule(**kwargs):
        scheduled.append((llm.finished, kwargs))
        return None

    monkeypatch.setattr(api_main, "llm", llm)
    monkeypatch.setattr(api_main, "gateway", MinimalGateway())
    monkeypatch.setattr(api_main, "_schedule_shadow_background", schedule)
    monkeypatch.setattr(api_main.settings, "decision_routing_mode", "shadow")
    monkeypatch.setattr(api_main.settings, "decision_shadow_routing", False)
    monkeypatch.setattr(api_main.settings, "decision_cde_rollout_percent", 0)

    req = api_main.ChatRequest(
        messages=[api_main.Message(role="user", content="Ola")],
    )
    key = {"id": None, "role": "user", "token_balance": None}

    result = asyncio.run(api_main._build_response(req, "session-test", key))

    assert result["choices"][0]["message"]["content"] == "ok"
    assert scheduled
    assert scheduled[0][0] is True
    assert scheduled[0][1]["heuristic_route"] == "fast"


def test_shadow_backlog_is_dropped_without_creating_task(monkeypatch):
    old_tasks = api_main.app.state.decision_shadow_tasks
    try:
        api_main.app.state.decision_shadow_tasks = {object()}
        monkeypatch.setattr(api_main.settings, "decision_shadow_max_pending", 1)
        before = api_main.app.state.metrics["decision_shadow_background_dropped"]

        result = api_main._schedule_shadow_background(
            api_key_id=None,
            text="x",
            heuristic_route="fast",
            bucket_key="session:x",
            input_chars=1,
            long_context_chars=12000,
            rollout_percent=0,
        )

        assert result is None
        assert api_main.app.state.metrics["decision_shadow_background_dropped"] == before + 1
    finally:
        api_main.app.state.decision_shadow_tasks = old_tasks


def test_background_shadow_persists_success(monkeypatch):
    persisted = []

    async def fake_select(*args, **kwargs):
        return {
            "heuristic": "fast",
            "cde": "think",
            "served_route": "fast",
            "routing_source": "shadow",
            "fallback_reason": None,
            "rollout_bucket": 3,
            "cde_latency_ms": 10,
            "evaluated_cde": True,
            "confidence": .9,
            "abstained": False,
        }

    monkeypatch.setattr(api_main, "select_serving_route", fake_select)
    monkeypatch.setattr(
        api_main,
        "_persist_routing_decision",
        lambda api_key_id, routing, rollout_percent: persisted.append(
            (api_key_id, routing, rollout_percent)
        ),
    )

    asyncio.run(api_main._run_shadow_background(
        api_key_id=7,
        text="x",
        heuristic_route="fast",
        bucket_key="api:7",
        input_chars=1,
        long_context_chars=12000,
        rollout_percent=0,
    ))

    assert persisted
    assert persisted[0][0] == 7
    assert persisted[0][1]["cde"] == "think"
