import asyncio

from core.memory import Memory
from core.planner import Planner
from core.router import route
from core.tools import Registry, Tool, calculator
from core.verification import Verifier


class FakeLLM:
    def __init__(self):
        self.calls = 0

    async def chat(self, messages, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "Plan: use the calculator to evaluate 2 + 3 * 4, then verify the result before answering.",
                        "tool_calls": [{
                            "id": "call-1",
                            "function": {
                                "name": "calculator",
                                "arguments": '{"expression":"2+3*4"}'
                            }
                        }]
                    }
                }]
            }
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "The verified result is 14."
                }
            }]
        }


def test_fast(): assert route("hello").mode == "fast"

def test_code(): assert route("debug this Python exception").mode == "code"

def test_long():
    long_text = (
        "Please read the entire architecture document and summarize the key decisions, constraints, "
        "and trade-offs for the team. "
    ) * 200
    assert route(long_text).mode == "long"


def test_calc(): assert calculator("2+3*4")["result"] == 14


def test_verifier_code():
    verifier = Verifier()
    assert verifier.verify_code("print(2 + 3)")["ok"] is True
    assert verifier.verify_code("if True:\n    print(1)")["ok"] is True
    assert verifier.verify_code("def broken(:\n    pass")["ok"] is False


def test_memory_search_and_procedures(tmp_path):
    mem = Memory(str(tmp_path / "memory.db"))
    mem.add("s1", "user", "Use the calculator to validate arithmetic before answering.")
    mem.add("s1", "assistant", "I will use the calculator and then answer.")
    mem.record_procedure("calculator", "Use the calculator for arithmetic validation and verification.")
    assert mem.search("s1", "calculator")
    assert mem.get_procedures("calculator")


def test_semantic_memory_search(tmp_path):
    mem = Memory(str(tmp_path / "memory.db"))
    mem.add_document("docs", "Mini-Council is a local-first coding and reasoning assistant.")
    mem.add_document("docs", "The calculator tool validates arithmetic expressions safely.")
    hits = mem.semantic_search("math verification with tool use", limit=2)
    assert hits
    assert any("calculator" in hit["text"].lower() or "arithmetic" in hit["text"].lower() for hit in hits)


def test_planner_and_verification_loop():
    planner = Planner()
    plan = planner.plan("Compute 2 + 3 * 4 and verify before answering.")
    assert plan["requires_tool"] is True
    assert plan["verification_required"] is True

    registry = Registry()
    registry.add(Tool("calculator", "Evaluate arithmetic safely.", {
        "type": "object",
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"],
        "additionalProperties": False
    }, calculator))

    async def run_agent():
        from core.agent import Agent
        agent = Agent(FakeLLM(), registry, planner=planner)
        result = await agent.run([{"role": "user", "content": "Compute 2 + 3 * 4 and verify before answering."}])
        assert result["answer"] == "The verified result is 14."
        assert result["verification"]["ok"] is True
        assert result["steps"] >= 1

    asyncio.run(run_agent())


def test_agent_without_planner_uses_default_plan():
    registry = Registry()
    registry.add(Tool("calculator", "Evaluate arithmetic safely.", {
        "type": "object",
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"],
        "additionalProperties": False
    }, calculator))

    async def run_agent():
        from core.agent import Agent
        agent = Agent(FakeLLM(), registry)
        result = await agent.run([{"role": "user", "content": "Compute 2 + 3 * 4 and verify before answering."}])
        assert result["answer"] == "The verified result is 14."
        assert result["verification"]["ok"] is True
        assert result["steps"] >= 1

    asyncio.run(run_agent())


def test_offline_fallback_response():
    from apps.api import main as api_main

    async def run_case():
        req = api_main.ChatRequest(
            model="mini-council",
            messages=[{"role": "user", "content": "hola"}],
            session_id="offline-demo",
            mode="fast",
        )
        result = await api_main._build_response(req, "offline-demo", {"id": None, "role": "user"})
        assert result["choices"][0]["message"]["content"]
        assert result["metadata"]["route"] == "fast"
        assert result["session_id"] == "offline-demo"

    asyncio.run(run_case())


def test_failover_uses_local_when_primary_fails_and_cools_down():
    import asyncio
    import httpx
    from core.inference import FailoverClient, VLLMClient

    calls = {"primary": 0, "fallback": 0}

    class Primary(VLLMClient):
        async def raw_chat(self, messages, **kw):
            calls["primary"] += 1
            request = httpx.Request("POST", "https://x")
            raise httpx.HTTPStatusError("boom", request=request, response=httpx.Response(503, request=request))

    class Fallback(VLLMClient):
        async def chat(self, messages, **kw):
            calls["fallback"] += 1
            return {"choices": [{"message": {"role": "assistant", "content": "local"}}]}

    client = FailoverClient(Primary("https://x", "grok"), Fallback("http://l", "local"), cooldown_seconds=60)

    async def run():
        first = await client.chat([{"role": "user", "content": "hola"}], thinking=False, max_tokens=50)
        second = await client.chat([{"role": "user", "content": "hola"}], thinking=False, max_tokens=50)
        return first, second

    first, second = asyncio.run(run())
    assert first["choices"][0]["message"]["content"] == "local" == second["choices"][0]["message"]["content"]
    assert calls == {"primary": 1, "fallback": 2}  # the second request skipped the failing primary (cooldown)
    assert not client.primary_available()


def test_failover_discards_partial_stream_when_primary_dies_midway():
    import asyncio
    from core.inference import FailoverClient, VLLMClient

    events = []

    class Primary(VLLMClient):
        async def raw_chat(self, messages, on_token=None, **kw):
            on_token("parcial")
            raise RuntimeError("connection reset")

    class Fallback(VLLMClient):
        async def chat(self, messages, on_token=None, **kw):
            on_token("local ok")
            return {"choices": [{"message": {"role": "assistant", "content": "local ok"}}]}

    client = FailoverClient(Primary("https://x", "grok"), Fallback("http://l", "local"))
    asyncio.run(client.chat([{"role": "user", "content": "x"}], thinking=False, max_tokens=50, on_token=events.append))
    assert events == ["parcial", None, "local ok"]  # None tells the UI to drop the partial text


def test_primary_uses_reasoning_model_only_without_tools():
    from core.inference import VLLMClient
    import asyncio

    seen = []

    class Recorder(VLLMClient):
        async def _chat_streaming(self, payload, timeout, on_token):
            seen.append(payload["model"]); return {"choices": [{"message": {"content": "x"}}]}

    c = Recorder("https://x", "fast", api_key="k", reasoning_model="deep", local=False)
    noop = lambda d: None
    asyncio.run(c.raw_chat([{"role": "user", "content": "a"}], thinking=True, max_tokens=50, on_token=noop))
    asyncio.run(c.raw_chat([{"role": "user", "content": "a"}], thinking=True, max_tokens=50, tools=[{"type": "function"}], on_token=noop))
    asyncio.run(c.raw_chat([{"role": "user", "content": "a"}], thinking=False, max_tokens=50, on_token=noop))
    assert seen == ["deep", "fast", "fast"]
