import asyncio
from core.decision_runtime import CeltIADecisionRuntime

class FakeLLM:
    async def chat(self, messages, **kwargs):
        return {
            "choices":[{"message":{"content":'{"scores":{"c0":0.1,"c1":2.0}}'}}],
            "usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18},
        }

def test_runtime_bridge():
    runtime=CeltIADecisionRuntime(FakeLLM(),abstain_below=0.5)
    out=asyncio.run(runtime.decide({"message":"complex task"},[{
        "id":"route","prompt":"Choose route","type":"choice","options":["fast","think"]
    }]))
    assert out[0].decision == "think"
    assert out[0].confidence > 0.5

class OfflineLLM:
    async def chat(self, messages, **kwargs):
        return {"meta":{"offline_fallback":True},"choices":[{"message":{"content":"demo"}}]}

def test_runtime_fails_closed_offline():
    runtime=CeltIADecisionRuntime(OfflineLLM())
    try:
        asyncio.run(runtime.decide({},[{"id":"x","prompt":"x","type":"boolean"}]))
        assert False
    except RuntimeError:
        pass


def test_runtime_rejects_oversized_context():
    runtime = CeltIADecisionRuntime(FakeLLM())
    try:
        asyncio.run(runtime.decide({"text": "x" * 50001}, [{"id":"x","prompt":"x","type":"boolean"}]))
        assert False
    except ValueError as exc:
        assert "context" in str(exc)


def test_runtime_rejects_non_json_context():
    runtime = CeltIADecisionRuntime(FakeLLM())
    try:
        asyncio.run(runtime.decide({"bad": object()}, [{"id":"x","prompt":"x","type":"boolean"}]))
        assert False
    except ValueError as exc:
        assert "JSON-serializable" in str(exc)

def test_runtime_rejects_duplicate_question_ids():
    runtime = CeltIADecisionRuntime(FakeLLM())
    questions = [
        {"id":"same","prompt":"one","type":"boolean"},
        {"id":"same","prompt":"two","type":"boolean"},
    ]
    try:
        asyncio.run(runtime.decide({}, questions))
        assert False
    except ValueError as exc:
        assert "unique" in str(exc)

def test_runtime_rejects_empty_or_excessive_questions():
    runtime = CeltIADecisionRuntime(FakeLLM())
    for questions in ([], [
        {"id":f"q{i}","prompt":"x","type":"boolean"} for i in range(33)
    ]):
        try:
            asyncio.run(runtime.decide({}, questions))
            assert False
        except ValueError as exc:
            assert "between 1 and 32" in str(exc)


def test_runtime_aggregates_usage_across_questions():
    runtime = CeltIADecisionRuntime(FakeLLM(), abstain_below=0.0, reject_suspected_ood=False)
    questions = [
        {"id":"q1","prompt":"route","type":"choice","options":["fast","think"]},
        {"id":"q2","prompt":"route","type":"choice","options":["fast","think"]},
    ]
    results, usage = asyncio.run(runtime.decide_with_usage({}, questions))
    assert len(results) == 2
    assert usage == {"prompt_tokens":22,"completion_tokens":14,"total_tokens":36}


class MalformedUsageLLM:
    async def chat(self, messages, **kwargs):
        return {
            "choices":[{"message":{"content":'{"scores":{"c0":0.1,"c1":2.0}}'}}],
            "usage":{"prompt_tokens":-5,"completion_tokens":"bad","total_tokens":999999},
        }


def test_runtime_ignores_malformed_or_inconsistent_usage_totals():
    runtime = CeltIADecisionRuntime(MalformedUsageLLM(), abstain_below=0.0, reject_suspected_ood=False)
    _, usage = asyncio.run(runtime.decide_with_usage({}, [{
        "id":"route","prompt":"route","type":"choice","options":["fast","think"]
    }]))
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    assert usage["total_tokens"] != 999999


class CaptureBudgetLLM:
    def __init__(self):
        self.kwargs = None

    async def chat(self, messages, **kwargs):
        self.kwargs = kwargs
        return {
            "choices":[{"message":{"content":'{"scores":{"c0":0.0,"c1":1.0}}'}}],
            "usage":{"prompt_tokens":1,"completion_tokens":1},
        }


def test_runtime_reserves_output_budget_for_indexed_scores():
    llm = CaptureBudgetLLM()
    runtime = CeltIADecisionRuntime(llm, abstain_below=0.0, reject_suspected_ood=False)
    asyncio.run(runtime.decide({}, [{
        "id":"route","prompt":"route","type":"choice","options":["fast","think"]
    }]))
    assert llm.kwargs["max_tokens"] == 1024
