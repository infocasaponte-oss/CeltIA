import asyncio
from core.decision_runtime import CeltIADecisionRuntime

class FakeLLM:
    async def chat(self, messages, **kwargs):
        return {"choices":[{"message":{"content":'{"scores":{"fast":0.1,"think":2.0}}'}}]}

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
