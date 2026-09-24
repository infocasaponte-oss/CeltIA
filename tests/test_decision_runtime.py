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
        self.calls = []

    async def chat(self, messages, **kwargs):
        self.kwargs = kwargs
        self.calls.append(kwargs)
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


def test_runtime_supports_lower_configured_question_limit():
    runtime = CeltIADecisionRuntime(FakeLLM(), max_questions=1)
    questions = [
        {"id":"q1","prompt":"x","type":"boolean"},
        {"id":"q2","prompt":"x","type":"boolean"},
    ]
    try:
        asyncio.run(runtime.decide({}, questions))
        assert False
    except ValueError as exc:
        assert "between 1 and 1" in str(exc)


def test_runtime_supports_configurable_output_token_cap():
    llm = CaptureBudgetLLM()
    runtime = CeltIADecisionRuntime(
        llm,
        abstain_below=0.0,
        reject_suspected_ood=False,
        max_output_tokens=256,
    )
    asyncio.run(runtime.decide({}, [{
        "id":"route","prompt":"route","type":"choice","options":["fast","think"]
    }]))
    assert llm.kwargs["max_tokens"] == 256


def test_runtime_rejects_invalid_cost_bounds():
    for kwargs in (
        {"max_questions": 0},
        {"max_questions": 33},
        {"max_output_tokens": 63},
        {"max_output_tokens": 2049},
    ):
        try:
            CeltIADecisionRuntime(FakeLLM(), **kwargs)
            assert False
        except ValueError:
            pass


def test_runtime_splits_total_output_budget_across_questions():
    llm = CaptureBudgetLLM()
    runtime = CeltIADecisionRuntime(
        llm,
        abstain_below=0.0,
        reject_suspected_ood=False,
        max_output_tokens=1024,
        max_total_output_tokens=512,
        max_questions=4,
    )
    questions = [
        {"id":f"q{i}","prompt":"route","type":"choice","options":["fast","think"]}
        for i in range(4)
    ]
    asyncio.run(runtime.decide({}, questions))
    assert len(llm.calls) == 4
    assert all(call["max_tokens"] == 128 for call in llm.calls)


def test_runtime_rejects_total_budget_too_small_for_question_cap():
    try:
        CeltIADecisionRuntime(
            FakeLLM(),
            max_questions=4,
            max_total_output_tokens=255,
        )
        assert False
    except ValueError as exc:
        assert "too small" in str(exc)


def test_runtime_rejects_invalid_total_output_budget():
    for value in (63, 65537):
        try:
            CeltIADecisionRuntime(FakeLLM(), max_total_output_tokens=value)
            assert False
        except ValueError:
            pass


def test_runtime_rejects_aggregate_prompt_budget_before_model_calls():
    llm = CaptureBudgetLLM()
    runtime = CeltIADecisionRuntime(
        llm,
        abstain_below=0.0,
        reject_suspected_ood=False,
        max_questions=2,
        max_total_prompt_chars=10000,
    )
    questions = [
        {"id":"q1","prompt":"route","type":"choice","options":["fast","think"]},
        {"id":"q2","prompt":"route","type":"choice","options":["fast","think"]},
    ]
    try:
        asyncio.run(runtime.decide({"text":"x" * 6000}, questions))
        assert False
    except ValueError as exc:
        assert "prompt budget exceeded" in str(exc)
    assert llm.calls == []


def test_runtime_accepts_request_within_aggregate_prompt_budget():
    llm = CaptureBudgetLLM()
    runtime = CeltIADecisionRuntime(
        llm,
        abstain_below=0.0,
        reject_suspected_ood=False,
        max_total_prompt_chars=10000,
    )
    asyncio.run(runtime.decide({"text":"short"}, [{
        "id":"route","prompt":"route","type":"choice","options":["fast","think"]
    }]))
    assert len(llm.calls) == 1


def test_runtime_rejects_invalid_total_prompt_budget():
    for value in (9999, 2000001):
        try:
            CeltIADecisionRuntime(FakeLLM(), max_total_prompt_chars=value)
            assert False
        except ValueError:
            pass


class SlowLLM:
    async def chat(self, messages, **kwargs):
        await asyncio.sleep(0.05)
        return {
            "choices":[{"message":{"content":'{"scores":{"c0":0.0,"c1":1.0}}'}}],
            "usage":{"prompt_tokens":1,"completion_tokens":1},
        }


def test_runtime_times_out_slow_scoring_call():
    runtime = CeltIADecisionRuntime(
        SlowLLM(),
        abstain_below=0.0,
        reject_suspected_ood=False,
        call_timeout_seconds=0.01,
    )
    try:
        asyncio.run(runtime.decide({}, [{
            "id":"route","prompt":"route","type":"choice","options":["fast","think"]
        }]))
        assert False
    except RuntimeError as exc:
        assert "timed out" in str(exc)


def test_runtime_rejects_invalid_call_timeout():
    for value in (0, -1, 301):
        try:
            CeltIADecisionRuntime(FakeLLM(), call_timeout_seconds=value)
            assert False
        except ValueError:
            pass


class MultiSlowLLM:
    def __init__(self):
        self.calls = 0

    async def chat(self, messages, **kwargs):
        self.calls += 1
        await asyncio.sleep(0.05)
        return {
            "choices":[{"message":{"content":'{"scores":{"c0":0.0,"c1":1.0}}'}}],
            "usage":{"prompt_tokens":1,"completion_tokens":1},
        }


def test_runtime_times_out_entire_multi_question_request():
    llm = MultiSlowLLM()
    runtime = CeltIADecisionRuntime(
        llm,
        abstain_below=0.0,
        reject_suspected_ood=False,
        call_timeout_seconds=1.0,
        request_timeout_seconds=0.07,
        max_questions=2,
    )
    questions = [
        {"id":"q1","prompt":"route","type":"choice","options":["fast","think"]},
        {"id":"q2","prompt":"route","type":"choice","options":["fast","think"]},
    ]
    try:
        asyncio.run(runtime.decide({}, questions))
        assert False
    except RuntimeError as exc:
        assert "decision request timed out" in str(exc)
    assert llm.calls >= 1


def test_runtime_rejects_invalid_request_timeout():
    for value in (0, -1, 1801):
        try:
            CeltIADecisionRuntime(FakeLLM(), request_timeout_seconds=value)
            assert False
        except ValueError:
            pass


def test_runtime_rejects_invalid_policy_configuration_at_construction():
    invalid = (
        {"abstain_below": -0.01},
        {"abstain_below": 1.01},
        {"abstain_below": float("nan")},
        {"temperature": 0},
        {"temperature": -1},
        {"temperature": float("inf")},
        {"ood_entropy_threshold": -0.01},
        {"ood_entropy_threshold": 1.01},
        {"ood_margin_threshold": -0.01},
        {"ood_margin_threshold": 1.01},
        {"ood_margin_threshold": float("nan")},
        {"reject_suspected_ood": 1},
    )
    for kwargs in invalid:
        try:
            CeltIADecisionRuntime(FakeLLM(), **kwargs)
            assert False, kwargs
        except ValueError:
            pass


def test_runtime_rejects_malformed_question_containers_cleanly():
    runtime = CeltIADecisionRuntime(FakeLLM())
    invalid_questions = (
        123,
        ["not-a-mapping"],
        [{"id":"q","prompt":"x"}],
        [{"id":"q","type":"boolean"}],
        [{"prompt":"x","type":"boolean"}],
    )
    for questions in invalid_questions:
        try:
            asyncio.run(runtime.decide({}, questions))
            assert False, questions
        except ValueError:
            pass


def test_runtime_rejects_pathologically_deep_context_cleanly():
    runtime = CeltIADecisionRuntime(FakeLLM())
    context = []
    cursor = context
    for _ in range(2000):
        child = []
        cursor.append(child)
        cursor = child
    try:
        asyncio.run(runtime.decide(context, [{"id":"x","prompt":"x","type":"boolean"}]))
        assert False
    except ValueError as exc:
        assert "JSON-serializable" in str(exc)
