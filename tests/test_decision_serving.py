import asyncio
from types import SimpleNamespace

from core.decision_serving import effective_routing_mode, rollout_bucket, select_serving_route


class Runtime:
    def __init__(self, *, route="agent", abstained=False, suspected_ood=False):
        self.route=route
        self.abstained=abstained
        self.suspected_ood=suspected_ood
        self.contexts=[]

    async def decide_with_usage(self, context, questions):
        self.contexts.append(context)
        return [
            SimpleNamespace(
                decision=self.route,
                confidence=.95,
                abstained=self.abstained,
                abstention_reason="low confidence" if self.abstained else None,
                suspected_ood=self.suspected_ood,
                normalized_entropy=.1,
                margin=.8,
            ),
            SimpleNamespace(probabilities={"false":.95,"true":.05}),
        ], {"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}


class FailingRuntime:
    async def decide_with_usage(self, context, questions):
        raise RuntimeError("provider down")


def test_effective_mode_keeps_legacy_shadow_compatibility():
    assert effective_routing_mode("legacy",False)=="legacy"
    assert effective_routing_mode("legacy",True)=="shadow"
    assert effective_routing_mode("canary",True)=="canary"


def test_rollout_bucket_is_stable_and_bounded():
    assert rollout_bucket("same-user")==rollout_bucket("same-user")
    assert 0 <= rollout_bucket("same-user") < 100


def test_legacy_does_not_call_cde():
    runtime=Runtime()
    out=asyncio.run(select_serving_route(
        runtime,"hello","fast",mode="legacy",rollout_percent=100,bucket_key="u1"
    ))
    assert out["served_route"]=="fast"
    assert out["routing_source"]=="legacy"
    assert runtime.contexts==[]


def test_shadow_observes_but_keeps_heuristic():
    runtime=Runtime(route="think")
    out=asyncio.run(select_serving_route(
        runtime,"compare these","fast",mode="shadow",rollout_percent=100,bucket_key="u1"
    ))
    assert out["cde"]=="think"
    assert out["served_route"]=="fast"
    assert out["routing_source"]=="shadow"


def test_cde_mode_serves_cde_route():
    runtime=Runtime(route="agent")
    out=asyncio.run(select_serving_route(
        runtime,"look this up","fast",mode="cde",rollout_percent=0,bucket_key="u1"
    ))
    assert out["served_route"]=="agent"
    assert out["routing_source"]=="cde"


def test_cde_abstention_falls_back_to_heuristic():
    runtime=Runtime(route="think",abstained=True)
    out=asyncio.run(select_serving_route(
        runtime,"hard question","fast",mode="cde",rollout_percent=100,bucket_key="u1"
    ))
    assert out["served_route"]=="fast"
    assert out["routing_source"]=="legacy_fallback"
    assert out["fallback_reason"]=="cde_abstained"


def test_cde_ood_signal_falls_back_without_blocking():
    runtime=Runtime(route="agent",suspected_ood=True)
    out=asyncio.run(select_serving_route(
        runtime,"meta input","fast",mode="cde",rollout_percent=100,bucket_key="u1"
    ))
    assert out["served_route"]=="fast"
    assert out["fallback_reason"]=="cde_suspected_ood"


def test_cde_error_falls_back_to_heuristic():
    out=asyncio.run(select_serving_route(
        FailingRuntime(),"hello","fast",mode="cde",rollout_percent=100,bucket_key="u1"
    ))
    assert out["served_route"]=="fast"
    assert out["routing_source"]=="legacy_fallback"
    assert out["fallback_reason"]=="cde_error"


def test_real_input_size_is_forwarded_for_long_routing():
    runtime=Runtime(route="fast")
    out=asyncio.run(select_serving_route(
        runtime,
        "visible tail",
        "fast",
        mode="shadow",
        rollout_percent=0,
        bucket_key="u1",
        input_chars=50000,
        long_context_chars=12000,
    ))
    assert runtime.contexts[0]["input_chars"]==50000
    assert out["cde"]=="long"


def test_canary_zero_percent_does_not_call_cde():
    runtime=Runtime(route="agent")
    out=asyncio.run(select_serving_route(
        runtime,"look up","fast",mode="canary",rollout_percent=0,bucket_key="u1"
    ))
    assert out["served_route"]=="fast"
    assert runtime.contexts==[]


def test_canary_full_percent_serves_cde():
    runtime=Runtime(route="agent")
    out=asyncio.run(select_serving_route(
        runtime,"look up","fast",mode="canary",rollout_percent=100,bucket_key="u1"
    ))
    assert out["served_route"]=="agent"
    assert out["routing_source"]=="cde"
