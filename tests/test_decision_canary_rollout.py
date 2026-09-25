from core.decision_rollout import evaluate_canary_readiness
from core.memory import Memory


def test_canary_readiness_passes_stage_5():
    report=evaluate_canary_readiness({
        "rollout_percent":5,
        "samples":120,
        "abstentions":2,
        "fallback_reasons":{"cde_error":1,"cde_invalid_route":0,"cde_suspected_ood":8},
        "avg_cde_latency_ms":1400,
        "p95_cde_latency_ms":4200,
    })
    assert report["eligible_for_next_stage"] is True
    assert report["next_stage_percent"] == 20


def test_canary_readiness_fails_on_operational_fallbacks():
    report=evaluate_canary_readiness({
        "rollout_percent":20,
        "samples":200,
        "abstentions":9,
        "fallback_reasons":{"cde_error":3,"cde_invalid_route":1},
        "avg_cde_latency_ms":1500,
        "p95_cde_latency_ms":5000,
    })
    assert report["eligible_for_next_stage"] is False
    assert "cde_error_rate_lte_1pct" in report["reasons"]
    assert "operational_fallback_rate_lte_5pct" in report["reasons"]


def test_ood_fallbacks_are_reported_but_not_operational_failures():
    report=evaluate_canary_readiness({
        "rollout_percent":50,
        "samples":300,
        "abstentions":0,
        "fallback_reasons":{"cde_suspected_ood":60},
        "avg_cde_latency_ms":1000,
        "p95_cde_latency_ms":3000,
    })
    assert report["eligible_for_next_stage"] is True
    assert report["metrics"]["ood_fallback_rate"] == 0.2
    assert report["metrics"]["operational_fallback_rate"] == 0.0


def test_canary_summary_isolated_by_rollout_stage():
    memory=Memory(":memory:")
    memory.record_decision_shadow(
        1,"fast","fast",.99,False,
        served_route="fast",routing_source="cde",
        cde_latency_ms=1000,rollout_percent=5,
    )
    memory.record_decision_shadow(
        1,"fast","think",.9,False,
        served_route="think",routing_source="cde",
        cde_latency_ms=1200,rollout_percent=20,
    )
    five=memory.decision_canary_summary(5,days=1)
    twenty=memory.decision_canary_summary(20,days=1)
    assert five["samples"] == 1
    assert five["rollout_percent"] == 5
    assert five["served_routes"]["fast"] == 1
    assert twenty["samples"] == 1
    assert twenty["served_routes"]["think"] == 1
