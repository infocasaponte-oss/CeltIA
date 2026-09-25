from core.memory import Memory


def test_shadow_storage_keeps_cde_errors_and_latency():
    memory=Memory(":memory:")
    memory.record_decision_shadow(
        1,
        "fast",
        None,
        None,
        False,
        served_route="fast",
        routing_source="shadow",
        fallback_reason="cde_error",
        rollout_bucket=17,
        cde_latency_ms=4200,
    )
    memory.record_decision_shadow(
        1,
        "fast",
        "fast",
        .99,
        False,
        served_route="fast",
        routing_source="shadow",
        rollout_bucket=17,
        cde_latency_ms=1000,
    )
    summary=memory.decision_shadow_summary(days=1)
    assert summary["samples"] == 2
    assert summary["fallback_reasons"]["cde_error"] == 1
    assert summary["routing_sources"]["shadow"] == 2
    assert summary["avg_cde_latency_ms"] == 2600
    assert summary["p95_cde_latency_ms"] == 4200
