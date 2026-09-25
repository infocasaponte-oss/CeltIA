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
    # Legacy rows from earlier telemetry generations must not count for v4 readiness.
    memory.record_decision_shadow(
        1,
        "fast",
        "fast",
        .9,
        False,
        served_route="fast",
        routing_source="shadow",
        cde_latency_ms=900,
        telemetry_version=2,
    )
    memory.record_decision_shadow(
        1,
        "fast",
        "fast",
        .9,
        False,
        served_route="fast",
        routing_source="shadow",
        cde_latency_ms=950,
        telemetry_version=3,
    )
    summary=memory.decision_shadow_readiness_summary(days=1)
    assert summary["samples"] == 2
    assert summary["telemetry_version"] == 4
    assert summary["fallback_reasons"]["cde_error"] == 1
    assert summary["routing_sources"]["shadow"] == 2
    assert summary["avg_cde_latency_ms"] == 2600
    assert summary["p95_cde_latency_ms"] == 4200
