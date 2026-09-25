from core.decision_rollout import evaluate_shadow_readiness


def test_shadow_readiness_passes_with_sufficient_healthy_evidence():
    report=evaluate_shadow_readiness({
        "samples":250,
        "fallback_reasons":{"cde_error":1},
        "abstentions":5,
        "avg_cde_latency_ms":1200,
        "p95_cde_latency_ms":3500,
    })
    assert report["eligible_for_5pct_canary"] is True
    assert report["reasons"] == []


def test_shadow_readiness_fails_closed_on_missing_or_weak_evidence():
    report=evaluate_shadow_readiness({
        "samples":50,
        "fallback_reasons":{"cde_error":2},
        "abstentions":4,
        "avg_cde_latency_ms":None,
        "p95_cde_latency_ms":None,
    })
    assert report["eligible_for_5pct_canary"] is False
    assert "samples_gte_200" in report["reasons"]
    assert "cde_error_rate_lte_1pct" in report["reasons"]
    assert "abstention_rate_lte_5pct" in report["reasons"]
    assert "avg_cde_latency_lte_3000ms" in report["reasons"]
    assert "p95_cde_latency_lte_8000ms" in report["reasons"]


def test_shadow_readiness_uses_error_rate_not_generic_disagreement():
    report=evaluate_shadow_readiness({
        "samples":200,
        "fallback_reasons":{},
        "abstentions":0,
        "avg_cde_latency_ms":2000,
        "p95_cde_latency_ms":7000,
        "top_disagreements":[{"heuristic":"fast","cde":"think","count":100}],
    })
    assert report["eligible_for_5pct_canary"] is True
