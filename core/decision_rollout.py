from __future__ import annotations

SHADOW_MIN_SAMPLES = 200
SHADOW_MAX_ERROR_RATE = 0.01
SHADOW_MAX_ABSTENTION_RATE = 0.05
SHADOW_MAX_AVG_LATENCY_MS = 3000
SHADOW_MAX_P95_LATENCY_MS = 8000


def evaluate_shadow_readiness(summary: dict) -> dict:
    """Evaluate whether shadow evidence is sufficient to start a 5% CDE canary.

    This is an operational gate, not an accuracy gate. Offline accuracy/OOD
    promotion evidence is evaluated separately.
    """
    samples = int(summary.get("samples") or 0)
    fallback_reasons = summary.get("fallback_reasons") or {}
    cde_errors = int(fallback_reasons.get("cde_error") or 0)
    abstentions = int(summary.get("abstentions") or 0)
    avg_latency = summary.get("avg_cde_latency_ms")
    p95_latency = summary.get("p95_cde_latency_ms")

    error_rate = (cde_errors / samples) if samples else None
    abstention_rate = (abstentions / samples) if samples else None

    checks = {
        "samples_gte_200": samples >= SHADOW_MIN_SAMPLES,
        "cde_error_rate_lte_1pct": error_rate is not None and error_rate <= SHADOW_MAX_ERROR_RATE,
        "abstention_rate_lte_5pct": (
            abstention_rate is not None and abstention_rate <= SHADOW_MAX_ABSTENTION_RATE
        ),
        "avg_cde_latency_lte_3000ms": (
            avg_latency is not None and float(avg_latency) <= SHADOW_MAX_AVG_LATENCY_MS
        ),
        "p95_cde_latency_lte_8000ms": (
            p95_latency is not None and float(p95_latency) <= SHADOW_MAX_P95_LATENCY_MS
        ),
    }

    reasons = [name for name, passed in checks.items() if not passed]
    return {
        "eligible_for_5pct_canary": all(checks.values()),
        "checks": checks,
        "reasons": reasons,
        "metrics": {
            "samples": samples,
            "cde_errors": cde_errors,
            "cde_error_rate": error_rate,
            "abstentions": abstentions,
            "abstention_rate": abstention_rate,
            "avg_cde_latency_ms": avg_latency,
            "p95_cde_latency_ms": p95_latency,
        },
        "thresholds": {
            "min_samples": SHADOW_MIN_SAMPLES,
            "max_cde_error_rate": SHADOW_MAX_ERROR_RATE,
            "max_abstention_rate": SHADOW_MAX_ABSTENTION_RATE,
            "max_avg_cde_latency_ms": SHADOW_MAX_AVG_LATENCY_MS,
            "max_p95_cde_latency_ms": SHADOW_MAX_P95_LATENCY_MS,
        },
    }


CANARY_STAGE_MIN_SAMPLES = {
    5: 100,
    20: 200,
    50: 300,
    100: 500,
}
CANARY_NEXT_STAGE = {
    5: 20,
    20: 50,
    50: 100,
    100: None,
}
CANARY_MAX_OPERATIONAL_FALLBACK_RATE = 0.05
CANARY_MAX_ERROR_RATE = 0.01
CANARY_MAX_ABSTENTION_RATE = 0.05
CANARY_MAX_AVG_LATENCY_MS = 3000
CANARY_MAX_P95_LATENCY_MS = 8000


def evaluate_canary_readiness(summary: dict) -> dict:
    percent = int(summary.get("rollout_percent") or 0)
    if percent not in CANARY_STAGE_MIN_SAMPLES:
        raise ValueError("rollout_percent must be one of 5, 20, 50, 100")

    samples = int(summary.get("samples") or 0)
    fallback_reasons = summary.get("fallback_reasons") or {}
    cde_errors = int(fallback_reasons.get("cde_error") or 0)
    abstentions = int(summary.get("abstentions") or 0)
    invalid_routes = int(fallback_reasons.get("cde_invalid_route") or 0)
    ood_fallbacks = int(fallback_reasons.get("cde_suspected_ood") or 0)
    operational_fallbacks = cde_errors + abstentions + invalid_routes

    error_rate = (cde_errors / samples) if samples else None
    abstention_rate = (abstentions / samples) if samples else None
    operational_fallback_rate = (operational_fallbacks / samples) if samples else None
    ood_fallback_rate = (ood_fallbacks / samples) if samples else None
    avg_latency = summary.get("avg_cde_latency_ms")
    p95_latency = summary.get("p95_cde_latency_ms")

    min_samples = CANARY_STAGE_MIN_SAMPLES[percent]
    checks = {
        f"samples_gte_{min_samples}": samples >= min_samples,
        "cde_error_rate_lte_1pct": error_rate is not None and error_rate <= CANARY_MAX_ERROR_RATE,
        "abstention_rate_lte_5pct": (
            abstention_rate is not None and abstention_rate <= CANARY_MAX_ABSTENTION_RATE
        ),
        "operational_fallback_rate_lte_5pct": (
            operational_fallback_rate is not None
            and operational_fallback_rate <= CANARY_MAX_OPERATIONAL_FALLBACK_RATE
        ),
        "avg_cde_latency_lte_3000ms": (
            avg_latency is not None and float(avg_latency) <= CANARY_MAX_AVG_LATENCY_MS
        ),
        "p95_cde_latency_lte_8000ms": (
            p95_latency is not None and float(p95_latency) <= CANARY_MAX_P95_LATENCY_MS
        ),
    }

    reasons = [name for name, passed in checks.items() if not passed]
    next_stage = CANARY_NEXT_STAGE[percent]
    return {
        "stage_percent": percent,
        "next_stage_percent": next_stage,
        "eligible_for_next_stage": all(checks.values()),
        "checks": checks,
        "reasons": reasons,
        "metrics": {
            "samples": samples,
            "cde_errors": cde_errors,
            "cde_error_rate": error_rate,
            "abstentions": abstentions,
            "abstention_rate": abstention_rate,
            "invalid_routes": invalid_routes,
            "operational_fallbacks": operational_fallbacks,
            "operational_fallback_rate": operational_fallback_rate,
            "ood_fallbacks": ood_fallbacks,
            "ood_fallback_rate": ood_fallback_rate,
            "avg_cde_latency_ms": avg_latency,
            "p95_cde_latency_ms": p95_latency,
        },
        "thresholds": {
            "min_samples": min_samples,
            "max_cde_error_rate": CANARY_MAX_ERROR_RATE,
            "max_abstention_rate": CANARY_MAX_ABSTENTION_RATE,
            "max_operational_fallback_rate": CANARY_MAX_OPERATIONAL_FALLBACK_RATE,
            "max_avg_cde_latency_ms": CANARY_MAX_AVG_LATENCY_MS,
            "max_p95_cde_latency_ms": CANARY_MAX_P95_LATENCY_MS,
        },
    }
