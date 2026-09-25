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
