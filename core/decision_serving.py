from __future__ import annotations

import hashlib
import time

from core.decision_shadow import evaluate_route_shadow

ROUTING_MODES = {"legacy", "shadow", "canary", "cde"}
VALID_ROUTES = {"fast", "think", "code", "agent", "long"}


def effective_routing_mode(configured_mode: str, legacy_shadow_flag: bool) -> str:
    """Keep the old shadow flag working while the new rollout mode is adopted."""
    mode = (configured_mode or "legacy").strip().lower()
    if mode not in ROUTING_MODES:
        raise ValueError(f"unsupported decision routing mode: {configured_mode!r}")
    if mode == "legacy" and legacy_shadow_flag:
        return "shadow"
    return mode


def rollout_bucket(key: str) -> int:
    """Stable 0..99 bucket so one session/user does not flap between routers."""
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 100


async def select_serving_route(
    runtime,
    text: str,
    heuristic_route: str,
    *,
    mode: str,
    rollout_percent: int,
    bucket_key: str,
    input_chars: int | None = None,
    long_context_chars: int = 12000,
) -> dict:
    if mode not in ROUTING_MODES:
        raise ValueError(f"unsupported decision routing mode: {mode!r}")
    if not 0 <= rollout_percent <= 100:
        raise ValueError("rollout_percent must be between 0 and 100")
    if heuristic_route not in VALID_ROUTES:
        raise ValueError(f"unsupported heuristic route: {heuristic_route!r}")

    bucket = rollout_bucket(bucket_key)
    selected_for_cde = mode == "cde" or (mode == "canary" and bucket < rollout_percent)
    should_evaluate = mode in {"shadow", "cde"} or (mode == "canary" and selected_for_cde)

    base = {
        "heuristic": heuristic_route,
        "cde": None,
        "served_route": heuristic_route,
        "routing_source": "legacy",
        "fallback_reason": None,
        "rollout_bucket": bucket,
        "selected_for_cde": selected_for_cde,
        "evaluated_cde": should_evaluate,
        "cde_latency_ms": None,
    }
    if not should_evaluate:
        return base

    started = time.monotonic()
    shadow = await evaluate_route_shadow(
        runtime,
        text,
        heuristic_route,
        input_chars=input_chars,
        long_context_chars=long_context_chars,
    )
    latency_ms = max(0, int((time.monotonic() - started) * 1000))
    if shadow is None:
        return {
            **base,
            "routing_source": "legacy_fallback" if selected_for_cde else "shadow",
            "fallback_reason": "cde_error",
            "cde_latency_ms": latency_ms,
        }

    result = {**base, **shadow, "cde_latency_ms": latency_ms}
    if mode == "shadow":
        result["routing_source"] = "shadow"
        return result

    if not selected_for_cde:
        result["routing_source"] = "legacy"
        return result

    cde_route = shadow.get("cde")
    if shadow.get("abstained"):
        result["routing_source"] = "legacy_fallback"
        result["fallback_reason"] = "cde_abstained"
        return result
    if shadow.get("suspected_ood"):
        # OOD is observable during routing rollout but is not yet a serving/blocking policy.
        result["routing_source"] = "legacy_fallback"
        result["fallback_reason"] = "cde_suspected_ood"
        return result
    if cde_route not in VALID_ROUTES:
        result["routing_source"] = "legacy_fallback"
        result["fallback_reason"] = "cde_invalid_route"
        return result

    result["served_route"] = cde_route
    result["routing_source"] = "cde"
    return result
