# Copyright (c) 2026 Luis Manuel Cousido Hermida. All rights reserved.
from __future__ import annotations
import asyncio
import logging

from core.decision_routes import (
    combine_route_results,
    route_decision_context,
    route_decision_question,
    route_ood_question,
)

logger = logging.getLogger(__name__)
async def evaluate_route_shadow(
    runtime,
    text: str,
    heuristic_route: str,
    *,
    input_chars: int | None = None,
    long_context_chars: int = 12000,
) -> dict | None:
    """Evaluate CDE routing without changing the route selected by the production router."""
    try:
        context = route_decision_context(
            text,
            input_chars=input_chars,
            long_context_chars=long_context_chars,
        )
        (route_results, route_usage), (ood_results, ood_usage) = await asyncio.gather(
            runtime.decide_with_usage(context, [route_decision_question()]),
            runtime.decide_with_usage(context, [route_ood_question()]),
        )
        if len(route_results) != 1 or len(ood_results) != 1:
            raise RuntimeError("CDE routing requires one route and one OOD result")
        usage = {
            "prompt_tokens": int(route_usage.get("prompt_tokens", 0)) + int(ood_usage.get("prompt_tokens", 0)),
            "completion_tokens": int(route_usage.get("completion_tokens", 0)) + int(ood_usage.get("completion_tokens", 0)),
            "total_tokens": int(route_usage.get("total_tokens", 0)) + int(ood_usage.get("total_tokens", 0)),
            "models": list(dict.fromkeys([
                *(route_usage.get("models") or []),
                *(ood_usage.get("models") or []),
            ])),
        }
        combined = combine_route_results(
            route_results[0],
            ood_results[0],
            text,
            input_chars=input_chars,
            long_context_chars=long_context_chars,
        )
        return {
            "heuristic": heuristic_route,
            "cde": combined["decision"],
            "confidence": combined["confidence"],
            "abstained": combined["abstained"],
            "abstention_reason": combined["abstention_reason"],
            "suspected_ood": combined["suspected_ood"],
            "normalized_entropy": combined["normalized_entropy"],
            "margin": combined["margin"],
            "ood_probability": combined["ood_probability"],
            "ood_classifier_confidence": combined["ood_classifier_confidence"],
            "ood_classifier_decision": combined["ood_classifier_decision"],
            "usage": usage,
        }
    except Exception as exc:
        logger.warning("CDE shadow routing failed: %s", exc)
        return None
