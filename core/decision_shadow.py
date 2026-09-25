# Copyright (c) 2026 Luis Manuel Cousido Hermida. All rights reserved.
from __future__ import annotations
import logging

from core.decision_routes import combine_route_results, route_decision_context, route_decision_questions

logger = logging.getLogger(__name__)
async def evaluate_route_shadow(runtime, text: str, heuristic_route: str, *, long_context_chars: int = 12000) -> dict | None:
    """Evaluate CDE routing without changing the route selected by the production router."""
    try:
        results, usage = await runtime.decide_with_usage(
            route_decision_context(text, long_context_chars=long_context_chars),
            route_decision_questions(),
        )
        if len(results) != 2:
            raise RuntimeError("CDE routing requires route and OOD results")
        combined = combine_route_results(results[0], results[1])
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
