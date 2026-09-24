# Copyright (c) 2026 Luis Manuel Cousido Hermida. All rights reserved.
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)
ROUTES = ("fast", "think", "code", "agent", "long")

async def evaluate_route_shadow(runtime, text: str, heuristic_route: str) -> dict | None:
    """Evaluate CDE routing without changing the route selected by the production router."""
    try:
        result = (await runtime.decide(
            {"user_message": text[-12000:], "heuristic_route": heuristic_route},
            [{"id":"route","prompt":"Select the most appropriate CeltIA execution route.",
              "type":"choice","options":list(ROUTES)}],
        ))[0]
        return {"heuristic": heuristic_route, "cde": result.decision,
                "confidence": result.confidence, "abstained": result.abstained}
    except Exception as exc:
        logger.warning("CDE shadow routing failed: %s", exc)
        return None
