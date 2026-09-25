from __future__ import annotations

from celtia.decision.route_contract import ROUTES
from core.config import settings

ROUTE_DECISION_PROMPT = "Select the most appropriate CeltIA execution route."


def route_decision_context(text: str, *, input_chars: int | None = None) -> dict:
    if input_chars is None:
        input_chars = len(text)
    if isinstance(input_chars, bool) or not isinstance(input_chars, int) or input_chars < len(text):
        raise ValueError("input_chars must be an integer at least as large as the visible user message")
    return {
        "user_message": text[-12000:],
        "input_chars": input_chars,
        "long_context_chars": settings.router_long_context_chars,
    }


def route_decision_question() -> dict:
    return {
        "id": "route",
        "prompt": ROUTE_DECISION_PROMPT,
        "type": "choice",
        "options": list(ROUTES),
    }
