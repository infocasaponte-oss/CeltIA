from __future__ import annotations

from celtia.decision.route_contract import ROUTES

DEFAULT_LONG_CONTEXT_CHARS = 12000
ROUTE_DECISION_PROMPT = "Select the most appropriate CeltIA execution route."


def route_decision_context(
    text: str,
    *,
    input_chars: int | None = None,
    long_context_chars: int = DEFAULT_LONG_CONTEXT_CHARS,
) -> dict:
    if input_chars is None:
        input_chars = len(text)
    if isinstance(input_chars, bool) or not isinstance(input_chars, int) or input_chars < len(text):
        raise ValueError("input_chars must be an integer at least as large as the visible user message")
    if isinstance(long_context_chars, bool) or not isinstance(long_context_chars, int) or long_context_chars <= 0:
        raise ValueError("long_context_chars must be a positive integer")
    return {
        "user_message": text[-DEFAULT_LONG_CONTEXT_CHARS:],
        "input_chars": input_chars,
        "long_context_chars": long_context_chars,
    }


def route_decision_question() -> dict:
    return {
        "id": "route",
        "prompt": ROUTE_DECISION_PROMPT,
        "type": "choice",
        "options": list(ROUTES),
    }
