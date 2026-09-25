from __future__ import annotations

from core.config import settings

ROUTES = ("fast", "think", "code", "agent", "long")

ROUTE_DECISION_PROMPT = """Select the most appropriate CeltIA execution route using this stable contract:
- fast: short, direct tasks that need little deliberation and no code execution or external tools.
- think: tasks that primarily need multi-step reasoning, analysis, comparison, planning, trade-off evaluation, or careful derivation.
- code: tasks whose primary output is code, debugging, a patch, a query, tests, or implementation guidance tied directly to source code.
- agent: tasks that require current/external information, web lookup, tools, execution, or interaction with external systems.
- long: tasks whose effective input/context is large enough to require long-context handling; use the supplied input_chars and long_context_chars rather than merely words such as 'long document'.

Choose based on the task itself. Do not use any legacy-router decision or benchmark label; those are not provided to the scorer."""


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
