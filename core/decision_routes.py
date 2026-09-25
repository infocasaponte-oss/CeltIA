from __future__ import annotations

from celtia.decision.route_contract import ROUTES
from core.ood_structure import structural_ood_reason

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


def route_ood_question() -> dict:
    return {
        "id": "route_ood",
        "prompt": "Is this input outside the CeltIA routing domain?",
        "type": "boolean",
    }


def route_decision_questions() -> list[dict]:
    return [route_decision_question(), route_ood_question()]


def combine_route_results(
    route_result,
    ood_result,
    text: str | None = None,
    *,
    input_chars: int | None = None,
    long_context_chars: int = DEFAULT_LONG_CONTEXT_CHARS,
) -> dict:
    """Merge the route head, the semantic OOD head and the deterministic guards.

    Trusted size metadata is authoritative for ``long``: an input at or above the
    long-context threshold is routed ``long`` without depending on model confidence,
    unless the OOD head or the structural detector rejects it as a non-task.
    """
    try:
        ood_probability = float(ood_result.probabilities["true"])
        in_domain_probability = float(ood_result.probabilities["false"])
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid explicit OOD result") from exc
    explicit_ood = ood_probability > in_domain_probability
    structural_reason = structural_ood_reason(text) if text is not None else None
    rejected = explicit_ood or structural_reason is not None
    size_chars = input_chars if input_chars is not None else (len(text) if text is not None else 0)
    size_long = size_chars >= long_context_chars and not rejected
    suspected_ood = rejected or (bool(route_result.suspected_ood) and not size_long)
    abstained = False if size_long else bool(route_result.abstained)
    return {
        "decision": "long" if size_long else (None if abstained else route_result.decision),
        "confidence": route_result.confidence,
        "abstained": abstained,
        "abstention_reason": None if size_long else route_result.abstention_reason,
        "suspected_ood": suspected_ood,
        "normalized_entropy": route_result.normalized_entropy,
        "margin": route_result.margin,
        "ood_probability": ood_probability,
        "ood_classifier_confidence": max(ood_probability, in_domain_probability),
        "ood_classifier_decision": explicit_ood,
    }
