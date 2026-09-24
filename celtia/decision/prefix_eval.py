from __future__ import annotations

import json
from collections.abc import Sequence

from .llm_scorer import AsyncLLMDecisionScorer
from .schema import DecisionQuestion


def common_prefix_chars(values: Sequence[str]) -> int:
    """Return the exact character length shared at the start of every value."""
    items = tuple(values)
    if len(items) < 2:
        raise ValueError("at least two prompts are required")
    shortest = min(len(value) for value in items)
    for index in range(shortest):
        char = items[0][index]
        if any(value[index] != char for value in items[1:]):
            return index
    return shortest


def prefix_reuse_report(prompts: Sequence[str]) -> dict:
    """Estimate deterministic pre-tokenization reuse from an exact shared prefix.

    This is a character-level proxy for deciding whether state/prefix caching is
    worth benchmarking with a real backend. It is not a latency or token claim.
    """
    items = tuple(prompts)
    if len(items) < 2:
        raise ValueError("at least two prompts are required")
    if any(not isinstance(value, str) or not value for value in items):
        raise ValueError("prompts must be non-empty strings")

    total_chars = sum(len(value) for value in items)
    shared_prefix_chars = common_prefix_chars(items)
    reusable_chars = shared_prefix_chars * (len(items) - 1)
    unique_chars = total_chars - reusable_chars
    return {
        "prompts": len(items),
        "total_chars": total_chars,
        "shared_prefix_chars": shared_prefix_chars,
        "reusable_chars": reusable_chars,
        "unique_chars": unique_chars,
        "reuse_fraction": reusable_chars / total_chars,
    }


def evaluate_scorer_prefix_reuse(context: object, questions: Sequence[DecisionQuestion]) -> dict:
    """Render the real scorer inputs and measure their exact shared prefix."""
    items = tuple(questions)
    if len(items) < 2:
        raise ValueError("at least two questions are required")

    prompts = []
    for question in items:
        messages = AsyncLLMDecisionScorer.messages_for(
            context,
            question,
            question.candidates(),
        )
        prompts.append(json.dumps(messages, ensure_ascii=False, separators=(",", ":")))

    report = prefix_reuse_report(prompts)
    report["context_serialized_chars"] = len(
        json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    )
    return report
