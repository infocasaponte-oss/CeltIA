from __future__ import annotations

import math
from collections.abc import Sequence

from .async_engine import AsyncCandidateScorer
from .robustness import option_order_report
from .schema import DecisionQuestion


def candidate_orders(candidates: Sequence[str], *, max_orders: int = 8) -> tuple[tuple[str, ...], ...]:
    """Return deterministic candidate permutations without factorial explosion."""
    values = tuple(candidates)
    if len(values) < 2:
        raise ValueError("at least two candidates are required")
    if max_orders < 2:
        raise ValueError("max_orders must be at least 2")

    orders: list[tuple[str, ...]] = []

    def add(order: tuple[str, ...]) -> None:
        if order not in orders and len(orders) < max_orders:
            orders.append(order)

    add(values)
    add(tuple(reversed(values)))
    for shift in range(1, len(values)):
        add(values[shift:] + values[:shift])
        if len(orders) >= max_orders:
            break
    return tuple(orders)


async def evaluate_option_order(
    scorer: AsyncCandidateScorer,
    context: object,
    question: DecisionQuestion,
    *,
    temperature: float = 1.0,
    max_orders: int = 8,
) -> dict:
    """Score deterministic permutations and report label-aligned instability."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    candidates = question.candidates()
    distributions: list[dict[str, float]] = []
    orders = candidate_orders(candidates, max_orders=max_orders)

    for order in orders:
        logits = list(await scorer.score(context, question, order))
        if len(logits) != len(order) or not logits:
            raise ValueError("scorer returned invalid logits")
        values = [float(v) for v in logits]
        if not all(math.isfinite(v) for v in values):
            raise ValueError("scorer returned invalid logits")
        scaled = [v / temperature for v in values]
        peak = max(scaled)
        exps = [math.exp(v - peak) for v in scaled]
        total = sum(exps)
        distributions.append(dict(zip(order, (v / total for v in exps), strict=True)))

    report = option_order_report(distributions)
    report["orders"] = [list(order) for order in orders]
    report["distributions"] = distributions
    return report
