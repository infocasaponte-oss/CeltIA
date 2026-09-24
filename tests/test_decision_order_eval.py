import asyncio

from celtia.decision.order_eval import candidate_orders, evaluate_option_order
from celtia.decision.schema import DecisionQuestion, DecisionType


class LabelStableScorer:
    async def score(self, context, question, candidates):
        values = {"a": 3.0, "b": 1.0, "c": 0.0}
        return [values[c] for c in candidates]


class PositionBiasedScorer:
    async def score(self, context, question, candidates):
        return [3.0] + [0.0] * (len(candidates) - 1)


def test_candidate_orders_are_bounded_and_deterministic():
    orders = candidate_orders(("a", "b", "c", "d"), max_orders=4)
    assert orders == (
        ("a", "b", "c", "d"),
        ("d", "c", "b", "a"),
        ("b", "c", "d", "a"),
        ("c", "d", "a", "b"),
    )


def test_option_order_eval_is_stable_for_label_based_scorer():
    q = DecisionQuestion("q", "choose", DecisionType.CHOICE, ("a", "b", "c"))
    report = asyncio.run(evaluate_option_order(LabelStableScorer(), {}, q))
    assert report["winner_stable"] is True
    assert report["max_deviation"] < 1e-12


def test_option_order_eval_detects_position_bias():
    q = DecisionQuestion("q", "choose", DecisionType.CHOICE, ("a", "b", "c"))
    report = asyncio.run(evaluate_option_order(PositionBiasedScorer(), {}, q))
    assert report["winner_stable"] is False
    assert report["max_deviation"] > .4
