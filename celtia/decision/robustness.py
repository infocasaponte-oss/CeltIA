from __future__ import annotations
import math
from collections.abc import Mapping, Sequence

def normalized_entropy(probabilities: Mapping[str, float]) -> float:
    values=[float(v) for v in probabilities.values()]
    if not values or any(v < 0 or not math.isfinite(v) for v in values):
        raise ValueError("invalid probabilities")
    total=sum(values)
    if total <= 0:
        raise ValueError("probabilities must sum to a positive value")
    p=[v/total for v in values]
    if len(p) == 1:
        return 0.0
    return -sum(v*math.log(v) for v in p if v > 0)/math.log(len(p))

def decision_margin(probabilities: Mapping[str, float]) -> float:
    values=sorted((float(v) for v in probabilities.values()),reverse=True)
    if len(values) < 2:
        return 1.0
    return values[0]-values[1]

def option_order_max_deviation(distributions: Sequence[Mapping[str,float]]) -> float:
    if not distributions:
        raise ValueError("at least one distribution is required")
    keys=set(distributions[0])
    if not keys or any(set(d) != keys for d in distributions):
        raise ValueError("all distributions must contain the same candidates")
    means={k:sum(float(d[k]) for d in distributions)/len(distributions) for k in keys}
    return max(abs(float(d[k])-means[k]) for d in distributions for k in keys)

def ood_signal(probabilities: Mapping[str,float], *, entropy_threshold: float=.90,
               margin_threshold: float=.10) -> dict:
    entropy=normalized_entropy(probabilities)
    margin=decision_margin(probabilities)
    return {
        "suspected_ood": entropy >= entropy_threshold or margin <= margin_threshold,
        "normalized_entropy": entropy,
        "margin": margin,
    }


def option_order_report(distributions: Sequence[Mapping[str, float]]) -> dict:
    """Summarize label-aligned stability across option permutations."""
    if len(distributions) < 2:
        raise ValueError("at least two distributions are required")
    keys = set(distributions[0])
    if not keys or any(set(d) != keys for d in distributions):
        raise ValueError("all distributions must contain the same candidates")
    winners = [max(d, key=d.get) for d in distributions]
    return {
        "permutations": len(distributions),
        "max_deviation": option_order_max_deviation(distributions),
        "winner_stable": len(set(winners)) == 1,
        "winners": winners,
    }
