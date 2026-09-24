from __future__ import annotations
import math
from collections.abc import Sequence

def negative_log_likelihood(probabilities: Sequence[float], targets: Sequence[int], eps: float = 1e-12) -> float:
    if len(probabilities) != len(targets) or not probabilities: raise ValueError("probabilities and targets must be non-empty and aligned")
    return -sum(math.log(max(eps, min(1.0, p if y else 1.0 - p))) for p, y in zip(probabilities, targets, strict=True)) / len(targets)

def brier_score(probabilities: Sequence[float], targets: Sequence[int]) -> float:
    if len(probabilities) != len(targets) or not probabilities: raise ValueError("probabilities and targets must be non-empty and aligned")
    return sum((p - y) ** 2 for p, y in zip(probabilities, targets, strict=True)) / len(targets)

def expected_calibration_error(confidences: Sequence[float], correct: Sequence[bool], bins: int = 10) -> float:
    if len(confidences) != len(correct) or not confidences or bins < 1: raise ValueError("invalid calibration inputs")
    n=len(confidences); total=0.0
    for i in range(bins):
        lo=i/bins; hi=(i+1)/bins
        idx=[j for j,c in enumerate(confidences) if lo <= c <= hi if (i==bins-1 or c < hi)]
        if not idx: continue
        avg_c=sum(confidences[j] for j in idx)/len(idx); acc=sum(bool(correct[j]) for j in idx)/len(idx)
        total += len(idx)/n * abs(avg_c-acc)
    return total
