from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from .robustness import ood_signal


def probability_distribution(
    candidates: Sequence[str],
    logits: Sequence[float],
    *,
    temperature: float,
) -> dict[str, float]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if len(logits) != len(candidates) or not logits:
        raise ValueError("scorer returned invalid logits")
    values = [float(value) for value in logits]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("scorer returned invalid logits")

    scaled = [value / temperature for value in values]
    peak = max(scaled)
    exps = [math.exp(value - peak) for value in scaled]
    total = sum(exps)
    probabilities = [value / total for value in exps]
    return dict(zip(candidates, probabilities, strict=True))


def decision_policy(
    probabilities: Mapping[str, float],
    *,
    abstain_below: float,
    reject_suspected_ood: bool,
    ood_entropy_threshold: float,
    ood_margin_threshold: float,
) -> dict:
    if not 0 <= abstain_below <= 1:
        raise ValueError("abstain_below must be between 0 and 1")

    candidates = tuple(probabilities)
    if not candidates:
        raise ValueError("probabilities cannot be empty")

    best = max(candidates, key=probabilities.__getitem__)
    confidence = float(probabilities[best])
    risk = ood_signal(
        probabilities,
        entropy_threshold=ood_entropy_threshold,
        margin_threshold=ood_margin_threshold,
    )
    low_confidence = confidence < abstain_below
    rejected_ood = reject_suspected_ood and risk["suspected_ood"]
    abstained = low_confidence or rejected_ood
    reason = "low_confidence" if low_confidence else ("suspected_ood" if rejected_ood else None)

    return {
        "decision": None if abstained else best,
        "confidence": confidence,
        "abstained": abstained,
        "abstention_reason": reason,
        "normalized_entropy": risk["normalized_entropy"],
        "margin": risk["margin"],
    }
