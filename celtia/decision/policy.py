from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from .robustness import ood_signal



def _finite_number(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def validate_policy_settings(
    *,
    abstain_below: object,
    temperature: object,
    reject_suspected_ood: object,
    ood_entropy_threshold: object,
    ood_margin_threshold: object,
) -> dict:
    abstain = _finite_number("abstain_below", abstain_below)
    temp = _finite_number("temperature", temperature)
    entropy = _finite_number("ood_entropy_threshold", ood_entropy_threshold)
    margin = _finite_number("ood_margin_threshold", ood_margin_threshold)
    if not 0 <= abstain <= 1:
        raise ValueError("abstain_below must be between 0 and 1")
    if temp <= 0:
        raise ValueError("temperature must be positive")
    if not 0 <= entropy <= 1:
        raise ValueError("ood_entropy_threshold must be between 0 and 1")
    if not 0 <= margin <= 1:
        raise ValueError("ood_margin_threshold must be between 0 and 1")
    if not isinstance(reject_suspected_ood, bool):
        raise ValueError("reject_suspected_ood must be boolean")
    return {
        "abstain_below": abstain,
        "temperature": temp,
        "reject_suspected_ood": reject_suspected_ood,
        "ood_entropy_threshold": entropy,
        "ood_margin_threshold": margin,
    }



def probability_distribution(
    candidates: Sequence[str],
    logits: Sequence[float],
    *,
    temperature: float,
) -> dict[str, float]:
    temperature = _finite_number("temperature", temperature)
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if len(logits) != len(candidates) or not logits:
        raise ValueError("scorer returned invalid logits")
    if any(isinstance(value, bool) for value in logits):
        raise ValueError("scorer returned invalid logits")
    try:
        values = [float(value) for value in logits]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("scorer returned invalid logits") from exc
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
    settings = validate_policy_settings(
        abstain_below=abstain_below,
        temperature=1.0,
        reject_suspected_ood=reject_suspected_ood,
        ood_entropy_threshold=ood_entropy_threshold,
        ood_margin_threshold=ood_margin_threshold,
    )
    abstain_below = settings["abstain_below"]
    reject_suspected_ood = settings["reject_suspected_ood"]
    ood_entropy_threshold = settings["ood_entropy_threshold"]
    ood_margin_threshold = settings["ood_margin_threshold"]

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
        "suspected_ood": bool(risk["suspected_ood"]),
        "normalized_entropy": risk["normalized_entropy"],
        "margin": risk["margin"],
    }
