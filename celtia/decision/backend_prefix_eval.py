from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence


def cached_prompt_tokens(usage: object) -> int | None:
    """Read OpenAI-compatible cached prompt token metadata when available."""
    if not isinstance(usage, Mapping):
        return None
    details = usage.get("prompt_tokens_details")
    if not isinstance(details, Mapping):
        return None
    raw = details.get("cached_tokens")
    if raw is None or isinstance(raw, bool):
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if value >= 0 else None


def _finite_latencies(samples: Sequence[Mapping]) -> list[float]:
    values: list[float] = []
    for sample in samples:
        raw = sample.get("latency_ms")
        if isinstance(raw, bool):
            raise ValueError("latency_ms must be a finite non-negative number")
        try:
            value = float(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("latency_ms must be a finite non-negative number") from exc
        if not math.isfinite(value) or value < 0:
            raise ValueError("latency_ms must be a finite non-negative number")
        values.append(value)
    if not values:
        raise ValueError("at least one sample is required")
    return values


def latency_summary(samples: Sequence[Mapping]) -> dict:
    """Summarize backend timing samples without third-party dependencies."""
    values = sorted(_finite_latencies(samples))
    p95_index = max(0, math.ceil(0.95 * len(values)) - 1)
    return {
        "samples": len(values),
        "median_ms": statistics.median(values),
        "p95_ms": values[p95_index],
        "min_ms": values[0],
        "max_ms": values[-1],
    }


def round_order_sequence(rounds: int, *, first: str = "control-first") -> tuple[str, ...]:
    """Alternate workload order to reduce systematic warm-cache/order bias."""
    if rounds < 1:
        raise ValueError("rounds must be at least 1")
    if first not in {"control-first", "shared-first"}:
        raise ValueError("first must be control-first or shared-first")
    other = "shared-first" if first == "control-first" else "control-first"
    return tuple(first if index % 2 == 0 else other for index in range(rounds))


def backend_prefix_report(
    shared_samples: Sequence[Mapping],
    control_samples: Sequence[Mapping],
    *,
    skip_first: bool = True,
) -> dict:
    """Compare shared-prefix calls with same-size unique-prefix controls.

    The first call can be excluded because a prefix cache cannot help until a
    matching prefix has already been observed by the backend.
    """
    shared = list(shared_samples)
    control = list(control_samples)
    if len(shared) != len(control):
        raise ValueError("shared and control sample counts must match")
    minimum = 2 if skip_first else 1
    if len(shared) < minimum:
        raise ValueError(f"at least {minimum} samples per workload are required")

    shared_eval = shared[1:] if skip_first else shared
    control_eval = control[1:] if skip_first else control
    shared_summary = latency_summary(shared_eval)
    control_summary = latency_summary(control_eval)

    shared_median = shared_summary["median_ms"]
    control_median = control_summary["median_ms"]
    speedup_ratio = None if shared_median == 0 else control_median / shared_median
    latency_reduction = None if control_median == 0 else 1 - (shared_median / control_median)

    shared_cached = [cached_prompt_tokens(sample.get("usage")) for sample in shared_eval]
    control_cached = [cached_prompt_tokens(sample.get("usage")) for sample in control_eval]
    cached_metadata_available = any(value is not None for value in shared_cached + control_cached)

    return {
        "skip_first": skip_first,
        "shared": shared_summary,
        "control": control_summary,
        "median_speedup_ratio": speedup_ratio,
        "median_latency_reduction": latency_reduction,
        "cached_token_metadata_available": cached_metadata_available,
        "shared_cached_prompt_tokens": shared_cached if cached_metadata_available else None,
        "control_cached_prompt_tokens": control_cached if cached_metadata_available else None,
    }



def backend_prefix_round_report(rounds: Sequence[Mapping]) -> dict:
    """Aggregate balanced benchmark rounds, dropping each workload's warm-up call."""
    items = list(rounds)
    if not items:
        raise ValueError("at least one benchmark round is required")

    shared_eval: list[Mapping] = []
    control_eval: list[Mapping] = []
    round_reports: list[dict] = []

    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise ValueError("benchmark rounds must be mappings")
        shared = item.get("shared_samples")
        control = item.get("control_samples")
        if not isinstance(shared, Sequence) or isinstance(shared, (str, bytes)):
            raise ValueError("each round requires shared_samples")
        if not isinstance(control, Sequence) or isinstance(control, (str, bytes)):
            raise ValueError("each round requires control_samples")
        shared_list = list(shared)
        control_list = list(control)
        report = backend_prefix_report(shared_list, control_list, skip_first=True)
        report["round"] = index + 1
        report["order"] = item.get("order")
        round_reports.append(report)
        shared_eval.extend(shared_list[1:])
        control_eval.extend(control_list[1:])

    aggregate = backend_prefix_report(shared_eval, control_eval, skip_first=False)
    aggregate["round_count"] = len(items)
    aggregate["round_reports"] = round_reports
    return aggregate
