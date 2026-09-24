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
    if len(shared) < 2:
        raise ValueError("at least two samples per workload are required")

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
