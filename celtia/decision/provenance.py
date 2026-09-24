from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


@dataclass(frozen=True)
class TrainingSource:
    id: str
    source: str
    revision: str
    license: str
    sha256: str
    allowed_for_training: bool
    allowed_for_distribution: bool
    notice: str | None = None


def validate_training_sources(
    sources: Iterable[TrainingSource],
    *,
    require_distribution: bool = False,
) -> tuple[TrainingSource, ...]:
    """Validate provenance before any source is admitted to a CDE training run."""
    rows = tuple(sources)
    if not rows:
        raise ValueError("at least one training source is required")

    ids: set[str] = set()
    for row in rows:
        if not row.id.strip():
            raise ValueError("training source id is required")
        if row.id in ids:
            raise ValueError(f"duplicate training source id: {row.id}")
        ids.add(row.id)

        for name, value in (
            ("source", row.source),
            ("revision", row.revision),
            ("license", row.license),
        ):
            if not value.strip():
                raise ValueError(f"training source {row.id} is missing {name}")

        if not _SHA256_RE.fullmatch(row.sha256):
            raise ValueError(f"training source {row.id} has invalid sha256")
        if not row.allowed_for_training:
            raise ValueError(f"training source {row.id} is not permitted for training")
        if require_distribution and not row.allowed_for_distribution:
            raise ValueError(f"training source {row.id} is not permitted for distribution")

    return rows
