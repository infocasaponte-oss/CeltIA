from __future__ import annotations

from collections.abc import Mapping

MAX_JSON_DEPTH = 128


def validate_json_depth(value: object, *, max_depth: int = MAX_JSON_DEPTH) -> None:
    """Reject pathologically deep JSON-like inputs without recursive traversal."""
    if max_depth < 1:
        raise ValueError("max_depth must be positive")

    stack: list[tuple[object, int]] = [(value, 0)]
    seen_containers: set[int] = set()

    while stack:
        current, depth = stack.pop()
        if depth > max_depth:
            raise ValueError(f"JSON nesting exceeds {max_depth} levels")

        if isinstance(current, Mapping):
            identity = id(current)
            if identity in seen_containers:
                continue
            seen_containers.add(identity)
            stack.extend((item, depth + 1) for item in current.values())
            continue

        if isinstance(current, (list, tuple)):
            identity = id(current)
            if identity in seen_containers:
                continue
            seen_containers.add(identity)
            stack.extend((item, depth + 1) for item in current)
