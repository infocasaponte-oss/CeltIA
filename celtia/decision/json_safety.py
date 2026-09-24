from __future__ import annotations

from collections.abc import Mapping

MAX_JSON_DEPTH = 128


def validate_json_depth(value: object, *, max_depth: int = MAX_JSON_DEPTH) -> None:
    """Reject pathologically deep JSON-like inputs without recursive traversal."""
    if max_depth < 1:
        raise ValueError("max_depth must be positive")

    # Track only containers on the active traversal path. Reusing the same
    # container in two branches is JSON-serializable; revisiting an active
    # ancestor is an actual cycle.
    stack: list[tuple[object, int, bool]] = [(value, 0, False)]
    active_containers: set[int] = set()

    while stack:
        current, depth, leaving = stack.pop()
        is_mapping = isinstance(current, Mapping)
        is_sequence = isinstance(current, (list, tuple))
        if not (is_mapping or is_sequence):
            continue

        identity = id(current)
        if leaving:
            active_containers.remove(identity)
            continue

        if depth > max_depth:
            raise ValueError(f"JSON nesting exceeds {max_depth} levels")
        if identity in active_containers:
            raise ValueError("cyclic JSON container")

        active_containers.add(identity)
        stack.append((current, depth, True))
        children = current.values() if is_mapping else current
        stack.extend((item, depth + 1, False) for item in children)
