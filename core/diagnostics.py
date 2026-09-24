# Copyright (c) 2026 Luis Manuel Cousido Hermida. All rights reserved.
from collections import deque
from datetime import datetime, timezone

_errors = deque(maxlen=50)


def record_error(context: str, detail: str):
    _errors.append({
        "context": context,
        "detail": detail[:500],
        "at": datetime.now(timezone.utc).isoformat(),
    })


def recent_errors(limit: int = 10):
    return list(_errors)[-limit:][::-1]
