from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

class DecisionType(str, Enum):
    BOOLEAN = "boolean"
    CHOICE = "choice"
    SCORE = "score"

@dataclass(frozen=True)
class DecisionQuestion:
    id: str
    prompt: str
    type: DecisionType
    options: tuple[str, ...] = ()
    minimum: int | None = None
    maximum: int | None = None

    def candidates(self) -> tuple[str, ...]:
        if self.type is DecisionType.BOOLEAN: return ("false", "true")
        if self.type is DecisionType.CHOICE:
            if len(self.options) < 2: raise ValueError("choice questions require at least two options")
            return self.options
        if self.minimum is None or self.maximum is None or self.minimum > self.maximum:
            raise ValueError("score questions require a valid minimum/maximum")
        return tuple(str(v) for v in range(self.minimum, self.maximum + 1))

@dataclass(frozen=True)
class DecisionRequest:
    context: Any
    questions: tuple[DecisionQuestion, ...]

@dataclass(frozen=True)
class DecisionResult:
    id: str
    probabilities: dict[str, float]
    decision: str | None
    confidence: float
    abstained: bool
