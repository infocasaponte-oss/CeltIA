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
            if len(self.options) > 64: raise ValueError("choice questions support at most 64 options")
            if len(set(self.options)) != len(self.options): raise ValueError("choice options must be unique")
            return self.options
        if self.minimum is None or self.maximum is None or self.minimum > self.maximum:
            raise ValueError("score questions require a valid minimum/maximum")
        if self.maximum - self.minimum + 1 > 101:
            raise ValueError("score questions support at most 101 candidate values")
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
