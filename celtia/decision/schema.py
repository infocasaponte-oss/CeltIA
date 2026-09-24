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
    MAX_ID_CHARS = 128
    MAX_PROMPT_CHARS = 8000
    MAX_OPTION_CHARS = 1000

    id: str
    prompt: str
    type: DecisionType
    options: tuple[str, ...] = ()
    minimum: int | None = None
    maximum: int | None = None

    def candidates(self) -> tuple[str, ...]:
        if not isinstance(self.id, str) or not self.id.strip() or len(self.id) > self.MAX_ID_CHARS:
            raise ValueError("question id must contain between 1 and 128 non-blank characters")
        if not isinstance(self.prompt, str) or not self.prompt.strip() or len(self.prompt) > self.MAX_PROMPT_CHARS:
            raise ValueError("question prompt must contain between 1 and 8000 non-blank characters")
        if not isinstance(self.type, DecisionType):
            raise ValueError("question type must be a DecisionType")
        if self.type is DecisionType.BOOLEAN:
            if self.options or self.minimum is not None or self.maximum is not None:
                raise ValueError("boolean questions do not accept options or score bounds")
            return ("false", "true")
        if self.type is DecisionType.CHOICE:
            if self.minimum is not None or self.maximum is not None:
                raise ValueError("choice questions do not accept score bounds")
            if len(self.options) < 2: raise ValueError("choice questions require at least two options")
            if len(self.options) > 64: raise ValueError("choice questions support at most 64 options")
            if any(not isinstance(option, str) for option in self.options):
                raise ValueError("choice options must be strings")
            if len(set(self.options)) != len(self.options): raise ValueError("choice options must be unique")
            if any(not option.strip() or len(option) > self.MAX_OPTION_CHARS for option in self.options):
                raise ValueError("choice options must contain between 1 and 1000 non-blank characters")
            return self.options
        if self.options:
            raise ValueError("score questions do not accept choice options")
        if (
            isinstance(self.minimum, bool)
            or isinstance(self.maximum, bool)
            or not isinstance(self.minimum, int)
            or not isinstance(self.maximum, int)
        ):
            raise ValueError("score bounds must be integers")
        if self.minimum >= self.maximum:
            raise ValueError("score questions require at least two ordered values")
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
    abstention_reason: str | None = None
    suspected_ood: bool = False
    normalized_entropy: float | None = None
    margin: float | None = None
    expected_score: float | None = None
