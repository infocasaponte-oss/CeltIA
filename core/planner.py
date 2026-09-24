# Copyright (c) 2026 Luis Manuel Cousido Hermida. All rights reserved.
import re


class Planner:
    """Deterministic planner used before tool execution and verification."""

    def plan(self, text: str) -> dict:
        lower = text.lower()
        requires_tool = bool(
            re.search(
                r"\b(calculate|calculator|compute|evaluate|python|debug|search|read|tool|verify|check)\b",
                lower,
            )
        )
        verification_required = bool(
            re.search(
                r"\b(verify|check|validate|confirm|prove|ensure|test)\b",
                lower,
            )
        )
        return {
            "requires_tool": requires_tool,
            "verification_required": verification_required,
            "steps": [
                "plan",
                "tool",
                "verify",
                "answer",
            ],
            "reason": "Use tools when needed and verify results before finalizing the answer.",
        }
