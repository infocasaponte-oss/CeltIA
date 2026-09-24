#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from celtia.decision.prefix_eval import evaluate_scorer_prefix_reuse
from celtia.decision.schema import DecisionQuestion, DecisionType


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure exact shared-prefix reuse in rendered CDE scorer prompts."
    )
    parser.add_argument("--context-chars", type=int, default=12000)
    parser.add_argument("--questions", type=int, default=8)
    parser.add_argument(
        "--min-reuse-fraction",
        type=float,
        help="Optional structural gate; exit 2 when the measured fraction is lower.",
    )
    args = parser.parse_args()

    if not 0 <= args.context_chars <= 50000:
        raise ValueError("context-chars must be between 0 and 50000")
    if not 2 <= args.questions <= 32:
        raise ValueError("questions must be between 2 and 32")
    if args.min_reuse_fraction is not None and not 0 <= args.min_reuse_fraction <= 1:
        raise ValueError("min-reuse-fraction must be between 0 and 1")

    context = {"shared": "x" * args.context_chars}
    options = ("fast", "think", "code", "agent", "long")
    questions = tuple(
        DecisionQuestion(
            id=f"q{index}",
            prompt=f"Synthetic routing question {index}",
            type=DecisionType.CHOICE,
            options=options,
        )
        for index in range(args.questions)
    )
    report = evaluate_scorer_prefix_reuse(context, questions)
    output = {
        "kind": "character_prefix_pre_tokenization_proxy",
        "context_chars": args.context_chars,
        "questions": args.questions,
        "report": report,
        "note": "This measures structural reuse only; it does not claim token, latency, or backend cache savings.",
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))

    if (
        args.min_reuse_fraction is not None
        and report["reuse_fraction"] < args.min_reuse_fraction
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
