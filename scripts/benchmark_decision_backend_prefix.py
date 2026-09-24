#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import time

from celtia.decision.backend_prefix_eval import (
    backend_prefix_round_report,
    round_order_sequence,
)
from celtia.decision.llm_scorer import AsyncLLMDecisionScorer
from celtia.decision.prefix_eval import evaluate_scorer_prefix_reuse
from celtia.decision.schema import DecisionQuestion, DecisionType
from core.config import settings
from core.inference import VLLMClient


ROUTES = ("fast", "think", "code", "agent", "long")


def shared_context(size: int) -> dict:
    return {"shared": "S" * size}


def control_context(size: int, round_index: int, question_index: int) -> dict:
    """Return a same-size context that differs near the start for every sample."""
    prefix = f"control:{round_index:04d}:{question_index:04d}:"
    fill = chr(65 + ((round_index + question_index) % 26))
    return {"shared": (prefix + fill * size)[:size]}


def questions_for(count: int, *, round_index: int = 0) -> tuple[DecisionQuestion, ...]:
    """Keep the shared context stable while varying post-context prompt text by round."""
    return tuple(
        DecisionQuestion(
            id=f"r{round_index}-q{index}",
            prompt=f"Synthetic routing round {round_index} question {index}: choose one route.",
            type=DecisionType.CHOICE,
            options=ROUTES,
        )
        for index in range(count)
    )


async def score_once(
    llm: VLLMClient,
    context: object,
    question: DecisionQuestion,
    *,
    max_output_tokens: int,
    timeout_seconds: float,
) -> dict:
    sample: dict = {}

    async def chat(messages: list[dict]) -> str:
        started = time.perf_counter()
        response = await asyncio.wait_for(
            llm.chat(
                messages,
                thinking=False,
                max_tokens=max_output_tokens,
                temperature=0.0,
            ),
            timeout=timeout_seconds,
        )
        sample["latency_ms"] = (time.perf_counter() - started) * 1000
        if (response.get("meta") or {}).get("offline_fallback"):
            raise RuntimeError("backend unavailable; refusing to benchmark demo fallback")
        choices = response.get("choices") or []
        if not choices:
            raise RuntimeError("backend returned no choices")
        content = (choices[0].get("message") or {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("backend returned empty content")
        sample["usage"] = response.get("usage") or {}
        return content.strip()

    scorer = AsyncLLMDecisionScorer(chat)
    await scorer.score(context, question, question.candidates())
    return sample


async def run_workload(
    llm: VLLMClient,
    questions: tuple[DecisionQuestion, ...],
    contexts: tuple[object, ...],
    *,
    max_output_tokens: int,
    timeout_seconds: float,
) -> list[dict]:
    samples = []
    for question, context in zip(questions, contexts, strict=True):
        samples.append(
            await score_once(
                llm,
                context,
                question,
                max_output_tokens=max_output_tokens,
                timeout_seconds=timeout_seconds,
            )
        )
    return samples


async def run_round(
    llm: VLLMClient,
    *,
    round_index: int,
    order: str,
    context_chars: int,
    question_count: int,
    max_output_tokens: int,
    timeout_seconds: float,
) -> dict:
    questions = questions_for(question_count, round_index=round_index)
    common = shared_context(context_chars)
    shared_contexts = tuple(common for _ in questions)
    control_contexts = tuple(
        control_context(context_chars, round_index, index)
        for index, _ in enumerate(questions)
    )

    async def shared():
        return await run_workload(
            llm,
            questions,
            shared_contexts,
            max_output_tokens=max_output_tokens,
            timeout_seconds=timeout_seconds,
        )

    async def control():
        return await run_workload(
            llm,
            questions,
            control_contexts,
            max_output_tokens=max_output_tokens,
            timeout_seconds=timeout_seconds,
        )

    if order == "shared-first":
        shared_samples = await shared()
        control_samples = await control()
    else:
        control_samples = await control()
        shared_samples = await shared()

    return {
        "round": round_index + 1,
        "order": order,
        "shared_samples": shared_samples,
        "control_samples": control_samples,
    }


async def run(args) -> dict:
    llm = VLLMClient(args.base_url, args.model)
    orders = round_order_sequence(args.rounds, first=args.first_order)
    rounds = []
    for round_index, order in enumerate(orders):
        rounds.append(
            await run_round(
                llm,
                round_index=round_index,
                order=order,
                context_chars=args.context_chars,
                question_count=args.questions,
                max_output_tokens=args.max_output_tokens,
                timeout_seconds=args.timeout_seconds,
            )
        )

    structural_questions = questions_for(args.questions)
    common = shared_context(args.context_chars)
    return {
        "kind": "backend_prefix_cache_benchmark",
        "backend": {"base_url": args.base_url, "model": args.model},
        "workload": {
            "context_chars": args.context_chars,
            "questions_per_round": args.questions,
            "rounds": args.rounds,
            "max_output_tokens": args.max_output_tokens,
            "round_orders": list(orders),
        },
        "structural_shared_prefix": evaluate_scorer_prefix_reuse(common, structural_questions),
        "backend_report": backend_prefix_round_report(rounds),
        "rounds": rounds,
        "note": (
            "Latency evidence is backend/environment specific. Each workload's first call is treated as warm-up, "
            "round order alternates, control prefixes are unique per sample, and post-context question text changes "
            "between rounds. Cached-token metadata is reported only when the backend exposes it."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark real-backend latency for repeated CDE shared prefixes against same-size controls."
    )
    parser.add_argument("--base-url", default=settings.vllm_base_url)
    parser.add_argument("--model", default=settings.model_serve_name)
    parser.add_argument("--context-chars", type=int, default=12000)
    parser.add_argument("--questions", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--first-order", choices=("shared-first", "control-first"), default="control-first")
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--min-latency-reduction",
        type=float,
        help="Optional manual gate. Exit 2 unless aggregate median latency reduction reaches this fraction.",
    )
    args = parser.parse_args()

    if not 0 <= args.context_chars <= 50000:
        raise ValueError("context-chars must be between 0 and 50000")
    if not 2 <= args.questions <= 32:
        raise ValueError("questions must be between 2 and 32")
    if not 1 <= args.rounds <= 20:
        raise ValueError("rounds must be between 1 and 20")
    if not 64 <= args.max_output_tokens <= 2048:
        raise ValueError("max-output-tokens must be between 64 and 2048")
    if not 0 < args.timeout_seconds <= 1800:
        raise ValueError("timeout-seconds must be greater than 0 and at most 1800")
    if args.min_latency_reduction is not None and not -1 <= args.min_latency_reduction <= 1:
        raise ValueError("min-latency-reduction must be between -1 and 1")

    report = asyncio.run(run(args))
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.min_latency_reduction is not None:
        observed = report["backend_report"]["median_latency_reduction"]
        if observed is None or observed < args.min_latency_reduction:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
