#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import time

from celtia.decision.backend_prefix_eval import backend_prefix_report
from celtia.decision.llm_scorer import AsyncLLMDecisionScorer
from celtia.decision.prefix_eval import evaluate_scorer_prefix_reuse
from celtia.decision.schema import DecisionQuestion, DecisionType
from core.config import settings
from core.inference import VLLMClient


ROUTES = ("fast", "think", "code", "agent", "long")


def shared_context(size: int) -> dict:
    return {"shared": "S" * size}


def control_context(size: int, index: int) -> dict:
    prefix = f"{index:04d}:"
    fill = chr(65 + (index % 26))
    return {"shared": (prefix + fill * size)[:size]}


def questions_for(count: int) -> tuple[DecisionQuestion, ...]:
    return tuple(
        DecisionQuestion(
            id=f"q{index}",
            prompt=f"Synthetic routing question {index}: choose one route.",
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


async def run(args) -> dict:
    questions = questions_for(args.questions)
    common = shared_context(args.context_chars)
    shared_contexts = tuple(common for _ in questions)
    control_contexts = tuple(
        control_context(args.context_chars, index)
        for index, _ in enumerate(questions)
    )
    llm = VLLMClient(args.base_url, args.model)

    async def shared():
        return await run_workload(
            llm,
            questions,
            shared_contexts,
            max_output_tokens=args.max_output_tokens,
            timeout_seconds=args.timeout_seconds,
        )

    async def control():
        return await run_workload(
            llm,
            questions,
            control_contexts,
            max_output_tokens=args.max_output_tokens,
            timeout_seconds=args.timeout_seconds,
        )

    if args.order == "shared-first":
        shared_samples = await shared()
        control_samples = await control()
    else:
        control_samples = await control()
        shared_samples = await shared()

    return {
        "kind": "backend_prefix_cache_benchmark",
        "backend": {"base_url": args.base_url, "model": args.model},
        "workload": {
            "context_chars": args.context_chars,
            "questions": args.questions,
            "max_output_tokens": args.max_output_tokens,
            "order": args.order,
        },
        "structural_shared_prefix": evaluate_scorer_prefix_reuse(common, questions),
        "backend_report": backend_prefix_report(shared_samples, control_samples),
        "shared_samples": shared_samples,
        "control_samples": control_samples,
        "note": (
            "Latency evidence is backend/environment specific. Logical prompt token counts may stay constant "
            "even when a backend reuses KV state; cached-token metadata is reported only when the backend exposes it."
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
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--order", choices=("shared-first", "control-first"), default="control-first")
    parser.add_argument(
        "--min-latency-reduction",
        type=float,
        help="Optional manual gate. Exit 2 unless median shared-prefix latency reduction reaches this fraction.",
    )
    args = parser.parse_args()

    if not 0 <= args.context_chars <= 50000:
        raise ValueError("context-chars must be between 0 and 50000")
    if not 2 <= args.questions <= 32:
        raise ValueError("questions must be between 2 and 32")
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
