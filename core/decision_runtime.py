# Copyright (c) 2026 Luis Manuel Cousido Hermida. All rights reserved.
from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping

from celtia.decision.async_engine import AsyncDecisionEngine
from celtia.decision.json_safety import validate_json_depth
from celtia.decision.llm_scorer import AsyncLLMDecisionScorer
from celtia.decision.policy import validate_policy_settings
from celtia.decision.schema import DecisionQuestion, DecisionRequest, DecisionType


class CeltIADecisionRuntime:
    MAX_CONTEXT_CHARS = 50000
    HARD_MAX_QUESTIONS = 32

    """Bridge between the independent CDE core and CeltIA's existing local LLM client."""

    def __init__(
        self,
        llm,
        *,
        abstain_below: float = 0.55,
        temperature: float = 1.0,
        reject_suspected_ood: bool = True,
        ood_entropy_threshold: float = 0.90,
        ood_margin_threshold: float = 0.10,
        max_questions: int = 32,
        max_output_tokens: int = 1024,
        max_total_output_tokens: int = 8192,
        max_total_prompt_chars: int = 250000,
        call_timeout_seconds: float = 30.0,
        request_timeout_seconds: float = 120.0,
    ):
        policy_settings = validate_policy_settings(
            abstain_below=abstain_below,
            temperature=temperature,
            reject_suspected_ood=reject_suspected_ood,
            ood_entropy_threshold=ood_entropy_threshold,
            ood_margin_threshold=ood_margin_threshold,
        )
        abstain_below = policy_settings["abstain_below"]
        temperature = policy_settings["temperature"]
        reject_suspected_ood = policy_settings["reject_suspected_ood"]
        ood_entropy_threshold = policy_settings["ood_entropy_threshold"]
        ood_margin_threshold = policy_settings["ood_margin_threshold"]
        if not 1 <= max_questions <= self.HARD_MAX_QUESTIONS:
            raise ValueError("max_questions must be between 1 and 32")
        if not 64 <= max_output_tokens <= 2048:
            raise ValueError("max_output_tokens must be between 64 and 2048")
        if not 64 <= max_total_output_tokens <= 65536:
            raise ValueError("max_total_output_tokens must be between 64 and 65536")
        if max_total_output_tokens < max_questions * 64:
            raise ValueError("max_total_output_tokens is too small for max_questions")
        if not 10000 <= max_total_prompt_chars <= 2000000:
            raise ValueError("max_total_prompt_chars must be between 10000 and 2000000")
        if not 0 < call_timeout_seconds <= 300:
            raise ValueError("call_timeout_seconds must be greater than 0 and at most 300")
        if not 0 < request_timeout_seconds <= 1800:
            raise ValueError("request_timeout_seconds must be greater than 0 and at most 1800")
        self.llm = llm
        self.max_questions = max_questions
        self.max_output_tokens = max_output_tokens
        self.max_total_output_tokens = max_total_output_tokens
        self.max_total_prompt_chars = max_total_prompt_chars
        self.call_timeout_seconds = call_timeout_seconds
        self.request_timeout_seconds = request_timeout_seconds
        self.engine_options = {
            "abstain_below": abstain_below,
            "temperature": temperature,
            "reject_suspected_ood": reject_suspected_ood,
            "ood_entropy_threshold": ood_entropy_threshold,
            "ood_margin_threshold": ood_margin_threshold,
        }

    def _request(self, context, questions) -> DecisionRequest:
        try:
            validate_json_depth(context)
            serialized_context = json.dumps(context, ensure_ascii=False)
        except (TypeError, ValueError, OverflowError, RecursionError) as exc:
            raise ValueError("context must be JSON-serializable") from exc
        if len(serialized_context) > self.MAX_CONTEXT_CHARS:
            raise ValueError("decision context exceeds 50000 serialized characters")
        try:
            question_items = tuple(questions)
        except TypeError as exc:
            raise ValueError("questions must be an iterable of mappings") from exc
        if not question_items or len(question_items) > self.max_questions:
            raise ValueError(f"questions must contain between 1 and {self.max_questions} items")
        if not all(isinstance(q, Mapping) for q in question_items):
            raise ValueError("questions must contain mappings")
        required_fields = {"id", "prompt", "type"}
        if any(not required_fields.issubset(q) for q in question_items):
            raise ValueError("each question requires id, prompt, and type")
        ids = [q["id"] for q in question_items]
        if len(ids) != len(set(ids)):
            raise ValueError("question ids must be unique")
        return DecisionRequest(
            context=context,
            questions=tuple(
                DecisionQuestion(
                    id=q["id"],
                    prompt=q["prompt"],
                    type=DecisionType(q["type"]),
                    options=tuple(q.get("options") or ()),
                    minimum=q.get("minimum"),
                    maximum=q.get("maximum"),
                )
                for q in question_items
            ),
        )

    async def decide_with_usage(self, context, questions):
        request = self._request(context, questions)
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "models": []}
        minimum_output_tokens=[
            max(64, 16 + 12 * len(question.candidates()))
            for question in request.questions
        ]
        minimum_total_output_tokens=sum(minimum_output_tokens)
        if minimum_total_output_tokens > self.max_total_output_tokens:
            raise ValueError(
                "decision output budget is too small for the requested candidate set"
            )
        remaining_output_tokens=self.max_total_output_tokens-minimum_total_output_tokens
        extra_per_question=remaining_output_tokens // len(request.questions)
        output_budgets=[
            min(self.max_output_tokens, minimum + extra_per_question)
            for minimum in minimum_output_tokens
        ]
        total_prompt_chars = 0
        for question in request.questions:
            messages = AsyncLLMDecisionScorer.messages_for(
                request.context,
                question,
                question.candidates(),
            )
            total_prompt_chars += len(json.dumps(
                messages,
                ensure_ascii=False,
                separators=(",", ":"),
            ))
        if total_prompt_chars > self.max_total_prompt_chars:
            raise ValueError(
                f"decision prompt budget exceeded: {total_prompt_chars} > {self.max_total_prompt_chars} characters"
            )

        output_budget_iter=iter(output_budgets)

        async def chat(messages: list[dict]) -> str:
            per_call_output_tokens=next(output_budget_iter)
            try:
                response = await asyncio.wait_for(
                    self.llm.chat(
                        messages,
                        thinking=False,
                        max_tokens=per_call_output_tokens,
                        temperature=0.0,
                    ),
                    timeout=self.call_timeout_seconds,
                )
            except asyncio.TimeoutError as exc:
                raise RuntimeError("decision backend timed out") from exc
            meta = response.get("meta") or {}
            if meta.get("offline_fallback"):
                raise RuntimeError("decision backend unavailable")
            model=meta.get("model")
            if isinstance(model,str) and model and model not in usage["models"]:
                usage["models"].append(model)
            choices = response.get("choices") or []
            if not choices:
                raise RuntimeError("decision backend returned no choices")
            content = (choices[0].get("message") or {}).get("content")
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError("decision backend returned empty content")

            raw_usage = response.get("usage") or {}

            def token_count(name: str, fallback: int) -> int:
                raw = raw_usage.get(name)
                if raw is None or isinstance(raw, bool):
                    return fallback
                try:
                    value = int(raw)
                except (TypeError, ValueError, OverflowError):
                    return fallback
                return value if value >= 0 else fallback

            prompt_tokens = token_count(
                "prompt_tokens",
                max(1, len(json.dumps(messages, ensure_ascii=False)) // 3),
            )
            completion_tokens = token_count(
                "completion_tokens",
                max(1, len(content) // 3),
            )
            total_tokens = prompt_tokens + completion_tokens
            usage["prompt_tokens"] += prompt_tokens
            usage["completion_tokens"] += completion_tokens
            usage["total_tokens"] += total_tokens
            return content.strip()

        engine = AsyncDecisionEngine(AsyncLLMDecisionScorer(chat), **self.engine_options)
        try:
            results = await asyncio.wait_for(
                engine.decide(request),
                timeout=self.request_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise RuntimeError("decision request timed out") from exc
        return results, usage

    async def decide(self, context, questions):
        results, _ = await self.decide_with_usage(context, questions)
        return results
