import asyncio

from core.decision_routes import (
    ROUTE_DECISION_PROMPT,
    route_decision_context,
    route_decision_question,
)
from scripts import collect_decision_eval_results as collector


def test_route_contract_defines_all_execution_modes():
    question=route_decision_question()
    assert question["options"] == ["fast","think","code","agent","long"]
    for route in question["options"]:
        assert f"- {route}:" in ROUTE_DECISION_PROMPT
    assert "input_chars" in ROUTE_DECISION_PROMPT
    assert "legacy-router" in ROUTE_DECISION_PROMPT


def test_route_context_exposes_size_without_heuristic_label():
    context=route_decision_context("hello",input_chars=20000)
    assert context["user_message"] == "hello"
    assert context["input_chars"] == 20000
    assert context["long_context_chars"] < context["input_chars"]
    assert "heuristic_route" not in context


def test_route_context_rejects_impossible_input_size():
    try:
        route_decision_context("hello",input_chars=3)
        assert False
    except ValueError as exc:
        assert "input_chars" in str(exc)


class CaptureRuntime:
    def __init__(self):
        self.context=None
        self.questions=None

    async def decide_with_usage(self,context,questions):
        self.context=context
        self.questions=questions
        result=type(
            "Result",
            (),
            {
                "decision":"long",
                "confidence":.9,
                "abstained":False,
                "suspected_ood":False,
                "abstention_reason":None,
                "normalized_entropy":.1,
                "margin":.8,
            },
        )()
        return [result],{"models":["fake-model"]}


def test_collector_uses_benchmark_input_size_without_baseline_leak(monkeypatch):
    runtime=CaptureRuntime()
    monkeypatch.setattr(collector,"route",lambda text:type("R",(),{"mode":"fast"})())
    row={"text":"short visible task","expected":"long","ood":False,"input_chars":20000}
    item=asyncio.run(collector.collect_one(runtime,row))
    assert item["heuristic"] == "fast"
    assert runtime.context["input_chars"] == 20000
    assert runtime.context["user_message"] == row["text"]
    assert "heuristic_route" not in runtime.context
    assert runtime.questions[0]["options"] == ["fast","think","code","agent","long"]
