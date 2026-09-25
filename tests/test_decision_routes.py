import asyncio

from core.decision_routes import combine_route_results, route_decision_context, route_decision_question, route_ood_question
from celtia.decision.llm_scorer import AsyncLLMDecisionScorer
from celtia.decision.route_contract import ROUTES, ROUTE_OOD_SYSTEM_GUIDANCE, ROUTE_SYSTEM_GUIDANCE
from celtia.decision.schema import DecisionQuestion, DecisionType
from scripts import collect_decision_eval_results as collector


def test_route_contract_defines_all_execution_modes():
    question=route_decision_question()
    assert question["options"] == list(ROUTES)
    for route in ROUTES:
        assert f"- {route}:" in ROUTE_SYSTEM_GUIDANCE
    assert "input_chars" in ROUTE_SYSTEM_GUIDANCE
    assert "legacy-router" in ROUTE_SYSTEM_GUIDANCE


def test_route_contract_is_in_trusted_system_prompt_only():
    seen={}
    async def chat(messages):
        seen["messages"]=messages
        return '{"scores":{"c0":0,"c1":1,"c2":0,"c3":0,"c4":0}}'
    q=DecisionQuestion("route","Select the most appropriate CeltIA execution route.",DecisionType.CHOICE,ROUTES)
    asyncio.run(AsyncLLMDecisionScorer(chat).score({"user_message":"compare options"},q,q.candidates()))
    system=seen["messages"][0]["content"]
    user=seen["messages"][1]["content"]
    assert ROUTE_SYSTEM_GUIDANCE in system
    assert ROUTE_SYSTEM_GUIDANCE not in user


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
        route_result=type(
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
        ood_result=type("OODResult",(),{"probabilities":{"false":.9,"true":.1}})()
        return [route_result,ood_result],{"models":["fake-model"]}


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



def test_ood_contract_is_in_trusted_system_prompt_only():
    seen={}
    async def chat(messages):
        seen["messages"]=messages
        return '{"scores":{"c0":1,"c1":0}}'
    q=DecisionQuestion("route_ood","Is this input outside the CeltIA routing domain?",DecisionType.BOOLEAN)
    asyncio.run(AsyncLLMDecisionScorer(chat).score({"user_message":"hello"},q,q.candidates()))
    system=seen["messages"][0]["content"]
    user=seen["messages"][1]["content"]
    assert ROUTE_OOD_SYSTEM_GUIDANCE in system
    assert ROUTE_OOD_SYSTEM_GUIDANCE not in user


def test_semantic_ood_forces_abstention_even_for_confident_route():
    route_result=type(
        "RouteResult",
        (),
        {
            "decision":"fast",
            "confidence":.99,
            "abstained":False,
            "suspected_ood":False,
            "abstention_reason":None,
            "normalized_entropy":.01,
            "margin":.98,
        },
    )()
    ood_result=type("OODResult",(),{"probabilities":{"false":.1,"true":.9}})()
    combined=combine_route_results(route_result,ood_result)
    assert combined["decision"] is None
    assert combined["abstained"] is True
    assert combined["suspected_ood"] is True
    assert combined["abstention_reason"] == "semantic_ood"
    assert combined["ood_probability"] == .9


def test_route_ood_question_is_boolean():
    assert route_ood_question() == {
        "id":"route_ood",
        "prompt":"Is this input outside the CeltIA routing domain?",
        "type":"boolean",
    }
