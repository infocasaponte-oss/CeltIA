import asyncio
from celtia.decision.llm_scorer import AsyncLLMDecisionScorer
from celtia.decision.schema import DecisionQuestion, DecisionType

def test_llm_scorer_preserves_candidate_order():
    async def chat(messages): return '{"scores":{"fast":1.0,"think":3.0}}'
    q=DecisionQuestion("route","route",DecisionType.CHOICE,("fast","think"))
    assert asyncio.run(AsyncLLMDecisionScorer(chat).score({},q,q.candidates())) == [1.0,3.0]


def test_llm_scorer_rejects_extra_envelope_fields():
    async def chat(messages):
        return '{"scores":{"fast":1.0,"think":2.0},"reasoning":"hidden"}'
    q=DecisionQuestion("route","route",DecisionType.CHOICE,("fast","think"))
    try:
        asyncio.run(AsyncLLMDecisionScorer(chat).score({},q,q.candidates()))
        assert False
    except ValueError:
        pass

def test_llm_scorer_rejects_non_finite_scores():
    async def chat(messages):
        return '{"scores":{"fast":1e999,"think":2.0}}'
    q=DecisionQuestion("route","route",DecisionType.CHOICE,("fast","think"))
    try:
        asyncio.run(AsyncLLMDecisionScorer(chat).score({},q,q.candidates()))
        assert False
    except ValueError:
        pass

def test_llm_scorer_rejects_boolean_scores():
    async def chat(messages):
        return '{"scores":{"fast":true,"think":2.0}}'
    q=DecisionQuestion("route","route",DecisionType.CHOICE,("fast","think"))
    try:
        asyncio.run(AsyncLLMDecisionScorer(chat).score({},q,q.candidates()))
        assert False
    except ValueError:
        pass
