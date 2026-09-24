import asyncio
from celtia.decision.llm_scorer import AsyncLLMDecisionScorer
from celtia.decision.schema import DecisionQuestion, DecisionType

def test_llm_scorer_preserves_candidate_order():
    async def chat(messages): return '{"scores":{"fast":1.0,"think":3.0}}'
    q=DecisionQuestion("route","route",DecisionType.CHOICE,("fast","think"))
    assert asyncio.run(AsyncLLMDecisionScorer(chat).score({},q,q.candidates())) == [1.0,3.0]
