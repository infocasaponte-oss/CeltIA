import asyncio
from celtia.decision.async_engine import AsyncDecisionEngine
from celtia.decision.schema import DecisionQuestion, DecisionRequest, DecisionType

class Scorer:
    async def score(self, context, question, candidates): return [0.0, 2.0]

def test_async_engine():
    q=DecisionQuestion("safe","safe?",DecisionType.BOOLEAN)
    result=asyncio.run(AsyncDecisionEngine(Scorer()).decide(DecisionRequest({},(q,))))[0]
    assert result.decision == "true"

class AmbiguousScorer:
    async def score(self, context, question, candidates): return [0.0, 0.0]

def test_async_engine_abstains_on_ambiguous_distribution():
    q=DecisionQuestion("safe","safe?",DecisionType.BOOLEAN)
    result=asyncio.run(AsyncDecisionEngine(AmbiguousScorer(),abstain_below=0.0).decide(DecisionRequest({},(q,))))[0]
    assert result.abstained
    assert result.decision is None
