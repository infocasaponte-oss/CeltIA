import math
from celtia.decision import DecisionEngine, DecisionQuestion, DecisionRequest, DecisionType

class FixedScorer:
    def __init__(self, logits): self.logits = logits
    def score(self, context, question, candidates): return self.logits

def test_choice_selects_highest_probability():
    q = DecisionQuestion("route", "Choose route", DecisionType.CHOICE, ("fast", "think", "agent"))
    result = DecisionEngine(FixedScorer([0.0, 3.0, 1.0]), abstain_below=0.5).decide(DecisionRequest({}, (q,)))[0]
    assert result.decision == "think" and not result.abstained
    assert math.isclose(sum(result.probabilities.values()), 1.0)

def test_low_confidence_abstains():
    q = DecisionQuestion("safe", "Safe?", DecisionType.BOOLEAN)
    result = DecisionEngine(FixedScorer([0.0, 0.0]), abstain_below=0.6).decide(DecisionRequest({}, (q,)))[0]
    assert result.decision is None and result.abstained

def test_score_candidates_are_generated():
    q = DecisionQuestion("quality", "Quality", DecisionType.SCORE, minimum=1, maximum=5)
    assert q.candidates() == ("1", "2", "3", "4", "5")
