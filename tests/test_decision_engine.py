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


def test_choice_rejects_duplicate_and_excessive_options():
    duplicate = DecisionQuestion("route", "Choose route", DecisionType.CHOICE, ("fast", "fast"))
    try:
        duplicate.candidates()
        assert False
    except ValueError:
        pass
    excessive = DecisionQuestion("route", "Choose route", DecisionType.CHOICE, tuple(str(i) for i in range(65)))
    try:
        excessive.candidates()
        assert False
    except ValueError:
        pass

def test_score_candidate_range_is_bounded():
    q = DecisionQuestion("score", "Score", DecisionType.SCORE, minimum=0, maximum=101)
    try:
        q.candidates()
        assert False
    except ValueError:
        pass

def test_question_text_limits():
    q = DecisionQuestion("x", "p" * 8001, DecisionType.BOOLEAN)
    try:
        q.candidates()
        assert False
    except ValueError:
        pass


def test_sync_engine_abstains_on_ambiguous_distribution_via_ood():
    q = DecisionQuestion("safe", "Safe?", DecisionType.BOOLEAN)
    result = DecisionEngine(
        FixedScorer([0.0, 0.0]),
        abstain_below=0.0,
        reject_suspected_ood=True,
    ).decide(DecisionRequest({}, (q,)))[0]
    assert result.abstained
    assert result.decision is None


def test_sync_engine_can_disable_ood_rejection():
    q = DecisionQuestion("safe", "Safe?", DecisionType.BOOLEAN)
    result = DecisionEngine(
        FixedScorer([0.0, 0.0]),
        abstain_below=0.0,
        reject_suspected_ood=False,
    ).decide(DecisionRequest({}, (q,)))[0]
    assert not result.abstained
    assert result.decision == "false"
