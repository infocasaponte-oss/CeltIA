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
    assert result.abstention_reason == "suspected_ood"
    assert result.suspected_ood is True
    assert result.normalized_entropy == 1.0
    assert result.margin == 0.0


def test_sync_engine_can_disable_ood_rejection():
    q = DecisionQuestion("safe", "Safe?", DecisionType.BOOLEAN)
    result = DecisionEngine(
        FixedScorer([0.0, 0.0]),
        abstain_below=0.0,
        reject_suspected_ood=False,
    ).decide(DecisionRequest({}, (q,)))[0]
    assert not result.abstained
    assert result.decision == "false"
    assert result.suspected_ood is True


def test_score_decision_reports_expected_value():
    q = DecisionQuestion("quality", "Quality", DecisionType.SCORE, minimum=1, maximum=3)
    result = DecisionEngine(
        FixedScorer([0.0, 0.0, 2.0]),
        abstain_below=0.0,
        reject_suspected_ood=False,
    ).decide(DecisionRequest({}, (q,)))[0]
    expected = sum(float(label) * probability for label, probability in result.probabilities.items())
    assert math.isclose(result.expected_score, expected)
    assert result.expected_score is not None


def test_non_score_decision_has_no_expected_value():
    q = DecisionQuestion("safe", "Safe?", DecisionType.BOOLEAN)
    result = DecisionEngine(
        FixedScorer([0.0, 2.0]),
        abstain_below=0.0,
        reject_suspected_ood=False,
    ).decide(DecisionRequest({}, (q,)))[0]
    assert result.expected_score is None


def test_boolean_rejects_choice_or_score_fields():
    for q in (
        DecisionQuestion("b", "bool", DecisionType.BOOLEAN, ("x", "y")),
        DecisionQuestion("b", "bool", DecisionType.BOOLEAN, minimum=0, maximum=1),
    ):
        try:
            q.candidates()
            assert False
        except ValueError:
            pass


def test_choice_rejects_score_bounds():
    q = DecisionQuestion(
        "route", "route", DecisionType.CHOICE, ("fast", "think"), minimum=0, maximum=1
    )
    try:
        q.candidates()
        assert False
    except ValueError:
        pass


def test_score_rejects_choice_options():
    q = DecisionQuestion(
        "score", "score", DecisionType.SCORE, ("low", "high"), minimum=1, maximum=5
    )
    try:
        q.candidates()
        assert False
    except ValueError:
        pass


def test_score_requires_at_least_two_candidate_values():
    q = DecisionQuestion("score", "Score", DecisionType.SCORE, minimum=5, maximum=5)
    try:
        q.candidates()
        assert False
    except ValueError as exc:
        assert "at least two ordered values" in str(exc)


def test_question_rejects_blank_or_wrongly_typed_text_fields():
    invalid = (
        DecisionQuestion("   ", "prompt", DecisionType.BOOLEAN),
        DecisionQuestion("id", "   ", DecisionType.BOOLEAN),
        DecisionQuestion(123, "prompt", DecisionType.BOOLEAN),
        DecisionQuestion("id", 123, DecisionType.BOOLEAN),
        DecisionQuestion("id", "prompt", "boolean"),
        DecisionQuestion("id", "prompt", DecisionType.CHOICE, ("ok", "   ")),
        DecisionQuestion("id", "prompt", DecisionType.CHOICE, ("ok", 123)),
    )
    for question in invalid:
        try:
            question.candidates()
            assert False, question
        except ValueError:
            pass


def test_score_bounds_reject_booleans_and_non_integers():
    invalid = (
        DecisionQuestion("score", "score", DecisionType.SCORE, minimum=True, maximum=5),
        DecisionQuestion("score", "score", DecisionType.SCORE, minimum=1, maximum=False),
        DecisionQuestion("score", "score", DecisionType.SCORE, minimum=1.0, maximum=5),
        DecisionQuestion("score", "score", DecisionType.SCORE, minimum=1, maximum=5.0),
    )
    for question in invalid:
        try:
            question.candidates()
            assert False, question
        except ValueError as exc:
            assert "score bounds must be integers" in str(exc)
