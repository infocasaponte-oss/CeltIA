from celtia.decision.prefix_eval import (
    common_prefix_chars,
    evaluate_scorer_prefix_reuse,
    prefix_reuse_report,
)
from celtia.decision.schema import DecisionQuestion, DecisionType


def test_common_prefix_chars_exact():
    assert common_prefix_chars(("abcdef", "abcxyz", "abc")) == 3


def test_prefix_reuse_report_arithmetic():
    report = prefix_reuse_report(("abcdef", "abcxyz", "abc123"))
    assert report["prompts"] == 3
    assert report["total_chars"] == 18
    assert report["shared_prefix_chars"] == 3
    assert report["reusable_chars"] == 6
    assert report["unique_chars"] == 12
    assert report["reuse_fraction"] == 6 / 18


def test_prefix_reuse_requires_multiple_nonempty_prompts():
    for prompts in (("one",), ("one", "")):
        try:
            prefix_reuse_report(prompts)
            assert False
        except ValueError:
            pass


def test_real_scorer_prefix_reuse_increases_with_shared_context():
    questions = (
        DecisionQuestion("q1", "Choose route one", DecisionType.CHOICE, ("fast", "think")),
        DecisionQuestion("q2", "Choose route two", DecisionType.CHOICE, ("fast", "think")),
        DecisionQuestion("q3", "Choose route three", DecisionType.CHOICE, ("fast", "think")),
        DecisionQuestion("q4", "Choose route four", DecisionType.CHOICE, ("fast", "think")),
    )
    small = evaluate_scorer_prefix_reuse({"text": "x" * 100}, questions)
    large = evaluate_scorer_prefix_reuse({"text": "x" * 5000}, questions)

    assert small["shared_prefix_chars"] > 0
    assert large["shared_prefix_chars"] > small["shared_prefix_chars"]
    assert large["reuse_fraction"] > small["reuse_fraction"]
    assert large["context_serialized_chars"] > small["context_serialized_chars"]


def test_real_scorer_prefix_reuse_requires_multiple_questions():
    question = DecisionQuestion("q", "safe?", DecisionType.BOOLEAN)
    try:
        evaluate_scorer_prefix_reuse({}, (question,))
        assert False
    except ValueError:
        pass
