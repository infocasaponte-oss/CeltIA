from celtia.decision.evaluation import ShadowSample, evaluate_shadow, promotion_gate

def test_shadow_metrics_and_gate():
    samples=[
        ShadowSample("fast","fast",.9,False,"fast"),
        ShadowSample("think","think",.8,False,"think"),
        ShadowSample("fast",None,.4,True,"think"),
    ]
    m=evaluate_shadow(samples)
    assert m["samples"] == 3
    assert m["decided"] == 2
    assert m["agreement_rate"] == 1.0
    assert m["cde_selective_accuracy"] == 1.0
    assert m["labeled_coverage"] == 2 / 3
    assert m["labeled_abstentions"] == 1
    gate=promotion_gate(m,min_samples=1,min_coverage=.5,min_labeled=1,min_accuracy_delta=0,min_labeled_coverage=.5)
    assert gate["eligible"]

def test_gate_blocks_small_dataset():
    g=promotion_gate({"samples":2,"coverage":1.0,"labeled":2,"heuristic_accuracy":.5,"cde_accuracy":1.0})
    assert not g["eligible"]
    assert "insufficient_samples" in g["reasons"]


def test_gate_blocks_excessive_labeled_abstention():
    metrics = {
        "samples": 100,
        "coverage": .95,
        "labeled": 100,
        "heuristic_accuracy": .50,
        "cde_accuracy": .70,
        "labeled_coverage": .70,
    }
    gate = promotion_gate(
        metrics,
        min_samples=1,
        min_coverage=.5,
        min_labeled=1,
        min_accuracy_delta=0,
        min_labeled_coverage=.8,
    )
    assert not gate["eligible"]
    assert "insufficient_labeled_coverage" in gate["reasons"]


def test_selective_accuracy_does_not_replace_total_accuracy():
    samples = [
        ShadowSample("fast", "fast", .9, False, "fast"),
        ShadowSample("fast", None, .1, True, "think"),
    ]
    metrics = evaluate_shadow(samples)
    assert metrics["cde_selective_accuracy"] == 1.0
    assert metrics["cde_accuracy"] == .5
    assert metrics["labeled_coverage"] == .5
    gate = promotion_gate(
        metrics,
        min_samples=1,
        min_coverage=.5,
        min_labeled=1,
        min_accuracy_delta=.1,
        min_labeled_coverage=.5,
    )
    assert not gate["eligible"]
    assert "accuracy_delta_below_gate" in gate["reasons"]


def test_shadow_evaluation_reports_ood_confusion_metrics():
    samples = [
        ShadowSample("fast", None, .2, True, None, suspected_ood=True, expected_ood=True),
        ShadowSample("fast", "fast", .9, False, "fast", suspected_ood=False, expected_ood=False),
        ShadowSample("think", None, .3, True, None, suspected_ood=False, expected_ood=True),
        ShadowSample("think", None, .4, True, None, suspected_ood=True, expected_ood=False),
        ShadowSample("agent", None, .0, True, None, suspected_ood=None, expected_ood=True),
    ]
    metrics = evaluate_shadow(samples)
    assert metrics["ood_labeled"] == 5
    assert metrics["ood_evaluated"] == 4
    assert metrics["ood_coverage"] == .8
    assert metrics["ood_true_positive"] == 1
    assert metrics["ood_false_positive"] == 1
    assert metrics["ood_true_negative"] == 1
    assert metrics["ood_false_negative"] == 1
    assert metrics["ood_precision"] == .5
    assert metrics["ood_recall"] == .5
    assert metrics["ood_specificity"] == .5
    assert metrics["ood_false_positive_rate"] == .5
    assert metrics["ood_accuracy"] == .5


def test_shadow_evaluation_keeps_route_and_ood_labels_independent():
    sample = ShadowSample(
        "fast", "fast", .9, False, "fast",
        suspected_ood=False,
        expected_ood=False,
    )
    metrics = evaluate_shadow([sample])
    assert metrics["labeled"] == 1
    assert metrics["cde_accuracy"] == 1.0
    assert metrics["ood_labeled"] == 1
    assert metrics["ood_accuracy"] == 1.0


def test_promotion_gate_can_require_ood_evidence():
    metrics = evaluate_shadow([
        ShadowSample("fast", "fast", .9, False, "fast", suspected_ood=False, expected_ood=False),
        ShadowSample("think", "think", .9, False, "think", suspected_ood=False, expected_ood=False),
        ShadowSample("fast", None, .2, True, None, suspected_ood=True, expected_ood=True),
        ShadowSample("think", None, .2, True, None, suspected_ood=True, expected_ood=True),
    ])
    gate = promotion_gate(
        metrics,
        min_samples=1,
        min_coverage=.5,
        min_labeled=1,
        min_accuracy_delta=0,
        min_labeled_coverage=.5,
        min_ood_labeled=4,
        min_ood_coverage=1.0,
        min_ood_recall=.8,
        max_ood_false_positive_rate=.2,
    )
    assert gate["eligible"]


def test_promotion_gate_blocks_unsafe_ood_behavior():
    metrics = evaluate_shadow([
        ShadowSample("fast", "fast", .9, False, "fast", suspected_ood=True, expected_ood=False),
        ShadowSample("think", "think", .9, False, "think", suspected_ood=False, expected_ood=False),
        ShadowSample("fast", None, .2, True, None, suspected_ood=False, expected_ood=True),
        ShadowSample("think", None, .2, True, None, suspected_ood=True, expected_ood=True),
    ])
    gate = promotion_gate(
        metrics,
        min_samples=1,
        min_coverage=.5,
        min_labeled=1,
        min_accuracy_delta=0,
        min_labeled_coverage=.5,
        min_ood_labeled=4,
        min_ood_coverage=1.0,
        min_ood_recall=.8,
        max_ood_false_positive_rate=.2,
    )
    assert not gate["eligible"]
    assert "ood_recall_below_gate" in gate["reasons"]
    assert "ood_false_positive_rate_above_gate" in gate["reasons"]


def test_promotion_gate_requires_both_ood_classes_when_thresholded():
    missing_positive = {
        "samples": 10,
        "coverage": 1.0,
        "labeled": 10,
        "labeled_coverage": 1.0,
        "heuristic_accuracy": .5,
        "cde_accuracy": .8,
        "ood_labeled": 10,
        "ood_coverage": 1.0,
        "ood_positive_labels": 0,
        "ood_negative_labels": 10,
        "ood_false_positive_rate": 0.0,
    }
    gate = promotion_gate(
        missing_positive,
        min_samples=1,
        min_coverage=.5,
        min_labeled=1,
        min_accuracy_delta=0,
        min_labeled_coverage=.5,
        min_ood_labeled=1,
        min_ood_coverage=.5,
        min_ood_recall=.5,
        max_ood_false_positive_rate=.5,
    )
    assert not gate["eligible"]
    assert "insufficient_ood_positive_samples" in gate["reasons"]
