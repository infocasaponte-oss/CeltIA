from celtia.decision.evaluation import ShadowSample, evaluate_shadow, promotion_gate
from scripts.evaluate_decision_routes import load_cde_results

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



def test_load_cde_results_validates_promotion_inputs(tmp_path):
    good = tmp_path / "good.jsonl"
    good.write_text(
        '{"text":"a","cde":"fast","confidence":0.9,"abstained":false,"suspected_ood":false}\n'
        '{"text":"b","cde":null,"confidence":0.2,"abstained":true,"suspected_ood":true}\n',
        encoding="utf-8",
    )
    rows = load_cde_results(good)
    assert rows["a"]["confidence"] == .9
    assert rows["b"]["abstained"] is True

    bad_rows = (
        '{"text":"a","cde":"fast","confidence":"NaN","abstained":false}',
        '{"text":"a","cde":"fast","confidence":1.1,"abstained":false}',
        '{"text":"a","cde":"fast","confidence":0.9,"abstained":"false"}',
        '{"text":"a","cde":null,"confidence":0.9,"abstained":false}',
        '{"text":"a","cde":"fast","confidence":0.9,"abstained":true}',
        '{"text":"a","cde":"unknown","confidence":0.9,"abstained":false}',
        '{"text":"a","cde":"fast","confidence":0.9,"abstained":false,"suspected_ood":1}',
    )
    for index, line in enumerate(bad_rows):
        path = tmp_path / f"bad-{index}.jsonl"
        path.write_text(line + "\n", encoding="utf-8")
        try:
            load_cde_results(path)
            assert False, line
        except ValueError:
            pass


def test_load_cde_results_rejects_duplicate_text(tmp_path):
    path = tmp_path / "duplicate.jsonl"
    path.write_text(
        '{"text":"same","cde":"fast","confidence":0.9,"abstained":false}\n'
        '{"text":"same","cde":"think","confidence":0.8,"abstained":false}\n',
        encoding="utf-8",
    )
    try:
        load_cde_results(path)
        assert False
    except ValueError as exc:
        assert "duplicate CDE result text" in str(exc)


def test_shadow_evaluation_reports_per_route_metrics():
    samples = [
        ShadowSample("fast", "fast", .9, False, "fast"),
        ShadowSample("fast", None, .2, True, "fast"),
        ShadowSample("think", "fast", .8, False, "think"),
        ShadowSample("code", "code", .9, False, "code"),
        ShadowSample("agent", "agent", .9, False, "agent"),
        ShadowSample("long", "long", .9, False, "long"),
    ]
    metrics = evaluate_shadow(samples)
    assert metrics["per_route"]["fast"] == {
        "labeled": 2,
        "decided": 1,
        "coverage": .5,
        "correct": 1,
        "accuracy": .5,
        "selective_accuracy": 1.0,
    }
    assert metrics["per_route"]["think"]["accuracy"] == 0.0
    assert metrics["min_route_labeled"] == 1
    assert metrics["min_route_coverage"] == .5
    assert metrics["min_route_accuracy"] == 0.0


def test_promotion_gate_blocks_weak_route_even_when_aggregate_is_acceptable():
    samples = []
    for route_name in ("fast", "think", "code", "agent"):
        samples.extend(
            ShadowSample(route_name, route_name, .9, False, route_name)
            for _ in range(10)
        )
    samples.extend(
        ShadowSample("long", None, .2, True, "long")
        for _ in range(10)
    )
    metrics = evaluate_shadow(samples)
    gate = promotion_gate(
        metrics,
        min_samples=1,
        min_coverage=.8,
        min_labeled=1,
        min_accuracy_delta=-1,
        min_labeled_coverage=.8,
        min_route_labeled=10,
        min_route_coverage=.7,
        min_route_accuracy=.6,
    )
    assert not gate["eligible"]
    assert "per_route_coverage_below_gate" in gate["reasons"]
    assert "per_route_accuracy_below_gate" in gate["reasons"]


def test_promotion_gate_requires_minimum_labels_for_every_route():
    metrics = evaluate_shadow([
        ShadowSample("fast", "fast", .9, False, "fast"),
        ShadowSample("think", "think", .9, False, "think"),
    ])
    gate = promotion_gate(
        metrics,
        min_samples=1,
        min_coverage=.5,
        min_labeled=1,
        min_accuracy_delta=-1,
        min_labeled_coverage=.5,
        min_route_labeled=1,
    )
    assert not gate["eligible"]
    assert "insufficient_per_route_labeled_samples" in gate["reasons"]


def test_route_summary_does_not_hide_unlabeled_routes():
    metrics = evaluate_shadow([
        ShadowSample("fast", "fast", .9, False, "fast"),
        ShadowSample("think", "think", .9, False, "think"),
    ])
    assert metrics["routes_with_labels"] == 2
    assert metrics["routes_without_labels"] == ["code", "agent", "long"]
    assert metrics["min_route_labeled"] == 0
    assert metrics["min_route_coverage"] is None
    assert metrics["min_route_accuracy"] is None
