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
