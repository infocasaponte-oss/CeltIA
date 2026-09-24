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
    gate=promotion_gate(m,min_samples=1,min_coverage=.5,min_labeled=1,min_accuracy_delta=0)
    assert gate["eligible"]

def test_gate_blocks_small_dataset():
    g=promotion_gate({"samples":2,"coverage":1.0,"labeled":2,"heuristic_accuracy":.5,"cde_accuracy":1.0})
    assert not g["eligible"]
    assert "insufficient_samples" in g["reasons"]
