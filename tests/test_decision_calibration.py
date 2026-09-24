from celtia.decision.calibration import multiclass_brier, top_label_calibration

def test_multiclass_brier_perfect():
    assert multiclass_brier([{"a":1.0,"b":0.0}],["a"]) == 0.0

def test_top_label_calibration():
    confidence,correct=top_label_calibration([{"a":.8,"b":.2},{"a":.3,"b":.7}],["a","a"])
    assert confidence == [.8,.7]
    assert correct == [True,False]
