from celtia.decision.robustness import decision_margin, normalized_entropy, option_order_max_deviation, ood_signal

def test_entropy_and_margin():
    assert normalized_entropy({"a":.5,"b":.5}) == 1.0
    assert decision_margin({"a":.8,"b":.2}) > .5
    assert ood_signal({"a":.5,"b":.5})["suspected_ood"]

def test_option_order_deviation_is_label_aligned():
    d=[{"a":.8,"b":.2},{"b":.22,"a":.78}]
    assert option_order_max_deviation(d) < .02
