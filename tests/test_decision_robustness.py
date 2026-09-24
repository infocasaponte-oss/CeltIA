from celtia.decision.robustness import decision_margin, normalized_entropy, option_order_max_deviation, option_order_report, ood_signal

def test_entropy_and_margin():
    assert normalized_entropy({"a":.5,"b":.5}) == 1.0
    assert decision_margin({"a":.8,"b":.2}) > .5
    assert ood_signal({"a":.5,"b":.5})["suspected_ood"]

def test_option_order_deviation_is_label_aligned():
    d=[{"a":.8,"b":.2},{"b":.22,"a":.78}]
    assert option_order_max_deviation(d) <= .011


def test_option_order_report_detects_winner_flip():
    report = option_order_report([
        {"a": .51, "b": .49},
        {"b": .52, "a": .48},
    ])
    assert report["permutations"] == 2
    assert report["winner_stable"] is False
    assert report["winners"] == ["a", "b"]

def test_option_order_report_requires_permutations():
    try:
        option_order_report([{"a": .8, "b": .2}])
        assert False
    except ValueError:
        pass
