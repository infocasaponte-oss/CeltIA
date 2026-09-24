from celtia.decision.backend_prefix_eval import (
    backend_prefix_report,
    backend_prefix_round_report,
    cached_prompt_tokens,
    latency_summary,
    round_order_sequence,
)


def test_cached_prompt_tokens_reads_optional_openai_metadata():
    assert cached_prompt_tokens({"prompt_tokens_details":{"cached_tokens":123}}) == 123
    assert cached_prompt_tokens({"prompt_tokens_details":{"cached_tokens":True}}) is None
    assert cached_prompt_tokens({"prompt_tokens":123}) is None
    assert cached_prompt_tokens(None) is None


def test_latency_summary_is_deterministic():
    summary = latency_summary([
        {"latency_ms": 30},
        {"latency_ms": 10},
        {"latency_ms": 20},
        {"latency_ms": 40},
    ])
    assert summary == {
        "samples": 4,
        "median_ms": 25.0,
        "p95_ms": 40.0,
        "min_ms": 10.0,
        "max_ms": 40.0,
    }


def test_backend_prefix_report_skips_uncacheable_first_call():
    shared = [
        {"latency_ms": 100, "usage": {"prompt_tokens_details":{"cached_tokens":0}}},
        {"latency_ms": 40, "usage": {"prompt_tokens_details":{"cached_tokens":800}}},
        {"latency_ms": 50, "usage": {"prompt_tokens_details":{"cached_tokens":800}}},
    ]
    control = [
        {"latency_ms": 100, "usage": {"prompt_tokens_details":{"cached_tokens":0}}},
        {"latency_ms": 80, "usage": {"prompt_tokens_details":{"cached_tokens":0}}},
        {"latency_ms": 100, "usage": {"prompt_tokens_details":{"cached_tokens":0}}},
    ]
    report = backend_prefix_report(shared, control)
    assert report["shared"]["median_ms"] == 45.0
    assert report["control"]["median_ms"] == 90.0
    assert report["median_speedup_ratio"] == 2.0
    assert report["median_latency_reduction"] == 0.5
    assert report["cached_token_metadata_available"] is True
    assert report["shared_cached_prompt_tokens"] == [800, 800]


def test_backend_prefix_report_rejects_invalid_samples():
    invalid_pairs = (
        ([{"latency_ms":1}], [{"latency_ms":1}]),
        ([{"latency_ms":1},{"latency_ms":2}], [{"latency_ms":1}]),
        ([{"latency_ms":1},{"latency_ms":float("nan")}], [{"latency_ms":1},{"latency_ms":2}]),
    )
    for shared, control in invalid_pairs:
        try:
            backend_prefix_report(shared, control)
            assert False
        except ValueError:
            pass



def test_round_order_sequence_alternates_deterministically():
    assert round_order_sequence(5) == (
        "control-first",
        "shared-first",
        "control-first",
        "shared-first",
        "control-first",
    )
    assert round_order_sequence(3, first="shared-first") == (
        "shared-first",
        "control-first",
        "shared-first",
    )
    for args in ((0,),):
        try:
            round_order_sequence(*args)
            assert False
        except ValueError:
            pass


def test_backend_prefix_report_allows_single_evaluated_sample_without_warmup_skip():
    report = backend_prefix_report(
        [{"latency_ms": 40}],
        [{"latency_ms": 80}],
        skip_first=False,
    )
    assert report["median_speedup_ratio"] == 2.0
    assert report["median_latency_reduction"] == 0.5


def test_backend_prefix_round_report_drops_each_round_warmup():
    rounds = [
        {
            "order": "control-first",
            "shared_samples": [
                {"latency_ms": 100},
                {"latency_ms": 40},
                {"latency_ms": 50},
            ],
            "control_samples": [
                {"latency_ms": 100},
                {"latency_ms": 80},
                {"latency_ms": 100},
            ],
        },
        {
            "order": "shared-first",
            "shared_samples": [
                {"latency_ms": 900},
                {"latency_ms": 60},
                {"latency_ms": 70},
            ],
            "control_samples": [
                {"latency_ms": 900},
                {"latency_ms": 120},
                {"latency_ms": 140},
            ],
        },
    ]
    report = backend_prefix_round_report(rounds)
    assert report["round_count"] == 2
    assert report["shared"]["samples"] == 4
    assert report["control"]["samples"] == 4
    assert report["shared"]["median_ms"] == 55.0
    assert report["control"]["median_ms"] == 110.0
    assert report["median_speedup_ratio"] == 2.0
    assert report["round_reports"][0]["order"] == "control-first"
    assert report["round_reports"][1]["order"] == "shared-first"


def test_backend_prefix_round_report_rejects_empty_or_malformed_rounds():
    invalid = (
        [],
        [{"shared_samples": [], "control_samples": []}],
        [{"shared_samples": "bad", "control_samples": []}],
    )
    for rounds in invalid:
        try:
            backend_prefix_round_report(rounds)
            assert False
        except ValueError:
            pass
