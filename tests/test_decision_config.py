from pydantic import ValidationError

from core.config import Settings


def test_decision_settings_accept_default_bounds_without_env_files():
    settings=Settings(_env_file=None)
    assert settings.decision_abstain_below == 0.55
    assert settings.decision_max_questions == 32
    assert settings.decision_max_total_output_tokens >= settings.decision_max_questions * 64


def test_decision_settings_reject_invalid_scalar_bounds():
    invalid=(
        {"decision_abstain_below":-0.01},
        {"decision_abstain_below":1.01},
        {"decision_temperature":0},
        {"decision_temperature":float("inf")},
        {"decision_ood_entropy_threshold":float("nan")},
        {"decision_ood_margin_threshold":1.01},
        {"decision_max_questions":0},
        {"decision_max_questions":33},
        {"decision_max_output_tokens":63},
        {"decision_max_output_tokens":2049},
        {"decision_max_total_output_tokens":65537},
        {"decision_max_total_prompt_chars":9999},
        {"decision_call_timeout_seconds":0},
        {"decision_request_timeout_seconds":1801},
    )
    for kwargs in invalid:
        try:
            Settings(_env_file=None,**kwargs)
            assert False,kwargs
        except ValidationError:
            pass


def test_decision_settings_reject_total_output_budget_below_question_floor():
    try:
        Settings(
            _env_file=None,
            decision_max_questions=4,
            decision_max_total_output_tokens=255,
        )
        assert False
    except ValidationError as exc:
        assert "64 tokens per decision question" in str(exc)


def test_decision_settings_accept_matching_total_output_budget_floor():
    settings=Settings(
        _env_file=None,
        decision_max_questions=4,
        decision_max_total_output_tokens=256,
    )
    assert settings.decision_max_total_output_tokens == 256
