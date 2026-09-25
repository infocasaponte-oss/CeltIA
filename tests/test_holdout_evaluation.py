import importlib.util
from pathlib import Path

import pytest

from celtia.decision.route_contract import ROUTE_OOD_SYSTEM_GUIDANCE, ROUTE_SYSTEM_GUIDANCE

_spec = importlib.util.spec_from_file_location(
    "run_cde_holdout", Path(__file__).resolve().parents[1] / "scripts" / "run_cde_holdout.py"
)
holdout = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(holdout)


def _row(expected, cde, ood=False, suspected=False, p=0.9):
    return {"expected": expected, "cde": cde, "expected_ood": ood, "suspected_ood": suspected, "ood_probability": p}


def _perfect():
    rows = [_row(route, route) for route in holdout.ROUTES for _ in range(4)]
    rows += [_row(None, None, ood=True, suspected=True, p=0.99) for _ in range(4)]
    return rows


def test_perfect_run_passes():
    report = holdout.evaluate(_perfect())
    assert report["pass"] is True
    assert report["ood"]["recall"] == 1.0 and report["ood"]["specificity"] == 1.0


def test_single_weak_route_fails_even_with_high_global_accuracy():
    rows = _perfect() + [_row("long", "long") for _ in range(60)]
    rows = [r for r in rows if not (r["expected"] == "agent")] + [_row("agent", "fast") for _ in range(4)]
    report = holdout.evaluate(rows)
    assert report["checks"]["each_route_accuracy"] is False
    assert report["pass"] is False


def test_logit_ties_fail_the_run():
    rows = _perfect()
    rows[-1]["ood_probability"] = 0.5
    assert holdout.evaluate(rows)["checks"]["no_logit_ties"] is False


def test_ood_metrics_count_missed_attacks_and_false_alarms():
    rows = _perfect() + [_row(None, None, ood=True, suspected=False, p=0.1) for _ in range(4)]
    rows += [_row("fast", "fast", ood=False, suspected=True, p=0.9) for _ in range(2)]
    report = holdout.evaluate(rows)
    assert report["ood"]["fn"] == 4 and report["ood"]["fp"] == 2
    assert report["checks"]["ood_recall"] is False


def test_evaluation_requires_both_kinds_of_rows():
    with pytest.raises(ValueError):
        holdout.evaluate([_row("fast", "fast")] * 3)


def test_contract_separates_execution_from_writing_and_uses_size_metadata():
    assert "Run the cleanup script" in ROUTE_SYSTEM_GUIDANCE
    assert "input_chars >= long_context_chars" in ROUTE_SYSTEM_GUIDANCE


def test_ood_contract_covers_meta_discourse_and_candidate_manipulation():
    assert "THIS system" in ROUTE_OOD_SYSTEM_GUIDANCE
    assert "whether or not it uses the word \"route\"" in ROUTE_OOD_SYSTEM_GUIDANCE
    assert "general knowledge question about classifiers" in ROUTE_OOD_SYSTEM_GUIDANCE
    assert "large document set" in ROUTE_OOD_SYSTEM_GUIDANCE
