import json
from pathlib import Path

from scripts.evaluate_decision_routes import dataset_sha256, validate_results_manifest


def test_results_manifest_matches_exact_dataset_evidence(tmp_path):
    dataset=tmp_path / "dataset.jsonl"
    dataset.write_text('{"text":"one","expected":"fast","ood":false}\\n',encoding="utf-8")
    results=tmp_path / "results.jsonl"
    results.write_text("",encoding="utf-8")
    manifest={"format_version":2,"dataset_sha256":dataset_sha256([str(dataset)])}
    Path(str(results)+".manifest.json").write_text(json.dumps(manifest),encoding="utf-8")
    loaded=validate_results_manifest(results,[str(dataset)])
    assert loaded["dataset_sha256"] == manifest["dataset_sha256"]


def test_results_manifest_rejects_changed_dataset(tmp_path):
    dataset=tmp_path / "dataset.jsonl"
    dataset.write_text("first\\n",encoding="utf-8")
    results=tmp_path / "results.jsonl"
    results.write_text("",encoding="utf-8")
    manifest={"format_version":2,"dataset_sha256":dataset_sha256([str(dataset)])}
    Path(str(results)+".manifest.json").write_text(json.dumps(manifest),encoding="utf-8")
    dataset.write_text("changed\\n",encoding="utf-8")
    try:
        validate_results_manifest(results,[str(dataset)])
        assert False
    except ValueError as exc:
        assert "does not match" in str(exc)


def test_results_manifest_is_required(tmp_path):
    results=tmp_path / "results.jsonl"
    results.write_text("",encoding="utf-8")
    try:
        validate_results_manifest(results,[])
        assert False
    except ValueError as exc:
        assert "manifest" in str(exc)
