import json
from pathlib import Path

from scripts.evaluate_decision_routes import RESULT_FORMAT_VERSION, dataset_sha256, file_sha256, validate_results_manifest


def test_results_manifest_matches_exact_dataset_evidence(tmp_path):
    dataset=tmp_path / "dataset.jsonl"
    dataset.write_text('{"text":"one","expected":"fast","ood":false}\\n',encoding="utf-8")
    results=tmp_path / "results.jsonl"
    results.write_text("",encoding="utf-8")
    manifest={"format_version":RESULT_FORMAT_VERSION,"status":"complete","dataset_sha256":dataset_sha256([str(dataset)]),"results_sha256":file_sha256(results)}
    Path(str(results)+".manifest.json").write_text(json.dumps(manifest),encoding="utf-8")
    loaded=validate_results_manifest(results,[str(dataset)])
    assert loaded["dataset_sha256"] == manifest["dataset_sha256"]


def test_results_manifest_rejects_changed_dataset(tmp_path):
    dataset=tmp_path / "dataset.jsonl"
    dataset.write_text("first\\n",encoding="utf-8")
    results=tmp_path / "results.jsonl"
    results.write_text("",encoding="utf-8")
    manifest={"format_version":RESULT_FORMAT_VERSION,"status":"complete","dataset_sha256":dataset_sha256([str(dataset)]),"results_sha256":file_sha256(results)}
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


def test_results_manifest_rejects_tampered_results(tmp_path):
    dataset=tmp_path / "dataset.jsonl"
    dataset.write_text('{"text":"one","expected":"fast","ood":false}\n',encoding="utf-8")
    results=tmp_path / "results.jsonl"
    results.write_text('{"text":"one"}\n',encoding="utf-8")
    manifest={
        "format_version":RESULT_FORMAT_VERSION,
        "status":"complete",
        "dataset_sha256":dataset_sha256([str(dataset)]),
        "results_sha256":file_sha256(results),
    }
    Path(str(results)+".manifest.json").write_text(json.dumps(manifest),encoding="utf-8")
    results.write_text('{"text":"changed"}\n',encoding="utf-8")
    try:
        validate_results_manifest(results,[str(dataset)])
        assert False
    except ValueError as exc:
        assert "does not match its manifest" in str(exc)


def test_results_manifest_rejects_incomplete_collection(tmp_path):
    dataset=tmp_path / "dataset.jsonl"
    dataset.write_text("one\n",encoding="utf-8")
    results=tmp_path / "results.jsonl"
    results.write_text("",encoding="utf-8")
    manifest={
        "format_version":RESULT_FORMAT_VERSION,
        "status":"collecting",
        "dataset_sha256":dataset_sha256([str(dataset)]),
    }
    Path(str(results)+".manifest.json").write_text(json.dumps(manifest),encoding="utf-8")
    try:
        validate_results_manifest(results,[str(dataset)])
        assert False
    except ValueError as exc:
        assert "not complete" in str(exc)


def test_results_manifest_rejects_previous_schema_version(tmp_path):
    results=tmp_path / "results.jsonl"
    results.write_text("",encoding="utf-8")
    Path(str(results)+".manifest.json").write_text(
        json.dumps({"format_version":RESULT_FORMAT_VERSION - 1}),
        encoding="utf-8",
    )
    try:
        validate_results_manifest(results,[])
        assert False
    except ValueError as exc:
        assert "incompatible" in str(exc)
