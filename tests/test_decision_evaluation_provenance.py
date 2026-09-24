import hashlib
import json
from pathlib import Path

from scripts.evaluate_decision_routes import RESULT_FORMAT_VERSION, dataset_sha256, file_sha256, validate_results_manifest


def test_results_manifest_matches_exact_dataset_evidence(tmp_path):
    dataset=tmp_path / "dataset.jsonl"
    dataset.write_text('{"text":"one","expected":"fast","ood":false}\\n',encoding="utf-8")
    results=tmp_path / "results.jsonl"
    results.write_text("",encoding="utf-8")
    manifest={"format_version":RESULT_FORMAT_VERSION,"status":"complete","collected_at":"2026-01-01T00:00:00+00:00","datasets":[str(dataset)],"backend":{},"policy":{},"dataset_sha256":dataset_sha256([str(dataset)]),"results_sha256":file_sha256(results),"result_rows":0}
    Path(str(results)+".manifest.json").write_text(json.dumps(manifest),encoding="utf-8")
    loaded=validate_results_manifest(results,[str(dataset)])
    assert loaded["dataset_sha256"] == manifest["dataset_sha256"]


def test_results_manifest_rejects_changed_dataset(tmp_path):
    dataset=tmp_path / "dataset.jsonl"
    dataset.write_text("first\\n",encoding="utf-8")
    results=tmp_path / "results.jsonl"
    results.write_text("",encoding="utf-8")
    manifest={"format_version":RESULT_FORMAT_VERSION,"status":"complete","collected_at":"2026-01-01T00:00:00+00:00","datasets":[str(dataset)],"backend":{},"policy":{},"dataset_sha256":dataset_sha256([str(dataset)]),"results_sha256":file_sha256(results),"result_rows":0}
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
        "collected_at":"2026-01-01T00:00:00+00:00",
        "datasets":[str(dataset)],
        "backend":{},
        "policy":{},
        "status":"complete",
        "dataset_sha256":dataset_sha256([str(dataset)]),
        "results_sha256":file_sha256(results),
        "result_rows":1,
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
        "collected_at":"2026-01-01T00:00:00+00:00",
        "datasets":[str(dataset)],
        "backend":{},
        "policy":{},
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


def test_results_manifest_rejects_wrong_result_row_count(tmp_path):
    dataset=tmp_path / "dataset.jsonl"
    dataset.write_text('{"text":"one","expected":"fast","ood":false}\n',encoding="utf-8")
    results=tmp_path / "results.jsonl"
    results.write_text('{"text":"one"}\n',encoding="utf-8")
    manifest={
        "format_version":RESULT_FORMAT_VERSION,
        "collected_at":"2026-01-01T00:00:00+00:00",
        "datasets":[str(dataset)],
        "backend":{},
        "policy":{},
        "status":"complete",
        "dataset_sha256":dataset_sha256([str(dataset)]),
        "results_sha256":file_sha256(results),
        "result_rows":2,
    }
    Path(str(results)+".manifest.json").write_text(json.dumps(manifest),encoding="utf-8")
    try:
        validate_results_manifest(results,[str(dataset)])
        assert False
    except ValueError as exc:
        assert "row count" in str(exc)


def test_results_manifest_rejects_invalid_result_row_count_type(tmp_path):
    dataset=tmp_path / "dataset.jsonl"
    dataset.write_text('{"text":"one","expected":"fast","ood":false}\n',encoding="utf-8")
    results=tmp_path / "results.jsonl"
    results.write_text("",encoding="utf-8")
    manifest={
        "format_version":RESULT_FORMAT_VERSION,
        "collected_at":"2026-01-01T00:00:00+00:00",
        "datasets":[str(dataset)],
        "backend":{},
        "policy":{},
        "status":"complete",
        "dataset_sha256":dataset_sha256([str(dataset)]),
        "results_sha256":file_sha256(results),
        "result_rows":True,
    }
    Path(str(results)+".manifest.json").write_text(json.dumps(manifest),encoding="utf-8")
    try:
        validate_results_manifest(results,[str(dataset)])
        assert False
    except ValueError as exc:
        assert "invalid result_rows" in str(exc)


def test_dataset_digest_streaming_preserves_schema_v3_bytes(tmp_path):
    first=tmp_path / "first.jsonl"
    second=tmp_path / "second.jsonl"
    first.write_bytes((b"a" * (1024 * 1024 + 17)) + b"\n")
    second.write_bytes(b"second\n")

    paths=[str(first),str(second)]
    expected=hashlib.sha256()
    for raw_path in paths:
        path=Path(raw_path)
        expected.update(str(path).encode("utf-8"))
        expected.update(b"\0")
        expected.update(path.read_bytes())
        expected.update(b"\0")

    assert dataset_sha256(paths) == expected.hexdigest()


def test_results_manifest_rejects_dataset_list_mismatch(tmp_path):
    dataset=tmp_path / "dataset.jsonl"
    dataset.write_text('{"text":"one","expected":"fast","ood":false}\n',encoding="utf-8")
    results=tmp_path / "results.jsonl"
    results.write_text("",encoding="utf-8")
    manifest={
        "format_version":RESULT_FORMAT_VERSION,
        "collected_at":"2026-01-01T00:00:00+00:00",
        "datasets":["different.jsonl"],
        "backend":{},
        "policy":{},
        "status":"complete",
        "dataset_sha256":dataset_sha256([str(dataset)]),
        "results_sha256":file_sha256(results),
        "result_rows":0,
    }
    Path(str(results)+".manifest.json").write_text(json.dumps(manifest),encoding="utf-8")
    try:
        validate_results_manifest(results,[str(dataset)])
        assert False
    except ValueError as exc:
        assert "dataset list" in str(exc)


def test_results_manifest_rejects_missing_structural_provenance(tmp_path):
    dataset=tmp_path / "dataset.jsonl"
    dataset.write_text('{"text":"one","expected":"fast","ood":false}\n',encoding="utf-8")
    results=tmp_path / "results.jsonl"
    results.write_text("",encoding="utf-8")
    base={
        "format_version":RESULT_FORMAT_VERSION,
        "collected_at":"2026-01-01T00:00:00+00:00",
        "datasets":[str(dataset)],
        "backend":{},
        "policy":{},
        "status":"complete",
        "dataset_sha256":dataset_sha256([str(dataset)]),
        "results_sha256":file_sha256(results),
        "result_rows":0,
    }
    for field in ("collected_at","backend","policy"):
        manifest=dict(base)
        manifest.pop(field)
        Path(str(results)+".manifest.json").write_text(json.dumps(manifest),encoding="utf-8")
        try:
            validate_results_manifest(results,[str(dataset)])
            assert False,field
        except ValueError:
            pass
