import hashlib
import json
from pathlib import Path

from scripts.evaluate_decision_routes import RESULT_FORMAT_VERSION, dataset_sha256, file_sha256, validate_results_manifest


def _backend_provenance():
    return {
        "client_type":"FakeLLM",
        "model":"fake-model",
        "primary_model":None,
        "fallback_model":None,
    }


def _policy_provenance():
    return {
        "abstain_below":0.55,
        "temperature":1.0,
        "reject_suspected_ood":True,
        "ood_entropy_threshold":0.90,
        "ood_margin_threshold":0.10,
    }


def _selection(dataset_rows=1, selected_rows=None, limit=0):
    if selected_rows is None:
        selected_rows=dataset_rows if limit == 0 else min(limit,dataset_rows)
    return {
        "dataset_rows":dataset_rows,
        "selected_rows":selected_rows,
        "limit":limit,
    }


def test_results_manifest_matches_exact_dataset_evidence(tmp_path):
    dataset=tmp_path / "dataset.jsonl"
    dataset.write_text('{"text":"one","expected":"fast","ood":false}\\n',encoding="utf-8")
    results=tmp_path / "results.jsonl"
    results.write_text('{"text":"one"}\n',encoding="utf-8")
    manifest={"format_version":RESULT_FORMAT_VERSION,"status":"complete","collected_at":"2026-01-01T00:00:00+00:00","datasets":[str(dataset)],"selection":_selection(),"backend":_backend_provenance(),"policy":_policy_provenance(),"dataset_sha256":dataset_sha256([str(dataset)]),"results_sha256":file_sha256(results),"result_rows":1}
    Path(str(results)+".manifest.json").write_text(json.dumps(manifest),encoding="utf-8")
    loaded=validate_results_manifest(results,[str(dataset)])
    assert loaded["dataset_sha256"] == manifest["dataset_sha256"]


def test_results_manifest_rejects_changed_dataset(tmp_path):
    dataset=tmp_path / "dataset.jsonl"
    dataset.write_text("first\\n",encoding="utf-8")
    results=tmp_path / "results.jsonl"
    results.write_text("",encoding="utf-8")
    manifest={"format_version":RESULT_FORMAT_VERSION,"status":"complete","collected_at":"2026-01-01T00:00:00+00:00","datasets":[str(dataset)],"selection":_selection(),"backend":_backend_provenance(),"policy":_policy_provenance(),"dataset_sha256":dataset_sha256([str(dataset)]),"results_sha256":file_sha256(results),"result_rows":0}
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
        "datasets":[str(dataset)],"selection":_selection(),
        "backend":_backend_provenance(),
        "policy":_policy_provenance(),
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
        "datasets":[str(dataset)],"selection":_selection(),
        "backend":_backend_provenance(),
        "policy":_policy_provenance(),
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
        "datasets":[str(dataset)],"selection":_selection(),
        "backend":_backend_provenance(),
        "policy":_policy_provenance(),
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
        "datasets":[str(dataset)],"selection":_selection(),
        "backend":_backend_provenance(),
        "policy":_policy_provenance(),
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
        "backend":_backend_provenance(),
        "policy":_policy_provenance(),
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
        "datasets":[str(dataset)],"selection":_selection(),
        "backend":_backend_provenance(),
        "policy":_policy_provenance(),
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



def test_results_manifest_rejects_naive_or_malformed_collection_timestamp(tmp_path):
    dataset=tmp_path / "dataset.jsonl"
    dataset.write_text('{"text":"one","expected":"fast","ood":false}\n',encoding="utf-8")
    results=tmp_path / "results.jsonl"
    results.write_text("",encoding="utf-8")
    base={
        "format_version":RESULT_FORMAT_VERSION,
        "datasets":[str(dataset)],"selection":_selection(),
        "backend":_backend_provenance(),
        "policy":_policy_provenance(),
        "status":"complete",
        "dataset_sha256":dataset_sha256([str(dataset)]),
        "results_sha256":file_sha256(results),
        "result_rows":0,
    }
    for value in ("2026-01-01T00:00:00","not-a-timestamp"):
        manifest={**base,"collected_at":value}
        Path(str(results)+".manifest.json").write_text(json.dumps(manifest),encoding="utf-8")
        try:
            validate_results_manifest(results,[str(dataset)])
            assert False,value
        except ValueError as exc:
            assert "collected_at" in str(exc)



def test_results_manifest_rejects_invalid_resumed_from_collection_timestamp(tmp_path):
    dataset=tmp_path / "dataset.jsonl"
    dataset.write_text('{"text":"one","expected":"fast","ood":false}\n',encoding="utf-8")
    results=tmp_path / "results.jsonl"
    results.write_text("",encoding="utf-8")
    manifest={
        "format_version":RESULT_FORMAT_VERSION,
        "collected_at":"2026-01-01T00:00:00+00:00",
        "resumed_from_collected_at":"2025-12-31T23:00:00",
        "datasets":[str(dataset)],"selection":_selection(),
        "backend":_backend_provenance(),
        "policy":_policy_provenance(),
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
        assert "resumed_from_collected_at" in str(exc)



def test_results_manifest_rejects_resume_timestamp_after_collection(tmp_path):
    dataset=tmp_path / "dataset.jsonl"
    dataset.write_text('{"text":"one","expected":"fast","ood":false}\n',encoding="utf-8")
    results=tmp_path / "results.jsonl"
    results.write_text("",encoding="utf-8")
    manifest={
        "format_version":RESULT_FORMAT_VERSION,
        "collected_at":"2026-01-01T00:00:00+00:00",
        "resumed_from_collected_at":"2026-01-01T00:00:01+00:00",
        "datasets":[str(dataset)],"selection":_selection(),
        "backend":_backend_provenance(),
        "policy":_policy_provenance(),
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
        assert "after collected_at" in str(exc)



def test_results_manifest_rejects_incomplete_backend_provenance(tmp_path):
    dataset=tmp_path / "dataset.jsonl"
    dataset.write_text('{"text":"one","expected":"fast","ood":false}\n',encoding="utf-8")
    results=tmp_path / "results.jsonl"
    results.write_text("",encoding="utf-8")
    manifest={
        "format_version":RESULT_FORMAT_VERSION,
        "collected_at":"2026-01-01T00:00:00+00:00",
        "datasets":[str(dataset)],"selection":_selection(),
        "backend":{"client_type":"FakeLLM"},
        "policy":_policy_provenance(),
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
        assert "backend provenance" in str(exc)


def test_results_manifest_rejects_invalid_policy_provenance(tmp_path):
    dataset=tmp_path / "dataset.jsonl"
    dataset.write_text('{"text":"one","expected":"fast","ood":false}\n',encoding="utf-8")
    results=tmp_path / "results.jsonl"
    results.write_text("",encoding="utf-8")
    invalid_policies=(
        {"abstain_below":0.55},
        {**_policy_provenance(),"reject_suspected_ood":"yes"},
        {**_policy_provenance(),"ood_entropy_threshold":1.1},
        {**_policy_provenance(),"temperature":0},
    )
    for policy in invalid_policies:
        manifest={
            "format_version":RESULT_FORMAT_VERSION,
            "collected_at":"2026-01-01T00:00:00+00:00",
            "datasets":[str(dataset)],"selection":_selection(),
            "backend":_backend_provenance(),
            "policy":policy,
            "status":"complete",
            "dataset_sha256":dataset_sha256([str(dataset)]),
            "results_sha256":file_sha256(results),
            "result_rows":0,
        }
        Path(str(results)+".manifest.json").write_text(json.dumps(manifest),encoding="utf-8")
        try:
            validate_results_manifest(results,[str(dataset)])
            assert False,policy
        except ValueError as exc:
            assert "policy provenance" in str(exc)



def test_results_manifest_rejects_partial_selection_for_promotion(tmp_path):
    dataset=tmp_path / "dataset.jsonl"
    dataset.write_text(
        '{"text":"one","expected":"fast","ood":false}\n'
        '{"text":"two","expected":"fast","ood":false}\n',
        encoding="utf-8",
    )
    results=tmp_path / "results.jsonl"
    results.write_text('{"text":"one"}\n',encoding="utf-8")
    manifest={
        "format_version":RESULT_FORMAT_VERSION,
        "collected_at":"2026-01-01T00:00:00+00:00",
        "datasets":[str(dataset)],
        "selection":_selection(dataset_rows=2,selected_rows=1,limit=1),
        "backend":_backend_provenance(),
        "policy":_policy_provenance(),
        "status":"complete",
        "dataset_sha256":dataset_sha256([str(dataset)]),
        "results_sha256":file_sha256(results),
        "result_rows":1,
    }
    Path(str(results)+".manifest.json").write_text(json.dumps(manifest),encoding="utf-8")
    try:
        validate_results_manifest(results,[str(dataset)])
        assert False
    except ValueError as exc:
        assert "full-dataset collection" in str(exc)

    loaded=validate_results_manifest(
        results,
        [str(dataset)],
        require_full_selection=False,
    )
    assert loaded["selection"]["selected_rows"] == 1


def test_results_manifest_rejects_selection_count_inconsistent_with_limit(tmp_path):
    dataset=tmp_path / "dataset.jsonl"
    dataset.write_text(
        '{"text":"one","expected":"fast","ood":false}\n'
        '{"text":"two","expected":"fast","ood":false}\n',
        encoding="utf-8",
    )
    results=tmp_path / "results.jsonl"
    results.write_text('{"text":"one"}\n',encoding="utf-8")
    manifest={
        "format_version":RESULT_FORMAT_VERSION,
        "collected_at":"2026-01-01T00:00:00+00:00",
        "datasets":[str(dataset)],
        "selection":_selection(dataset_rows=2,selected_rows=1,limit=0),
        "backend":_backend_provenance(),
        "policy":_policy_provenance(),
        "status":"complete",
        "dataset_sha256":dataset_sha256([str(dataset)]),
        "results_sha256":file_sha256(results),
        "result_rows":1,
    }
    Path(str(results)+".manifest.json").write_text(json.dumps(manifest),encoding="utf-8")
    try:
        validate_results_manifest(results,[str(dataset)])
        assert False
    except ValueError as exc:
        assert "selection does not match its limit" in str(exc)
