import asyncio
import json
from argparse import Namespace

from scripts import collect_decision_eval_results as collector


class FakeResult:
    def __init__(self, decision="fast"):
        self.decision=decision
        self.confidence=.9
        self.abstained=False
        self.suspected_ood=False
        self.abstention_reason=None
        self.normalized_entropy=.1
        self.margin=.8


class FakeRuntime:
    def __init__(self):
        self.llm=type("FakeLLM", (), {"model":"fake-model"})()
        self.engine_options={"abstain_below":.55,"temperature":1.0,"reject_suspected_ood":True,"ood_entropy_threshold":.90,"ood_margin_threshold":.10}

    async def decide(self, context, questions):
        return [FakeResult()]

    async def decide_with_usage(self, context, questions):
        return [FakeResult()], {"prompt_tokens":1,"completion_tokens":1,"total_tokens":2,"models":["fake-model"]}


def _args(tmp_path, *, resume=False, limit=0, checkpoint_every=2):
    return Namespace(
        dataset=[],
        output=str(tmp_path / "results.jsonl"),
        limit=limit,
        sleep_seconds=0.0,
        resume=resume,
        checkpoint_every=checkpoint_every,
    )


def test_collect_checkpoints_and_resumes_without_duplicate_calls(tmp_path, monkeypatch):
    rows=[
        {"text":"one","expected":"fast","ood":False},
        {"text":"two","expected":"fast","ood":False},
        {"text":"three","expected":"fast","ood":False},
    ]
    monkeypatch.setattr(collector, "load_datasets", lambda paths: rows)
    monkeypatch.setattr(collector, "build_runtime", FakeRuntime)
    monkeypatch.setattr(collector, "route", lambda text: type("R", (), {"mode":"fast"})())

    first=asyncio.run(collector.collect(_args(tmp_path, limit=2, checkpoint_every=1)))
    assert first["written"] == 2
    assert first["selected"] == 2
    path=tmp_path / "results.jsonl"
    saved=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [item["text"] for item in saved] == ["one","two"]
    assert all(item["models_used"] == ["fake-model"] for item in saved)
    assert not list(tmp_path.glob("results.jsonl.*.tmp"))
    manifest_path=tmp_path / "results.jsonl.manifest.json"
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["format_version"] == collector.RESULT_FORMAT_VERSION
    assert manifest["status"] == "complete"
    assert manifest["selection"] == {"dataset_rows":3,"selected_rows":2,"limit":2}
    assert manifest["result_rows"] == 2
    assert manifest["results_sha256"] == collector.file_sha256(path)
    assert first["manifest_output"] == str(manifest_path)

    second=asyncio.run(collector.collect(_args(tmp_path, resume=True, limit=2)))
    assert second["written"] == 0
    assert second["skipped"] == 2
    saved_again=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [item["text"] for item in saved_again] == ["one","two"]


def test_atomic_writer_replaces_complete_file(tmp_path):
    path=tmp_path / "results.jsonl"
    path.write_text("stale\n",encoding="utf-8")
    rows=[
        {"text":"one","cde":"fast","confidence":.9,"abstained":False},
        {"text":"two","cde":None,"confidence":.2,"abstained":True},
    ]
    collector._write_atomic_jsonl(path,rows)
    parsed=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert parsed == rows
    assert not list(tmp_path.glob("results.jsonl.*.tmp"))


def test_atomic_writer_preserves_destination_and_cleans_temp_on_replace_failure(tmp_path, monkeypatch):
    path=tmp_path / "results.jsonl"
    path.write_text('{"text":"old"}\n',encoding="utf-8")

    def fail_replace(source, destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(collector.os, "replace", fail_replace)
    try:
        collector._write_atomic_jsonl(path,[{"text":"new"}])
        assert False
    except OSError as exc:
        assert "simulated replace failure" in str(exc)

    assert json.loads(path.read_text(encoding="utf-8")) == {"text":"old"}
    assert not list(tmp_path.glob("results.jsonl.*.tmp"))


def test_runtime_manifest_records_dataset_digest_backend_and_policy(tmp_path):
    dataset=tmp_path / "dataset.jsonl"
    dataset.write_text('{"text":"one","expected":"fast","ood":false}\n',encoding="utf-8")
    runtime=FakeRuntime()
    manifest=collector._runtime_manifest(runtime,[str(dataset)])
    assert manifest["format_version"] == collector.RESULT_FORMAT_VERSION
    assert manifest["datasets"] == [str(dataset)]
    assert len(manifest["dataset_sha256"]) == 64
    assert manifest["backend"]["client_type"] == "FakeLLM"
    assert manifest["backend"]["model"] == "fake-model"
    assert manifest["policy"]["abstain_below"] == .55
    assert manifest["collected_at"].endswith("+00:00")


def test_dataset_digest_changes_when_evaluation_data_changes(tmp_path):
    dataset=tmp_path / "dataset.jsonl"
    dataset.write_text("first\n",encoding="utf-8")
    first=collector.dataset_sha256([str(dataset)])
    dataset.write_text("second\n",encoding="utf-8")
    second=collector.dataset_sha256([str(dataset)])
    assert first != second


def test_atomic_json_writer_preserves_destination_on_replace_failure(tmp_path, monkeypatch):
    path=tmp_path / "manifest.json"
    path.write_text('{"old":true}\n',encoding="utf-8")
    monkeypatch.setattr(
        collector.os,
        "replace",
        lambda source,destination: (_ for _ in ()).throw(OSError("replace failed")),
    )
    try:
        collector._write_atomic_json(path,{"new":True})
        assert False
    except OSError:
        pass
    assert json.loads(path.read_text(encoding="utf-8")) == {"old":True}
    assert not list(tmp_path.glob("manifest.json.*.tmp"))


def test_resume_rejects_missing_provenance_manifest(tmp_path, monkeypatch):
    rows=[{"text":"one","expected":"fast","ood":False}]
    monkeypatch.setattr(collector,"load_datasets",lambda paths: rows)
    path=tmp_path / "results.jsonl"
    path.write_text('{"text":"one","cde":"fast","confidence":0.9,"abstained":false}\n',encoding="utf-8")
    try:
        asyncio.run(collector.collect(_args(tmp_path,resume=True)))
        assert False
    except ValueError as exc:
        assert "provenance manifest" in str(exc)


def test_resume_rejects_changed_dataset_provenance(tmp_path, monkeypatch):
    dataset=tmp_path / "dataset.jsonl"
    dataset.write_text('{"text":"one","expected":"fast","ood":false}\n',encoding="utf-8")
    args=_args(tmp_path)
    args.dataset=[str(dataset)]
    monkeypatch.setattr(collector,"build_runtime",FakeRuntime)
    monkeypatch.setattr(collector,"route",lambda text: type("R",(),{"mode":"fast"})())
    asyncio.run(collector.collect(args))

    dataset.write_text(
        '{"text":"one","expected":"fast","ood":false}\n'
        '{"text":"two","expected":"fast","ood":false}\n',
        encoding="utf-8",
    )
    resume_args=_args(tmp_path,resume=True)
    resume_args.dataset=[str(dataset)]
    try:
        asyncio.run(collector.collect(resume_args))
        assert False
    except ValueError as exc:
        assert "dataset provenance" in str(exc)


def test_resume_rejects_changed_runtime_policy(tmp_path, monkeypatch):
    dataset=tmp_path / "dataset.jsonl"
    dataset.write_text('{"text":"one","expected":"fast","ood":false}\n',encoding="utf-8")
    args=_args(tmp_path)
    args.dataset=[str(dataset)]
    monkeypatch.setattr(collector,"build_runtime",FakeRuntime)
    monkeypatch.setattr(collector,"route",lambda text: type("R",(),{"mode":"fast"})())
    asyncio.run(collector.collect(args))

    class ChangedRuntime(FakeRuntime):
        def __init__(self):
            super().__init__()
            self.engine_options={"abstain_below":.75,"temperature":1.0}

    monkeypatch.setattr(collector,"build_runtime",ChangedRuntime)
    resume_args=_args(tmp_path,resume=True)
    resume_args.dataset=[str(dataset)]
    try:
        asyncio.run(collector.collect(resume_args))
        assert False
    except ValueError as exc:
        assert "policy provenance" in str(exc)


def test_fresh_collection_refuses_to_overwrite_existing_results(tmp_path, monkeypatch):
    path=tmp_path / "results.jsonl"
    path.write_text('{"text":"old"}\n',encoding="utf-8")
    monkeypatch.setattr(collector,"load_datasets",lambda paths:[{"text":"one","expected":"fast","ood":False}])
    try:
        asyncio.run(collector.collect(_args(tmp_path)))
        assert False
    except ValueError as exc:
        assert "already exists" in str(exc)
    assert path.read_text(encoding="utf-8") == '{"text":"old"}\n'


def test_interrupted_fresh_collection_persists_manifest_for_resume(tmp_path, monkeypatch):
    rows=[
        {"text":"one","expected":"fast","ood":False},
        {"text":"two","expected":"fast","ood":False},
    ]
    monkeypatch.setattr(collector,"load_datasets",lambda paths: rows)
    monkeypatch.setattr(collector,"build_runtime",FakeRuntime)
    monkeypatch.setattr(collector,"route",lambda text:type("R",(),{"mode":"fast"})())

    calls={"count":0}
    original=collector.collect_one

    async def fail_after_first(runtime,row):
        calls["count"]+=1
        if calls["count"] > 1:
            raise RuntimeError("simulated interruption")
        return await original(runtime,row)

    monkeypatch.setattr(collector,"collect_one",fail_after_first)
    try:
        asyncio.run(collector.collect(_args(tmp_path,checkpoint_every=1)))
        assert False
    except RuntimeError as exc:
        assert "interruption" in str(exc)

    path=tmp_path / "results.jsonl"
    manifest_path=tmp_path / "results.jsonl.manifest.json"
    assert path.exists()
    assert manifest_path.exists()
    interrupted_manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    assert interrupted_manifest["format_version"] == collector.RESULT_FORMAT_VERSION
    assert interrupted_manifest["status"] == "collecting"

    monkeypatch.setattr(collector,"collect_one",original)
    resumed=asyncio.run(collector.collect(_args(tmp_path,resume=True,checkpoint_every=1)))
    assert resumed["skipped"] == 1
    assert resumed["written"] == 1
    saved=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [item["text"] for item in saved] == ["one","two"]


def test_resume_rejects_tampered_completed_results(tmp_path, monkeypatch):
    dataset=tmp_path / "dataset.jsonl"
    dataset.write_text('{"text":"one","expected":"fast","ood":false}\n',encoding="utf-8")
    args=_args(tmp_path)
    args.dataset=[str(dataset)]
    monkeypatch.setattr(collector,"build_runtime",FakeRuntime)
    monkeypatch.setattr(collector,"route",lambda text:type("R",(),{"mode":"fast"})())
    asyncio.run(collector.collect(args))

    path=tmp_path / "results.jsonl"
    path.write_text(
        '{"text":"one","cde":"fast","confidence":0.1,"abstained":false,"suspected_ood":false}\n',
        encoding="utf-8",
    )
    resume_args=_args(tmp_path,resume=True)
    resume_args.dataset=[str(dataset)]
    try:
        asyncio.run(collector.collect(resume_args))
        assert False
    except ValueError as exc:
        assert "completed provenance manifest" in str(exc)


def test_repeated_resume_preserves_original_collection_timestamp(tmp_path, monkeypatch):
    dataset=tmp_path / "dataset.jsonl"
    dataset.write_text('{"text":"one","expected":"fast","ood":false}\n',encoding="utf-8")
    monkeypatch.setattr(collector,"build_runtime",FakeRuntime)
    monkeypatch.setattr(collector,"route",lambda text:type("R",(),{"mode":"fast"})())

    args=_args(tmp_path)
    args.dataset=[str(dataset)]
    asyncio.run(collector.collect(args))

    manifest_path=tmp_path / "results.jsonl.manifest.json"
    original=json.loads(manifest_path.read_text(encoding="utf-8"))
    original_collected_at=original["collected_at"]

    resume_args=_args(tmp_path,resume=True)
    resume_args.dataset=[str(dataset)]
    asyncio.run(collector.collect(resume_args))
    first_resume=json.loads(manifest_path.read_text(encoding="utf-8"))
    assert first_resume["resumed_from_collected_at"] == original_collected_at

    asyncio.run(collector.collect(resume_args))
    second_resume=json.loads(manifest_path.read_text(encoding="utf-8"))
    assert second_resume["resumed_from_collected_at"] == original_collected_at



def test_resume_rejects_collecting_manifest_dataset_list_mismatch(tmp_path, monkeypatch):
    dataset=tmp_path / "dataset.jsonl"
    dataset.write_text('{"text":"one","expected":"fast","ood":false}\n',encoding="utf-8")
    path=tmp_path / "results.jsonl"
    path.write_text(
        '{"text":"one","cde":"fast","confidence":0.9,"abstained":false,"models_used":["fake-model"]}\n',
        encoding="utf-8",
    )
    manifest=collector._runtime_manifest(FakeRuntime(),[str(dataset)])
    manifest["datasets"]=["different.jsonl"]
    (tmp_path / "results.jsonl.manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    monkeypatch.setattr(collector,"build_runtime",FakeRuntime)

    args=_args(tmp_path,resume=True)
    args.dataset=[str(dataset)]
    try:
        asyncio.run(collector.collect(args))
        assert False
    except ValueError as exc:
        assert "dataset list" in str(exc)


def test_resume_rejects_collecting_manifest_missing_structural_provenance(tmp_path, monkeypatch):
    dataset=tmp_path / "dataset.jsonl"
    dataset.write_text('{"text":"one","expected":"fast","ood":false}\n',encoding="utf-8")
    path=tmp_path / "results.jsonl"
    path.write_text(
        '{"text":"one","cde":"fast","confidence":0.9,"abstained":false,"models_used":["fake-model"]}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(collector,"build_runtime",FakeRuntime)

    for field in ("collected_at","backend","policy","code_revision","code_dirty"):
        manifest=collector._runtime_manifest(FakeRuntime(),[str(dataset)])
        manifest.pop(field)
        (tmp_path / "results.jsonl.manifest.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )
        args=_args(tmp_path,resume=True)
        args.dataset=[str(dataset)]
        try:
            asyncio.run(collector.collect(args))
            assert False,field
        except ValueError:
            pass



def test_resume_rejects_tampered_collecting_checkpoint(tmp_path, monkeypatch):
    rows=[
        {"text":"one","expected":"fast","ood":False},
        {"text":"two","expected":"fast","ood":False},
    ]
    monkeypatch.setattr(collector,"load_datasets",lambda paths: rows)
    monkeypatch.setattr(collector,"build_runtime",FakeRuntime)
    monkeypatch.setattr(collector,"route",lambda text:type("R",(),{"mode":"fast"})())

    calls={"count":0}
    original=collector.collect_one

    async def fail_after_first(runtime,row):
        calls["count"]+=1
        if calls["count"] > 1:
            raise RuntimeError("simulated interruption")
        return await original(runtime,row)

    monkeypatch.setattr(collector,"collect_one",fail_after_first)
    try:
        asyncio.run(collector.collect(_args(tmp_path,checkpoint_every=1)))
        assert False
    except RuntimeError:
        pass

    path=tmp_path / "results.jsonl"
    manifest_path=tmp_path / "results.jsonl.manifest.json"
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "collecting"
    assert manifest["results_sha256"] == collector.file_sha256(path)
    assert manifest["result_rows"] == 1

    path.write_text(
        '{"text":"one","cde":"fast","confidence":0.1,"abstained":false,"models_used":["fake-model"]}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(collector,"collect_one",original)
    try:
        asyncio.run(collector.collect(_args(tmp_path,resume=True,checkpoint_every=1)))
        assert False
    except ValueError as exc:
        assert "collecting checkpoint results" in str(exc)



def test_resume_rejects_collecting_manifest_naive_collection_timestamp(tmp_path, monkeypatch):
    dataset=tmp_path / "dataset.jsonl"
    dataset.write_text('{"text":"one","expected":"fast","ood":false}\n',encoding="utf-8")
    path=tmp_path / "results.jsonl"
    path.write_text(
        '{"text":"one","cde":"fast","confidence":0.9,"abstained":false,"models_used":["fake-model"]}\n',
        encoding="utf-8",
    )
    manifest=collector._runtime_manifest(FakeRuntime(),[str(dataset)])
    manifest["collected_at"]="2026-01-01T00:00:00"
    manifest["results_sha256"]=collector.file_sha256(path)
    manifest["result_rows"]=1
    (tmp_path / "results.jsonl.manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    monkeypatch.setattr(collector,"build_runtime",FakeRuntime)

    args=_args(tmp_path,resume=True)
    args.dataset=[str(dataset)]
    try:
        asyncio.run(collector.collect(args))
        assert False
    except ValueError as exc:
        assert "collected_at" in str(exc)



def test_resume_rejects_collecting_manifest_invalid_root_timestamp(tmp_path, monkeypatch):
    dataset=tmp_path / "dataset.jsonl"
    dataset.write_text('{"text":"one","expected":"fast","ood":false}\n',encoding="utf-8")
    path=tmp_path / "results.jsonl"
    path.write_text(
        '{"text":"one","cde":"fast","confidence":0.9,"abstained":false,"models_used":["fake-model"]}\n',
        encoding="utf-8",
    )
    manifest=collector._runtime_manifest(FakeRuntime(),[str(dataset)])
    manifest["resumed_from_collected_at"]="2025-12-31T23:00:00"
    manifest["results_sha256"]=collector.file_sha256(path)
    manifest["result_rows"]=1
    (tmp_path / "results.jsonl.manifest.json").write_text(json.dumps(manifest),encoding="utf-8")
    monkeypatch.setattr(collector,"build_runtime",FakeRuntime)

    args=_args(tmp_path,resume=True)
    args.dataset=[str(dataset)]
    try:
        asyncio.run(collector.collect(args))
        assert False
    except ValueError as exc:
        assert "resumed_from_collected_at" in str(exc)



def test_resume_rejects_collecting_manifest_root_after_current_timestamp(tmp_path, monkeypatch):
    dataset=tmp_path / "dataset.jsonl"
    dataset.write_text('{"text":"one","expected":"fast","ood":false}\n',encoding="utf-8")
    path=tmp_path / "results.jsonl"
    path.write_text(
        '{"text":"one","cde":"fast","confidence":0.9,"abstained":false,"models_used":["fake-model"]}\n',
        encoding="utf-8",
    )
    manifest=collector._runtime_manifest(FakeRuntime(),[str(dataset)])
    manifest["collected_at"]="2026-01-01T00:00:00+00:00"
    manifest["resumed_from_collected_at"]="2026-01-01T00:00:01+00:00"
    manifest["results_sha256"]=collector.file_sha256(path)
    manifest["result_rows"]=1
    (tmp_path / "results.jsonl.manifest.json").write_text(json.dumps(manifest),encoding="utf-8")
    monkeypatch.setattr(collector,"build_runtime",FakeRuntime)

    args=_args(tmp_path,resume=True)
    args.dataset=[str(dataset)]
    try:
        asyncio.run(collector.collect(args))
        assert False
    except ValueError as exc:
        assert "after collected_at" in str(exc)



def test_partial_pilot_can_resume_into_full_collection(tmp_path, monkeypatch):
    rows=[
        {"text":"one","expected":"fast","ood":False},
        {"text":"two","expected":"fast","ood":False},
        {"text":"three","expected":"fast","ood":False},
    ]
    monkeypatch.setattr(collector,"load_datasets",lambda paths: rows)
    monkeypatch.setattr(collector,"build_runtime",FakeRuntime)
    monkeypatch.setattr(collector,"route",lambda text:type("R",(),{"mode":"fast"})())

    pilot=asyncio.run(collector.collect(_args(tmp_path,limit=2,checkpoint_every=1)))
    assert pilot["written"] == 2
    pilot_manifest=pilot["manifest"]
    assert pilot_manifest["selection"] == {
        "dataset_rows":3,
        "selected_rows":2,
        "limit":2,
    }

    full_args=_args(tmp_path,resume=True,limit=0,checkpoint_every=1)
    completed=asyncio.run(collector.collect(full_args))
    assert completed["skipped"] == 2
    assert completed["written"] == 1
    assert completed["manifest"]["selection"] == {
        "dataset_rows":3,
        "selected_rows":3,
        "limit":0,
    }
    saved=[
        json.loads(line)
        for line in (tmp_path / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [item["text"] for item in saved] == ["one","two","three"]



def test_resume_rejects_rows_outside_prior_deterministic_selection(tmp_path, monkeypatch):
    rows=[
        {"text":"one","expected":"fast","ood":False},
        {"text":"two","expected":"fast","ood":False},
        {"text":"three","expected":"fast","ood":False},
    ]
    monkeypatch.setattr(collector,"load_datasets",lambda paths: rows)
    monkeypatch.setattr(collector,"build_runtime",FakeRuntime)
    monkeypatch.setattr(collector,"route",lambda text:type("R",(),{"mode":"fast"})())

    pilot=asyncio.run(collector.collect(_args(tmp_path,limit=2,checkpoint_every=1)))
    path=tmp_path / "results.jsonl"
    manifest_path=tmp_path / "results.jsonl.manifest.json"

    saved=[
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    saved[1]["text"]="three"
    collector._write_atomic_jsonl(path,saved)

    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["results_sha256"]=collector.file_sha256(path)
    manifest["result_rows"]=2
    collector._write_atomic_json(manifest_path,manifest)

    try:
        asyncio.run(collector.collect(_args(tmp_path,resume=True,limit=0,checkpoint_every=1)))
        assert False
    except ValueError as exc:
        assert "prior deterministic selection" in str(exc)



def test_resume_rejects_collecting_manifest_invalid_code_revision(tmp_path, monkeypatch):
    dataset=tmp_path / "dataset.jsonl"
    dataset.write_text('{"text":"one","expected":"fast","ood":false}\n',encoding="utf-8")
    path=tmp_path / "results.jsonl"
    path.write_text(
        '{"text":"one","cde":"fast","confidence":0.9,"abstained":false,"models_used":["fake-model"]}\n',
        encoding="utf-8",
    )
    manifest=collector._runtime_manifest(FakeRuntime(),[str(dataset)])
    manifest["code_revision"]="NOT-A-SHA"
    manifest["results_sha256"]=collector.file_sha256(path)
    manifest["result_rows"]=1
    (tmp_path / "results.jsonl.manifest.json").write_text(json.dumps(manifest),encoding="utf-8")
    monkeypatch.setattr(collector,"build_runtime",FakeRuntime)

    args=_args(tmp_path,resume=True)
    args.dataset=[str(dataset)]
    try:
        asyncio.run(collector.collect(args))
        assert False
    except ValueError as exc:
        assert "code provenance" in str(exc)



def test_resume_allows_null_code_revision_when_current_environment_is_unresolved(tmp_path, monkeypatch):
    dataset=tmp_path / "dataset.jsonl"
    dataset.write_text('{"text":"one","expected":"fast","ood":false}\n',encoding="utf-8")
    monkeypatch.setattr(collector,"build_runtime",FakeRuntime)
    monkeypatch.setattr(collector,"route",lambda text:type("R",(),{"mode":"fast"})())
    monkeypatch.setattr(collector,"_code_revision",lambda:None)
    monkeypatch.setattr(collector,"_code_dirty",lambda:None)

    args=_args(tmp_path)
    args.dataset=[str(dataset)]
    first=asyncio.run(collector.collect(args))
    assert first["manifest"]["code_revision"] is None
    assert first["manifest"]["code_dirty"] is None

    resume_args=_args(tmp_path,resume=True)
    resume_args.dataset=[str(dataset)]
    second=asyncio.run(collector.collect(resume_args))
    assert second["written"] == 0
    assert second["skipped"] == 1
    assert second["manifest"]["code_revision"] is None
    assert second["manifest"]["code_dirty"] is None



def test_resume_rejects_changed_tracked_worktree_state(tmp_path, monkeypatch):
    dataset=tmp_path / "dataset.jsonl"
    dataset.write_text('{"text":"one","expected":"fast","ood":false}\n',encoding="utf-8")
    monkeypatch.setattr(collector,"build_runtime",FakeRuntime)
    monkeypatch.setattr(collector,"route",lambda text:type("R",(),{"mode":"fast"})())
    monkeypatch.setattr(collector,"_code_revision",lambda:"a" * 40)
    monkeypatch.setattr(collector,"_code_dirty",lambda:False)

    args=_args(tmp_path)
    args.dataset=[str(dataset)]
    first=asyncio.run(collector.collect(args))
    assert first["manifest"]["code_revision"] == "a" * 40
    assert first["manifest"]["code_dirty"] is False

    monkeypatch.setattr(collector,"_code_dirty",lambda:True)
    resume_args=_args(tmp_path,resume=True)
    resume_args.dataset=[str(dataset)]
    try:
        asyncio.run(collector.collect(resume_args))
        assert False
    except ValueError as exc:
        assert "code_dirty provenance does not match current runtime" in str(exc)
