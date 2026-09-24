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
        self.engine_options={"abstain_below":.55,"temperature":1.0}

    async def decide(self, context, questions):
        return [FakeResult()]


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
    assert not list(tmp_path.glob("results.jsonl.*.tmp"))
    manifest_path=tmp_path / "results.jsonl.manifest.json"
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["format_version"] == 2
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
    assert manifest["format_version"] == 2
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
