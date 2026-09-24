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
