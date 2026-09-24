#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from core.config import settings
from core.decision_runtime import CeltIADecisionRuntime
from core.inference import build_llm
from core.router import route
from scripts.evaluate_decision_routes import load_cde_results, load_jsonl

ROUTES=("fast","think","code","agent","long")
RESULT_FORMAT_VERSION=2
DEFAULT_CHECKPOINT_EVERY=10


def build_runtime() -> CeltIADecisionRuntime:
    return CeltIADecisionRuntime(
        build_llm(),
        abstain_below=settings.decision_abstain_below,
        temperature=settings.decision_temperature,
        reject_suspected_ood=settings.decision_reject_ood,
        ood_entropy_threshold=settings.decision_ood_entropy_threshold,
        ood_margin_threshold=settings.decision_ood_margin_threshold,
        max_questions=settings.decision_max_questions,
        max_output_tokens=settings.decision_max_output_tokens,
        max_total_output_tokens=settings.decision_max_total_output_tokens,
        max_total_prompt_chars=settings.decision_max_total_prompt_chars,
        call_timeout_seconds=settings.decision_call_timeout_seconds,
        request_timeout_seconds=settings.decision_request_timeout_seconds,
    )


async def collect_one(runtime: CeltIADecisionRuntime, row: dict) -> dict:
    text=row["text"]
    heuristic=route(text).mode
    result=(await runtime.decide(
        {"user_message": text[-12000:], "heuristic_route": heuristic},
        [{
            "id":"route",
            "prompt":"Select the most appropriate CeltIA execution route.",
            "type":"choice",
            "options":list(ROUTES),
        }],
    ))[0]
    return {
        "text":text,
        "cde":result.decision,
        "confidence":result.confidence,
        "abstained":result.abstained,
        "suspected_ood":result.suspected_ood,
        "abstention_reason":result.abstention_reason,
        "normalized_entropy":result.normalized_entropy,
        "margin":result.margin,
        "heuristic":heuristic,
        "expected":row.get("expected"),
        "expected_ood":row.get("ood"),
    }


def _dataset_sha256(paths: list[str]) -> str:
    digest=hashlib.sha256()
    for raw_path in paths:
        path=Path(raw_path)
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _runtime_manifest(runtime: CeltIADecisionRuntime, datasets: list[str]) -> dict:
    llm=runtime.llm
    primary=getattr(llm,"primary",None)
    fallback=getattr(llm,"fallback",None)
    return {
        "format_version":RESULT_FORMAT_VERSION,
        "collected_at":datetime.now(timezone.utc).isoformat(),
        "dataset_sha256":_dataset_sha256(datasets),
        "datasets":datasets,
        "backend":{
            "client_type":type(llm).__name__,
            "model":getattr(llm,"model",None),
            "primary_model":getattr(primary,"model",None),
            "fallback_model":getattr(fallback,"model",None),
        },
        "policy":dict(runtime.engine_options),
    }


def load_datasets(paths: list[str]) -> list[dict]:
    rows=[]
    seen=set()
    for raw_path in paths:
        path=Path(raw_path)
        for row in load_jsonl(path):
            if row["text"] in seen:
                raise ValueError(f"duplicate benchmark text across datasets: {row['text']!r}")
            seen.add(row["text"])
            rows.append(row)
    return rows


def _fsync(handle) -> None:
    handle.flush()
    os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    if not hasattr(os, "O_DIRECTORY"):
        return
    try:
        fd=os.open(path,os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_atomic_json(output: Path, value: dict) -> None:
    output.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp_name=tempfile.mkstemp(prefix=output.name + ".",suffix=".tmp",dir=output.parent,text=True)
    tmp=Path(tmp_name)
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as handle:
            json.dump(value,handle,ensure_ascii=False,indent=2,sort_keys=True)
            handle.write("\n")
            _fsync(handle)
        os.replace(tmp,output)
        _fsync_directory(output.parent)
    except BaseException:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


def _write_atomic_jsonl(output: Path, rows: list[dict]) -> None:
    output.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp_name=tempfile.mkstemp(
        prefix=output.name + ".",
        suffix=".tmp",
        dir=output.parent,
        text=True,
    )
    tmp=Path(tmp_name)
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as handle:
            for item in rows:
                handle.write(json.dumps(item,ensure_ascii=False,separators=(",",":"))+"\n")
            _fsync(handle)
        os.replace(tmp,output)
        _fsync_directory(output.parent)
    except BaseException:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


async def collect(args) -> dict:
    datasets=args.dataset or [
        "benchmarks/decision_routes.jsonl",
        "benchmarks/decision_routes_ood.jsonl",
    ]
    rows=load_datasets(datasets)
    if args.limit:
        rows=rows[:args.limit]

    output=Path(args.output)
    existing={}
    if args.resume and output.exists():
        existing=load_cde_results(output)
        unknown=set(existing)-{row["text"] for row in rows}
        if unknown:
            first=sorted(unknown)[0]
            raise ValueError(f"existing result text not present in selected datasets: {first!r}")

    runtime=build_runtime()
    manifest=_runtime_manifest(runtime,datasets)
    manifest_output=Path(str(output) + ".manifest.json")
    written=0
    skipped=0
    output.parent.mkdir(parents=True,exist_ok=True)
    collected=[existing[row["text"]] for row in rows if row["text"] in existing]
    pending_since_checkpoint=0

    for row in rows:
        if row["text"] in existing:
            skipped+=1
            continue
        item=await collect_one(runtime,row)
        collected.append(item)
        written+=1
        pending_since_checkpoint+=1
        if pending_since_checkpoint >= args.checkpoint_every:
            _write_atomic_jsonl(output,collected)
            pending_since_checkpoint=0
        if args.sleep_seconds:
            await asyncio.sleep(args.sleep_seconds)

    # Always materialize a complete valid checkpoint, including zero-write resume runs.
    _write_atomic_jsonl(output,collected)
    _write_atomic_json(manifest_output,manifest)

    return {
        "format_version":RESULT_FORMAT_VERSION,
        "output":str(output),
        "manifest_output":str(manifest_output),
        "datasets":datasets,
        "selected":len(rows),
        "written":written,
        "skipped":skipped,
        "checkpoint_every":args.checkpoint_every,
        "manifest":manifest,
    }


def main() -> None:
    parser=argparse.ArgumentParser(
        description="Collect real CDE shadow-style routing results for offline promotion evaluation."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        help="Benchmark JSONL. Repeat to combine datasets; defaults to route + OOD suites.",
    )
    parser.add_argument("--output",default="decision_eval_results.jsonl")
    parser.add_argument("--limit",type=int,default=0,help="Optional deterministic prefix of samples; 0 means all.")
    parser.add_argument("--sleep-seconds",type=float,default=0.0,help="Delay between model calls.")
    parser.add_argument("--resume",action="store_true",help="Reuse valid existing rows and collect only missing texts.")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY,
        help="Atomically rewrite a complete valid checkpoint after this many new rows.",
    )
    args=parser.parse_args()
    if args.limit < 0:
        raise ValueError("limit must be non-negative")
    if not 0 <= args.sleep_seconds <= 60:
        raise ValueError("sleep-seconds must be between 0 and 60")
    if not 1 <= args.checkpoint_every <= 100:
        raise ValueError("checkpoint-every must be between 1 and 100")
    summary=asyncio.run(collect(args))
    print(json.dumps(summary,indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
