#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from core.config import settings
from core.decision_runtime import CeltIADecisionRuntime
from core.inference import build_llm
from core.router import route
from scripts.evaluate_decision_routes import (
    RESULT_FORMAT_VERSION,
    SHA256_RE,
    dataset_sha256,
    declared_backend_models,
    file_sha256,
    load_cde_results,
    load_jsonl,
    parse_aware_timestamp,
    validate_backend_provenance,
    validate_code_provenance,
    validate_policy_provenance,
    validate_result_models,
    validate_results_manifest,
    validate_selection_provenance,
)

ROUTES=("fast","think","code","agent","long")
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
    results,usage=await runtime.decide_with_usage(
        {"user_message": text[-12000:], "heuristic_route": heuristic},
        [{
            "id":"route",
            "prompt":"Select the most appropriate CeltIA execution route.",
            "type":"choice",
            "options":list(ROUTES),
        }],
    )
    result=results[0]
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
        "models_used":list(usage.get("models") or []),
    }


def _code_revision() -> str | None:
    try:
        completed=subprocess.run(
            ["git","rev-parse","HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError,subprocess.SubprocessError):
        return None
    value=completed.stdout.strip()
    return value if SHA256_RE.fullmatch(value) or re.fullmatch(r"^[0-9a-f]{40}$",value) else None


def _code_dirty() -> bool | None:
    try:
        completed=subprocess.run(
            ["git","status","--porcelain","--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError,subprocess.SubprocessError):
        return None
    return bool(completed.stdout.strip())


def _load_manifest(path: Path) -> dict:
    try:
        value=json.loads(path.read_text(encoding="utf-8"))
    except (OSError,json.JSONDecodeError) as exc:
        raise ValueError(f"invalid evaluation manifest: {path}") from exc
    if not isinstance(value,dict) or value.get("format_version") != RESULT_FORMAT_VERSION:
        raise ValueError(f"incompatible evaluation manifest: {path}")
    return value


def _validate_collecting_manifest(manifest: dict, datasets: list[str], results_path: Path, *, dataset_rows: int) -> None:
    if manifest.get("status") != "collecting":
        raise ValueError("resume manifest has invalid collection status")
    manifest_datasets=manifest.get("datasets")
    if (
        not isinstance(manifest_datasets,list)
        or any(not isinstance(item,str) or not item for item in manifest_datasets)
        or manifest_datasets != datasets
    ):
        raise ValueError("resume manifest dataset list does not match selected datasets")
    collected_at=manifest.get("collected_at")
    collected_at_value=parse_aware_timestamp(collected_at)
    if collected_at_value is None:
        raise ValueError("resume manifest has invalid collected_at")
    resumed_from_collected_at=manifest.get("resumed_from_collected_at")
    if resumed_from_collected_at is not None:
        resumed_from_value=parse_aware_timestamp(resumed_from_collected_at)
        if resumed_from_value is None:
            raise ValueError("resume manifest has invalid resumed_from_collected_at")
        if resumed_from_value > collected_at_value:
            raise ValueError("resume manifest root timestamp is after collected_at")
    if not validate_backend_provenance(manifest.get("backend")):
        raise ValueError("resume manifest has invalid backend provenance")
    if not validate_policy_provenance(manifest.get("policy")):
        raise ValueError("resume manifest has invalid policy provenance")
    if (
        "code_revision" not in manifest
        or "code_dirty" not in manifest
        or not validate_code_provenance(
            manifest.get("code_revision"),
            manifest.get("code_dirty"),
        )
    ):
        raise ValueError("resume manifest has invalid code provenance")
    selected_rows=validate_selection_provenance(
        manifest.get("selection"),
        dataset_rows=dataset_rows,
        require_full_selection=False,
    )
    expected_dataset_sha=manifest.get("dataset_sha256")
    if (
        not isinstance(expected_dataset_sha,str)
        or not SHA256_RE.fullmatch(expected_dataset_sha)
        or expected_dataset_sha != dataset_sha256(datasets)
    ):
        raise ValueError("resume dataset provenance does not match current datasets")
    expected_results_sha=manifest.get("results_sha256")
    if (
        not isinstance(expected_results_sha,str)
        or not SHA256_RE.fullmatch(expected_results_sha)
        or expected_results_sha != file_sha256(results_path)
    ):
        raise ValueError("resume collecting checkpoint results do not match provenance manifest")
    expected_rows=manifest.get("result_rows")
    if isinstance(expected_rows,bool) or not isinstance(expected_rows,int) or expected_rows < 0:
        raise ValueError("resume collecting checkpoint has invalid result_rows")
    actual_rows=sum(
        1
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if expected_rows != actual_rows:
        raise ValueError("resume collecting checkpoint row count does not match provenance manifest")
    if expected_rows > selected_rows:
        raise ValueError("resume collecting checkpoint exceeds declared selection")


def _runtime_manifest(runtime: CeltIADecisionRuntime, datasets: list[str], *, dataset_rows: int | None = None, selected_rows: int | None = None, limit: int = 0) -> dict:
    llm=runtime.llm
    primary=getattr(llm,"primary",None)
    fallback=getattr(llm,"fallback",None)
    if dataset_rows is None:
        dataset_rows=len(load_datasets(datasets))
    if selected_rows is None:
        selected_rows=dataset_rows if limit == 0 else min(limit,dataset_rows)
    return {
        "format_version":RESULT_FORMAT_VERSION,
        "status":"collecting",
        "collected_at":datetime.now(timezone.utc).isoformat(),
        "dataset_sha256":dataset_sha256(datasets),
        "code_revision":_code_revision(),
        "code_dirty":_code_dirty(),
        "datasets":datasets,
        "selection":{
            "dataset_rows":dataset_rows,
            "selected_rows":selected_rows,
            "limit":limit,
        },
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
    all_rows=load_datasets(datasets)
    dataset_rows=len(all_rows)
    rows=all_rows[:args.limit] if args.limit else all_rows
    selected_rows=len(rows)

    output=Path(args.output)
    manifest_output=Path(str(output) + ".manifest.json")
    existing={}
    resume_manifest=None
    if output.exists() and not args.resume:
        raise ValueError("output already exists; use --resume or choose a new output path")
    if args.resume and output.exists():
        if not manifest_output.exists():
            raise ValueError("resume requires the evaluation provenance manifest")
        resume_manifest=_load_manifest(manifest_output)
        if resume_manifest.get("status") == "complete":
            try:
                resume_manifest=validate_results_manifest(output,datasets,require_full_selection=False,require_clean_code=False,dataset_rows=dataset_rows)
            except ValueError as exc:
                raise ValueError(f"resume completed provenance manifest is invalid: {exc}") from exc
        else:
            _validate_collecting_manifest(resume_manifest,datasets,output,dataset_rows=dataset_rows)
        existing=load_cde_results(
            output,
            require_models_used=True,
            allowed_models=declared_backend_models(resume_manifest["backend"]),
        )
        previous_selection=resume_manifest["selection"]
        previous_selected_rows=previous_selection["selected_rows"]
        previous_selected_texts={row["text"] for row in all_rows[:previous_selected_rows]}
        unexpected_previous=set(existing)-previous_selected_texts
        if unexpected_previous:
            first=sorted(unexpected_previous)[0]
            raise ValueError(
                f"existing result text not present in prior deterministic selection: {first!r}"
            )
        unknown=set(existing)-{row["text"] for row in rows}
        if unknown:
            first=sorted(unknown)[0]
            raise ValueError(f"existing result text not present in selected datasets: {first!r}")

    runtime=build_runtime()
    manifest=_runtime_manifest(runtime,datasets,dataset_rows=dataset_rows,selected_rows=selected_rows,limit=args.limit)
    if resume_manifest is not None:
        if resume_manifest.get("dataset_sha256") != manifest["dataset_sha256"]:
            raise ValueError("resume dataset provenance does not match current datasets")
        for field in ("backend","policy","code_revision","code_dirty"):
            if resume_manifest.get(field) != manifest[field]:
                raise ValueError(f"resume {field} provenance does not match current runtime")
        manifest["resumed_from_collected_at"]=(
            resume_manifest.get("resumed_from_collected_at")
            or resume_manifest.get("collected_at")
        )
    written=0
    skipped=0
    output.parent.mkdir(parents=True,exist_ok=True)
    if resume_manifest is None:
        # Persist provenance before the first checkpoint so an interrupted fresh
        # collection remains resumable once any result checkpoint exists.
        _write_atomic_json(manifest_output,manifest)
    collected=[existing[row["text"]] for row in rows if row["text"] in existing]
    pending_since_checkpoint=0

    for row in rows:
        if row["text"] in existing:
            skipped+=1
            continue
        item=await collect_one(runtime,row)
        validate_result_models(
            item.get("models_used"),
            require_models_used=True,
            allowed_models=declared_backend_models(manifest["backend"]),
        )
        collected.append(item)
        written+=1
        pending_since_checkpoint+=1
        if pending_since_checkpoint >= args.checkpoint_every:
            _write_atomic_jsonl(output,collected)
            manifest["results_sha256"]=file_sha256(output)
            manifest["result_rows"]=len(collected)
            _write_atomic_json(manifest_output,manifest)
            pending_since_checkpoint=0
        if args.sleep_seconds:
            await asyncio.sleep(args.sleep_seconds)

    # Always materialize a complete valid checkpoint, including zero-write resume runs.
    _write_atomic_jsonl(output,collected)
    manifest["status"]="complete"
    manifest["results_sha256"]=file_sha256(output)
    manifest["result_rows"]=len(collected)
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
