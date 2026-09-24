#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import math
import hashlib
import re
from datetime import datetime
from pathlib import Path

from celtia.decision.evaluation import ShadowSample, evaluate_shadow, promotion_gate

ROUTES={"fast","think","code","agent","long"}
RESULT_FORMAT_VERSION=3
SHA256_RE=re.compile(r"^[0-9a-f]{64}$")


def parse_aware_timestamp(value: object) -> datetime | None:
    if not isinstance(value,str) or not value.strip():
        return None
    try:
        parsed=datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed


def valid_aware_timestamp(value: object) -> bool:
    return parse_aware_timestamp(value) is not None


def validate_backend_provenance(value: object) -> bool:
    if not isinstance(value,dict):
        return False
    required={"client_type","model","primary_model","fallback_model"}
    if not required.issubset(value):
        return False
    if not isinstance(value.get("client_type"),str) or not value["client_type"].strip():
        return False
    for field in ("model","primary_model","fallback_model"):
        model=value.get(field)
        if model is not None and (not isinstance(model,str) or not model.strip()):
            return False
    return True


def validate_policy_provenance(value: object) -> bool:
    if not isinstance(value,dict):
        return False
    required={
        "abstain_below",
        "temperature",
        "reject_suspected_ood",
        "ood_entropy_threshold",
        "ood_margin_threshold",
    }
    if not required.issubset(value):
        return False
    if not isinstance(value.get("reject_suspected_ood"),bool):
        return False
    for field in ("abstain_below","ood_entropy_threshold","ood_margin_threshold"):
        raw=value.get(field)
        if isinstance(raw,bool):
            return False
        try:
            number=float(raw)
        except (TypeError,ValueError,OverflowError):
            return False
        if not math.isfinite(number) or not 0 <= number <= 1:
            return False
    temperature=value.get("temperature")
    if isinstance(temperature,bool):
        return False
    try:
        temperature_value=float(temperature)
    except (TypeError,ValueError,OverflowError):
        return False
    return math.isfinite(temperature_value) and temperature_value > 0


def validate_selection_provenance(
    value: object,
    *,
    dataset_rows: int,
    require_full_selection: bool,
) -> int:
    if not isinstance(value,dict):
        raise ValueError("CDE results manifest has invalid selection provenance")
    selected_rows=value.get("selected_rows")
    manifest_dataset_rows=value.get("dataset_rows")
    limit=value.get("limit")
    for raw in (selected_rows,manifest_dataset_rows,limit):
        if isinstance(raw,bool) or not isinstance(raw,int) or raw < 0:
            raise ValueError("CDE results manifest has invalid selection provenance")
    if manifest_dataset_rows != dataset_rows:
        raise ValueError("CDE results manifest selection dataset_rows does not match selected datasets")
    expected_selected=dataset_rows if limit == 0 else min(limit,dataset_rows)
    if selected_rows != expected_selected:
        raise ValueError("CDE results manifest selection does not match its limit")
    if require_full_selection and selected_rows != dataset_rows:
        raise ValueError("CDE promotion requires a full-dataset collection")
    return selected_rows


def file_sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_sha256(paths: list[str]) -> str:
    digest=hashlib.sha256()
    for raw_path in paths:
        path=Path(raw_path)
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def validate_results_manifest(results_path: Path, datasets: list[str], *, require_full_selection: bool = True, dataset_rows: int | None = None) -> dict:
    manifest_path=Path(str(results_path) + ".manifest.json")
    try:
        manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError,json.JSONDecodeError) as exc:
        raise ValueError(f"missing or invalid CDE results manifest: {manifest_path}") from exc
    if not isinstance(manifest,dict) or manifest.get("format_version") != RESULT_FORMAT_VERSION:
        raise ValueError(f"incompatible CDE results manifest: {manifest_path}")
    manifest_datasets=manifest.get("datasets")
    if (
        not isinstance(manifest_datasets,list)
        or any(not isinstance(item,str) or not item for item in manifest_datasets)
        or manifest_datasets != datasets
    ):
        raise ValueError("CDE results manifest dataset list does not match selected datasets")
    collected_at=manifest.get("collected_at")
    collected_at_value=parse_aware_timestamp(collected_at)
    if collected_at_value is None:
        raise ValueError("CDE results manifest has invalid collected_at")
    resumed_from_collected_at=manifest.get("resumed_from_collected_at")
    if resumed_from_collected_at is not None:
        resumed_from_value=parse_aware_timestamp(resumed_from_collected_at)
        if resumed_from_value is None:
            raise ValueError("CDE results manifest has invalid resumed_from_collected_at")
        if resumed_from_value > collected_at_value:
            raise ValueError("CDE results manifest resume timestamp is after collected_at")
    if not validate_backend_provenance(manifest.get("backend")):
        raise ValueError("CDE results manifest has invalid backend provenance")
    if not validate_policy_provenance(manifest.get("policy")):
        raise ValueError("CDE results manifest has invalid policy provenance")
    expected_dataset_sha=manifest.get("dataset_sha256")
    if (
        not isinstance(expected_dataset_sha,str)
        or not SHA256_RE.fullmatch(expected_dataset_sha)
        or expected_dataset_sha != dataset_sha256(datasets)
    ):
        raise ValueError("CDE results manifest dataset provenance does not match selected datasets")
    if dataset_rows is None:
        dataset_rows=sum(
            1
            for raw_path in datasets
            for line in Path(raw_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    selected_rows=validate_selection_provenance(
        manifest.get("selection"),
        dataset_rows=dataset_rows,
        require_full_selection=require_full_selection,
    )
    if manifest.get("status") != "complete":
        raise ValueError("CDE results manifest is not complete")
    expected_results_sha=manifest.get("results_sha256")
    if (
        not isinstance(expected_results_sha,str)
        or not SHA256_RE.fullmatch(expected_results_sha)
        or expected_results_sha != file_sha256(results_path)
    ):
        raise ValueError("CDE results file does not match its manifest")
    expected_rows=manifest.get("result_rows")
    if isinstance(expected_rows,bool) or not isinstance(expected_rows,int) or expected_rows < 0:
        raise ValueError("CDE results manifest has invalid result_rows")
    actual_rows=sum(1 for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip())
    if expected_rows != actual_rows:
        raise ValueError("CDE results row count does not match its manifest")
    if expected_rows != selected_rows:
        raise ValueError("CDE completed results do not cover the declared selection")
    return manifest

def load_jsonl(path: Path):
    rows=[]
    for n,line in enumerate(path.read_text(encoding="utf-8").splitlines(),1):
        if not line.strip(): continue
        row=json.loads(line)
        expected=row.get("expected")
        expected_ood=row.get("ood")
        if not isinstance(row.get("text"),str) or not row["text"].strip():
            raise ValueError(f"invalid benchmark row {path}:{n}")
        if expected_ood is not None and not isinstance(expected_ood, bool):
            raise ValueError(f"invalid benchmark row {path}:{n}")
        if expected not in ROUTES:
            if not (expected is None and expected_ood is True):
                raise ValueError(f"invalid benchmark row {path}:{n}")
        rows.append(row)
    return rows

def load_cde_results(path: Path, *, require_models_used: bool = False) -> dict[str, dict]:
    by_text={}
    for n,line in enumerate(path.read_text(encoding="utf-8").splitlines(),1):
        if not line.strip():
            continue
        item=json.loads(line)
        text=item.get("text")
        cde=item.get("cde")
        confidence=item.get("confidence")
        abstained=item.get("abstained")
        suspected_ood=item.get("suspected_ood")
        models_used=item.get("models_used")

        if models_used is not None or require_models_used:
            if (
                not isinstance(models_used,list)
                or any(not isinstance(model,str) or not model.strip() for model in models_used)
                or len(models_used) != len(set(models_used))
            ):
                raise ValueError(f"invalid CDE result models_used {path}:{n}")

        if not isinstance(text,str) or not text.strip():
            raise ValueError(f"invalid CDE result row {path}:{n}")
        if text in by_text:
            raise ValueError(f"duplicate CDE result text {path}:{n}")
        if not isinstance(abstained,bool):
            raise ValueError(f"invalid CDE result row {path}:{n}")
        if isinstance(confidence,bool):
            raise ValueError(f"invalid CDE result row {path}:{n}")
        try:
            confidence_value=float(confidence)
        except (TypeError,ValueError,OverflowError) as exc:
            raise ValueError(f"invalid CDE result row {path}:{n}") from exc
        if not math.isfinite(confidence_value) or not 0 <= confidence_value <= 1:
            raise ValueError(f"invalid CDE result row {path}:{n}")
        if suspected_ood is not None and not isinstance(suspected_ood,bool):
            raise ValueError(f"invalid CDE result row {path}:{n}")
        if abstained:
            if cde is not None:
                raise ValueError(f"invalid CDE result row {path}:{n}")
        elif cde not in ROUTES:
            raise ValueError(f"invalid CDE result row {path}:{n}")

        by_text[text]={
            **item,
            "confidence":confidence_value,
        }
    return by_text


def main():
    # Keep the reusable JSONL validation helpers dependency-free. Routing needs
    # application settings, so import it only when the CLI actually runs.
    from core.router import route

    p=argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        action="append",
        help="Benchmark JSONL. Repeat to combine route and OOD datasets.",
    )
    p.add_argument(
        "--cde-results",
        help="Optional JSONL with text,cde,confidence,abstained and optional suspected_ood",
    )
    p.add_argument("--min-samples",type=int,default=200)
    p.add_argument("--min-coverage",type=float,default=.80)
    p.add_argument("--min-labeled",type=int,default=50)
    p.add_argument("--min-accuracy-delta",type=float,default=.02)
    p.add_argument("--min-labeled-coverage",type=float,default=.80)
    p.add_argument("--min-route-labeled",type=int,default=10)
    p.add_argument("--min-route-coverage",type=float,default=.70)
    p.add_argument("--min-route-accuracy",type=float,default=.60)
    p.add_argument("--min-ood-labeled",type=int,default=20)
    p.add_argument("--min-ood-coverage",type=float,default=.80)
    p.add_argument("--min-ood-recall",type=float,default=.80)
    p.add_argument("--max-ood-false-positive-rate",type=float,default=.20)
    p.add_argument("--require-eligible",action="store_true",help="Exit non-zero when the promotion gate fails")
    args=p.parse_args()
    datasets=args.dataset or ["benchmarks/decision_routes.jsonl"]
    rows=[]
    seen_text=set()
    for dataset in datasets:
        for row in load_jsonl(Path(dataset)):
            if row["text"] in seen_text:
                raise ValueError(f"duplicate benchmark text across datasets: {row['text']!r}")
            seen_text.add(row["text"])
            rows.append(row)
    if args.cde_results:
        results_path=Path(args.cde_results)
        validate_results_manifest(results_path,datasets,dataset_rows=len(rows))
        by_text=load_cde_results(results_path,require_models_used=True)
    else:
        by_text={}
    unknown_results=set(by_text)-seen_text
    if unknown_results:
        first=sorted(unknown_results)[0]
        raise ValueError(f"CDE result text not present in benchmark datasets: {first!r}")
    samples=[]
    for row in rows:
        h=route(row["text"]).mode
        item=by_text.get(row["text"])
        samples.append(ShadowSample(
            h,
            item.get("cde") if item else None,
            item["confidence"] if item else 0.0,
            item["abstained"] if item else True,
            row.get("expected"),
            suspected_ood=item.get("suspected_ood") if item else None,
            expected_ood=row.get("ood"),
        ))
    metrics=evaluate_shadow(samples)
    gate=promotion_gate(
        metrics,
        min_samples=args.min_samples,
        min_coverage=args.min_coverage,
        min_labeled=args.min_labeled,
        min_accuracy_delta=args.min_accuracy_delta,
        min_labeled_coverage=args.min_labeled_coverage,
        min_route_labeled=args.min_route_labeled,
        min_route_coverage=args.min_route_coverage,
        min_route_accuracy=args.min_route_accuracy,
        min_ood_labeled=args.min_ood_labeled,
        min_ood_coverage=args.min_ood_coverage,
        min_ood_recall=args.min_ood_recall,
        max_ood_false_positive_rate=args.max_ood_false_positive_rate,
    )
    print(json.dumps({"metrics":metrics,"gate":gate},indent=2,ensure_ascii=False))
    if args.require_eligible and not gate["eligible"]:
        raise SystemExit(2)

if __name__=="__main__":
    main()
