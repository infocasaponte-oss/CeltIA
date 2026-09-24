#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

from celtia.decision.evaluation import ShadowSample, evaluate_shadow, promotion_gate

ROUTES={"fast","think","code","agent","long"}

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

def load_cde_results(path: Path) -> dict[str, dict]:
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
    by_text=load_cde_results(Path(args.cde_results)) if args.cde_results else {}
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
