#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path

from core.router import route
from celtia.decision.evaluation import ShadowSample, evaluate_shadow, promotion_gate

def load_jsonl(path: Path):
    rows=[]
    for n,line in enumerate(path.read_text(encoding="utf-8").splitlines(),1):
        if not line.strip(): continue
        row=json.loads(line)
        expected=row.get("expected")
        expected_ood=row.get("ood", False)
        if not isinstance(row.get("text"),str) or not row["text"].strip():
            raise ValueError(f"invalid benchmark row {n}")
        if not isinstance(expected_ood, bool):
            raise ValueError(f"invalid benchmark row {n}")
        if expected not in {"fast","think","code","agent","long"}:
            if not (expected is None and expected_ood):
                raise ValueError(f"invalid benchmark row {n}")
        rows.append(row)
    return rows

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",default="benchmarks/decision_routes.jsonl")
    p.add_argument("--cde-results",help="Optional JSONL with text,cde,confidence,abstained")
    p.add_argument("--min-samples",type=int,default=200)
    p.add_argument("--min-coverage",type=float,default=.80)
    p.add_argument("--min-labeled",type=int,default=50)
    p.add_argument("--min-accuracy-delta",type=float,default=.02)
    p.add_argument("--min-labeled-coverage",type=float,default=.80)
    p.add_argument("--require-eligible",action="store_true",help="Exit non-zero when the promotion gate fails")
    args=p.parse_args()
    rows=load_jsonl(Path(args.dataset))
    by_text={}
    if args.cde_results:
        for line in Path(args.cde_results).read_text(encoding="utf-8").splitlines():
            if line.strip():
                item=json.loads(line)
                if not isinstance(item.get("text"), str):
                    raise ValueError("invalid CDE result row")
                if "suspected_ood" in item and not isinstance(item["suspected_ood"], bool):
                    raise ValueError("invalid CDE result row")
                by_text[item["text"]]=item
    samples=[]
    for row in rows:
        h=route(row["text"]).mode
        item=by_text.get(row["text"])
        samples.append(ShadowSample(
            h,
            item.get("cde") if item else None,
            float(item.get("confidence",0.0)) if item else 0.0,
            bool(item.get("abstained",True)) if item else True,
            row.get("expected"),
            suspected_ood=item.get("suspected_ood") if item else None,
            expected_ood=row.get("ood", False),
        ))
    metrics=evaluate_shadow(samples)
    gate=promotion_gate(
        metrics,
        min_samples=args.min_samples,
        min_coverage=args.min_coverage,
        min_labeled=args.min_labeled,
        min_accuracy_delta=args.min_accuracy_delta,
        min_labeled_coverage=args.min_labeled_coverage,
    )
    print(json.dumps({"metrics":metrics,"gate":gate},indent=2,ensure_ascii=False))
    if args.require_eligible and not gate["eligible"]:
        raise SystemExit(2)

if __name__=="__main__":
    main()
