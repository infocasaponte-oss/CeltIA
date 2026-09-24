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
        if not isinstance(row.get("text"),str) or row.get("expected") not in {"fast","think","code","agent","long"}:
            raise ValueError(f"invalid benchmark row {n}")
        rows.append(row)
    return rows

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",default="benchmarks/decision_routes.jsonl")
    p.add_argument("--cde-results",help="Optional JSONL with text,cde,confidence,abstained")
    args=p.parse_args()
    rows=load_jsonl(Path(args.dataset))
    by_text={}
    if args.cde_results:
        for line in Path(args.cde_results).read_text(encoding="utf-8").splitlines():
            if line.strip():
                item=json.loads(line); by_text[item["text"]]=item
    samples=[]
    for row in rows:
        h=route(row["text"]).mode
        item=by_text.get(row["text"])
        samples.append(ShadowSample(h,item.get("cde") if item else None,
            float(item.get("confidence",0.0)) if item else 0.0,
            bool(item.get("abstained",True)) if item else True,row["expected"]))
    metrics=evaluate_shadow(samples)
    print(json.dumps({"metrics":metrics,"gate":promotion_gate(metrics)},indent=2,ensure_ascii=False))

if __name__=="__main__":
    main()
