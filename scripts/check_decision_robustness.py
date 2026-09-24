#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
from celtia.decision.calibration import multiclass_brier,top_label_calibration
from celtia.decision.metrics import expected_calibration_error
from celtia.decision.robustness import option_order_max_deviation,ood_signal

def main():
    p=argparse.ArgumentParser()
    p.add_argument("results",help="JSONL: expected, distributions[]")
    p.add_argument("--max-order-deviation",type=float,default=.10)
    args=p.parse_args()
    rows=[json.loads(x) for x in Path(args.results).read_text(encoding="utf-8").splitlines() if x.strip()]
    canonical=[r["distributions"][0] for r in rows]
    targets=[r["expected"] for r in rows]
    conf,correct=top_label_calibration(canonical,targets)
    report={
        "samples":len(rows),
        "multiclass_brier":multiclass_brier(canonical,targets),
        "ece":expected_calibration_error(conf,correct),
        "mean_order_deviation":sum(option_order_max_deviation(r["distributions"]) for r in rows)/len(rows) if rows else None,
        "suspected_ood":sum(ood_signal(d)["suspected_ood"] for d in canonical),
    }
    report["order_robustness_pass"]=report["mean_order_deviation"] is not None and report["mean_order_deviation"] <= args.max_order_deviation
    print(json.dumps(report,indent=2))

if __name__=="__main__": main()
