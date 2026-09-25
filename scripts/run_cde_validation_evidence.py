#!/usr/bin/env python3
"""Collect and validate the separate 80-case CDE robustness set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scripts.run_cde_live_evidence import (
    ROOT,
    _require_clean_checkout,
    _require_live_credential,
    _run,
)


DATASET=Path("benchmarks/holdout/decision_routes_holdout.jsonl")
DEFAULT_OUTPUT=Path("results/cde_routes_validation.jsonl")


def main() -> int:
    parser=argparse.ArgumentParser(
        description=(
            "Collect and evaluate the separate 80-case CDE validation set. "
            "This is a robustness check, not a replacement for the 200-case promotion gate."
        )
    )
    parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint-every",type=int,default=10)
    parser.add_argument("--sleep-seconds",type=float,default=0.05)
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()

    revision=_require_clean_checkout()
    _require_live_credential()

    output=args.output
    if not output.is_absolute():
        output=ROOT / output
    output.parent.mkdir(parents=True,exist_ok=True)

    collector=[
        sys.executable,
        "scripts/collect_decision_eval_results.py",
        "--dataset",
        str(DATASET),
        "--output",
        str(output),
        "--checkpoint-every",
        str(args.checkpoint_every),
        "--sleep-seconds",
        str(args.sleep_seconds),
    ]
    if args.resume:
        collector.append("--resume")

    print(f"Collecting separate validation evidence at git revision {revision}.")
    print(f"Output: {output}")
    _run(*collector)

    evaluator=[
        sys.executable,
        "scripts/evaluate_decision_routes.py",
        "--dataset",
        str(DATASET),
        "--cde-results",
        str(output),
        "--min-samples",
        "80",
        "--min-coverage",
        "0.80",
        "--min-labeled",
        "40",
        "--min-accuracy-delta",
        "-1",
        "--min-labeled-coverage",
        "0.80",
        "--min-route-labeled",
        "8",
        "--min-route-coverage",
        "0.70",
        "--min-route-accuracy",
        "0.60",
        "--min-ood-labeled",
        "80",
        "--min-ood-coverage",
        "0.80",
        "--min-ood-recall",
        "0.80",
        "--max-ood-false-positive-rate",
        "0.20",
        "--require-eligible",
    ]

    print("Evaluating separate validation thresholds.")
    result=_run(*evaluator,check=False)
    if result.returncode:
        print(
            f"Separate validation did not pass (exit {result.returncode}); "
            "the JSONL and manifest remain available for inspection.",
            file=sys.stderr,
        )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
