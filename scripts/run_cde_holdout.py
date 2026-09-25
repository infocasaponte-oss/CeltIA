#!/usr/bin/env python3
"""Sealed holdout validation for the CDE route + OOD contract.

Collects live results for a frozen dataset from a clean checkout, then evaluates
the pre-registered acceptance criteria exactly once, on aggregate numbers only.
API/network failures are retried with ``--resume`` on the same output file.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROUTES = ("fast", "think", "code", "agent", "long")

# Pre-registered thresholds. Do not adjust after seeing results.
CRITERIA = {
    "in_domain_accuracy": 0.98,
    "each_route_accuracy": 0.90,
    "ood_recall": 0.80,
    "ood_specificity": 0.85,
    "ood_precision": 0.80,
}


def evaluate(rows: list[dict]) -> dict:
    """Aggregate metrics and acceptance checks. Pure function over collected rows."""
    in_domain = [r for r in rows if not r["expected_ood"]]
    if not in_domain or not any(r["expected_ood"] for r in rows):
        raise ValueError("holdout needs both in-domain and OOD rows")
    per_route_total = Counter(r["expected"] for r in in_domain)
    per_route_ok = Counter(r["expected"] for r in in_domain if r["cde"] == r["expected"])
    route_accuracy = {route: per_route_ok[route] / per_route_total[route] for route in sorted(per_route_total)}
    accuracy = sum(r["cde"] == r["expected"] for r in in_domain) / len(in_domain)

    tp = sum(bool(r["suspected_ood"]) and bool(r["expected_ood"]) for r in rows)
    fn = sum((not r["suspected_ood"]) and bool(r["expected_ood"]) for r in rows)
    fp = sum(bool(r["suspected_ood"]) and not r["expected_ood"] for r in rows)
    tn = sum((not r["suspected_ood"]) and not r["expected_ood"] for r in rows)
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    ties = sum(abs(float(r["ood_probability"]) - 0.5) < 1e-12 for r in rows)

    checks = {
        "in_domain_accuracy": accuracy >= CRITERIA["in_domain_accuracy"],
        "each_route_accuracy": min(route_accuracy.values()) >= CRITERIA["each_route_accuracy"],
        "ood_recall": recall >= CRITERIA["ood_recall"],
        "ood_specificity": specificity >= CRITERIA["ood_specificity"],
        "ood_precision": precision >= CRITERIA["ood_precision"],
        "no_logit_ties": ties == 0,
    }
    return {
        "rows": len(rows),
        "in_domain_accuracy": accuracy,
        "route_accuracy": route_accuracy,
        "ood": {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "recall": recall,
                "specificity": specificity, "precision": precision, "ties": ties},
        "criteria": CRITERIA,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()


def _collect(dataset: Path, output: Path, attempts: int) -> None:
    for attempt in range(1, attempts + 1):
        cmd = [sys.executable, "scripts/collect_decision_eval_results.py", "--dataset", str(dataset),
               "--output", str(output), "--checkpoint-every", "10", "--sleep-seconds", "0.05"]
        if attempt > 1:
            cmd.append("--resume")
        if subprocess.run(cmd, cwd=ROOT, stdout=subprocess.DEVNULL).returncode == 0:
            return
        print(f"collector attempt {attempt}/{attempts} failed; resuming the same output", file=sys.stderr)
        time.sleep(15)
    raise SystemExit("collection did not complete; rerun with the same --output to resume")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=0, help="Fail unless exactly this many rows exist.")
    parser.add_argument("--attempts", type=int, default=5)
    args = parser.parse_args()

    if _git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("Tracked checkout is dirty; commit before a sealed run.")
    sha = _git("rev-parse", "HEAD")
    print(f"git revision {sha}", file=sys.stderr)

    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    _collect(args.dataset, output, args.attempts)

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.expected_rows and len(rows) != args.expected_rows:
        raise SystemExit(f"expected {args.expected_rows} rows, found {len(rows)}")
    report = {"git_revision": sha, **evaluate(rows)}
    output.with_suffix(".evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
