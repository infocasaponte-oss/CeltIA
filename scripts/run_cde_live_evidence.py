#!/usr/bin/env python3
"""Run a reproducible live CDE evidence collection from a local checkout."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from core.config import settings


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path("results/cde_routes.jsonl")
DATASETS = (
    Path("benchmarks/decision_routes.jsonl"),
    Path("benchmarks/decision_routes_ood.jsonl"),
)


def _run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        check=check,
    )


def _require_clean_checkout() -> str:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    if dirty:
        raise SystemExit(
            "Tracked checkout is dirty; commit or revert tracked changes before collecting promotion evidence."
        )
    return revision


def _require_live_credential() -> None:
    if not settings.primary_llm_key:
        raise SystemExit(
            "No live provider credential found in environment, .env, or .env.local "
            "(expected LLM_PRIMARY_API_KEY, XAI_API_KEY, or IMAGE_API_KEY)."
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect and evaluate the 200-case live CDE evidence set from a clean local checkout."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing compatible result/manifest pair instead of starting fresh.",
    )
    args = parser.parse_args()

    revision = _require_clean_checkout()
    _require_live_credential()

    output = args.output
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)

    collector = [
        sys.executable,
        "scripts/collect_decision_eval_results.py",
    ]
    for dataset in DATASETS:
        collector.extend(["--dataset", str(dataset)])
    collector.extend(
        [
            "--output",
            str(output),
            "--checkpoint-every",
            str(args.checkpoint_every),
            "--sleep-seconds",
            str(args.sleep_seconds),
        ]
    )
    if args.resume:
        collector.append("--resume")

    print(f"Collecting live CDE evidence at git revision {revision}.")
    print(f"Output: {output}")
    _run(*collector)

    evaluator = [
        sys.executable,
        "scripts/evaluate_decision_routes.py",
    ]
    for dataset in DATASETS:
        evaluator.extend(["--dataset", str(dataset)])
    evaluator.extend(
        [
            "--cde-results",
            str(output),
            "--require-eligible",
        ]
    )

    print("Evaluating promotion gate.")
    result = _run(*evaluator, check=False)
    if result.returncode:
        print(
            f"Promotion gate did not pass (exit {result.returncode}); "
            "the collected JSONL and v4 manifest remain available for inspection.",
            file=sys.stderr,
        )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
