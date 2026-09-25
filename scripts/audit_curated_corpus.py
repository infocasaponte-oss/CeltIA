#!/usr/bin/env python3
"""Audit a downloaded CeltIA JSONL corpus before training.

Pure-stdlib, streaming, and safe for very large files. It computes:
- documents / bytes / malformed rows
- source and code-language distributions
- length percentiles from a bounded reservoir sample
- exact duplicate rate within the inspected sample
- Spanish-likeness / repetition heuristics for Spanish files
- suspicious path/license coverage for code
- per-file and global pass/fail signals

This is an audit layer, not a replacement for tokenizer-aware or MinHash near-dedup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path(r"D:\corpus_llm_grande")
SPANISH_STOPWORDS = {
    "de","la","que","el","en","y","a","los","del","se","las","por","un","para",
    "con","no","una","su","al","lo","como","más","pero","sus","le","ya","o","este",
    "sí","porque","esta","entre","cuando","muy","sin","sobre","también","me","hasta",
}

def percentile(values: list[int], q: float) -> int | None:
    if not values:
        return None
    vals = sorted(values)
    idx = min(len(vals) - 1, max(0, math.ceil(q * len(vals)) - 1))
    return vals[idx]

def spanish_signal(text: str) -> bool:
    sample = text[:12000].lower()
    words = re.findall(r"[a-záéíóúüñ]+", sample)
    if len(words) < 40:
        return False
    hits = sum(1 for w in words if w in SPANISH_STOPWORDS)
    accented = sum(sample.count(ch) for ch in "áéíóúüñ¿¡")
    alpha = sum(ch.isalpha() for ch in sample)
    printable = sum(ch.isprintable() or ch in "\n\t" for ch in sample)
    return (
        hits / len(words) >= 0.045
        and alpha / max(1, len(sample)) >= 0.45
        and printable / max(1, len(sample)) >= 0.97
        and (hits >= 4 or accented >= 2)
    )

def repeated_line_ratio(text: str) -> float:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 4:
        return 0.0
    return 1.0 - (len(set(lines)) / len(lines))

class Reservoir:
    def __init__(self, size: int, seed: int = 17):
        self.size = max(1, int(size))
        self.items: list[dict[str, Any]] = []
        self.seen = 0
        self.rng = random.Random(seed)

    def add(self, item: dict[str, Any]) -> None:
        self.seen += 1
        if len(self.items) < self.size:
            self.items.append(item)
            return
        j = self.rng.randrange(self.seen)
        if j < self.size:
            self.items[j] = item

def inspect_file(path: Path, sample_size: int) -> dict[str, Any]:
    reservoir = Reservoir(sample_size)
    docs = 0
    malformed = 0
    bytes_total = 0
    source_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    license_present = 0

    with path.open("rb") as fh:
        for raw in fh:
            bytes_total += len(raw)
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except Exception:
                malformed += 1
                continue
            docs += 1
            text = row.get("text")
            if not isinstance(text, str):
                text = ""
            source_counts[str(row.get("source") or "unknown")] += 1
            if row.get("language"):
                language_counts[str(row["language"]).lower()] += 1
            if row.get("license"):
                license_present += 1
            reservoir.add({
                "text": text,
                "source": row.get("source"),
                "language": row.get("language"),
                "license": row.get("license"),
                "path": row.get("path"),
            })

    sample = reservoir.items
    lengths = [len(x["text"]) for x in sample]
    empty = sum(1 for x in sample if not x["text"].strip())
    repeated = sum(1 for x in sample if repeated_line_ratio(x["text"]) > 0.35)

    hashes: Counter[str] = Counter()
    for x in sample:
        h = hashlib.blake2b(x["text"].encode("utf-8"), digest_size=16).hexdigest()
        hashes[h] += 1
    duplicate_docs = sum(n - 1 for n in hashes.values() if n > 1)

    is_spanish_file = "castellano" in {p.lower() for p in path.parts}
    spanish_ok = sum(1 for x in sample if spanish_signal(x["text"])) if is_spanish_file else None

    suspicious_code_paths = 0
    if not is_spanish_file:
        for x in sample:
            p = str(x.get("path") or "").lower().replace("\\", "/")
            if any(bit in p for bit in ("/vendor/", "/dist/", "/build/", "/node_modules/", ".min.js", ".map")):
                suspicious_code_paths += 1

    n = len(sample)
    malformed_rate = malformed / max(1, docs + malformed)
    empty_rate = empty / max(1, n)
    duplicate_rate = duplicate_docs / max(1, n)
    repetition_rate = repeated / max(1, n)
    spanish_rate = (spanish_ok / max(1, n)) if spanish_ok is not None else None
    license_rate = license_present / max(1, docs)
    suspicious_path_rate = suspicious_code_paths / max(1, n) if not is_spanish_file else None

    checks = {
        "malformed_rate_lte_0_1pct": malformed_rate <= 0.001,
        "empty_rate_lte_1pct": empty_rate <= 0.01,
        "sample_exact_duplicate_rate_lte_1pct": duplicate_rate <= 0.01,
        "sample_repetition_rate_lte_2pct": repetition_rate <= 0.02,
    }
    if is_spanish_file:
        checks["spanish_signal_gte_95pct"] = (spanish_rate or 0.0) >= 0.95
    else:
        checks["suspicious_code_path_rate_lte_0_5pct"] = (suspicious_path_rate or 0.0) <= 0.005

    return {
        "file": str(path),
        "documents": docs,
        "bytes": bytes_total,
        "gb": bytes_total / (1024**3),
        "malformed_rows": malformed,
        "sample_size": n,
        "length_chars": {
            "p50": percentile(lengths, 0.50),
            "p95": percentile(lengths, 0.95),
            "p99": percentile(lengths, 0.99),
            "max": max(lengths) if lengths else None,
        },
        "source_counts": dict(source_counts.most_common()),
        "language_counts_top20": dict(language_counts.most_common(20)),
        "metrics": {
            "malformed_rate": malformed_rate,
            "empty_rate": empty_rate,
            "sample_exact_duplicate_rate": duplicate_rate,
            "sample_repetition_rate": repetition_rate,
            "spanish_signal_rate": spanish_rate,
            "license_present_rate": license_rate if not is_spanish_file else None,
            "suspicious_code_path_rate": suspicious_path_rate,
        },
        "checks": checks,
        "eligible": all(checks.values()),
    }

def main() -> int:
    p = argparse.ArgumentParser(description="Audita corpus JSONL de CeltIA antes de adestramento.")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--sample-size", type=int, default=10000)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    root = args.root.resolve()
    files = sorted((root / "castellano").glob("*.jsonl")) + sorted((root / "codigo").glob("*.jsonl"))
    if not files:
        print(f"Non se atoparon JSONL en {root}", flush=True)
        return 2

    reports = []
    for path in files:
        print(f"[audit] {path}", flush=True)
        reports.append(inspect_file(path, args.sample_size))

    total_docs = sum(r["documents"] for r in reports)
    total_bytes = sum(r["bytes"] for r in reports)
    report = {
        "root": str(root),
        "files": reports,
        "summary": {
            "files": len(reports),
            "documents": total_docs,
            "gb": total_bytes / (1024**3),
            "eligible_files": sum(1 for r in reports if r["eligible"]),
            "all_files_eligible": all(r["eligible"] for r in reports),
        },
    }
    out = args.output or (root / "audit_report.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Informe: {out}")
    return 0 if report["summary"]["all_files_eligible"] else 3

if __name__ == "__main__":
    raise SystemExit(main())
