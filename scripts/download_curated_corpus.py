#!/usr/bin/env python3
"""Curated streaming corpus downloader for CeltIA.

Defaults are intentionally conservative:
- Spanish: Wikipedia ES, CulturaX ES, Spanish public-domain books.
- Code: Common Pile StackV2 Edu (open-license filtered + educational signal).
- OSCAR/mC4 are optional fallbacks because CulturaX already incorporates them.

Outputs JSONL under D:\\corpus_llm_grande by default and keeps Hugging Face cache on D:.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

DEFAULT_ROOT = Path(r"D:\corpus_llm_grande")
SPANISH_STOPWORDS = {
    "de","la","que","el","en","y","a","los","del","se","las","por","un","para",
    "con","no","una","su","al","lo","como","más","pero","sus","le","ya","o","este",
    "sí","porque","esta","entre","cuando","muy","sin","sobre","también","me","hasta",
}
CODE_EXTENSIONS = {
    ".py":"python",".js":"javascript",".jsx":"javascript",".ts":"typescript",".tsx":"typescript",
    ".java":"java",".c":"c",".h":"c",".cc":"cpp",".cpp":"cpp",".cxx":"cpp",".hpp":"cpp",
    ".cs":"csharp",".go":"go",".rs":"rust",".rb":"ruby",".php":"php",".swift":"swift",
    ".kt":"kotlin",".kts":"kotlin",".scala":"scala",".sh":"shell",".bash":"shell",
    ".ps1":"powershell",".sql":"sql",".html":"html",".css":"css",".vue":"vue",
    ".svelte":"svelte",".lua":"lua",".r":"r",".dart":"dart",
}

@dataclass(frozen=True)
class Source:
    key: str
    group: str
    dataset: str
    config: str | None = None
    text_field: str = "text"
    gated: bool = False
    enabled: bool = True
    max_gb: float = 0.0

SOURCES = [
    Source("wikipedia_es", "spanish", "wikimedia/wikipedia", "20231101.es", max_gb=8),
    Source("culturax_es", "spanish", "uonlp/CulturaX", "es", gated=True, max_gb=40),
    # High-value public-domain books, but the current HF repository has shard schema
    # inconsistencies; keep opt-in until upstream normalizes it.
    Source("spanish_pd_books", "spanish_optional", "PleIAs/Spanish-PD-Books", enabled=False, max_gb=30),
    Source("spanish_pd_newspapers", "spanish_optional", "PleIAs/Spanish-PD-Newspapers", enabled=False, max_gb=20),
    Source("stackv2_edu", "code", "common-pile/stackv2_edu_filtered", max_gb=60),
    # Optional fallbacks / expansion. Disabled to reduce overlap with CulturaX.
    Source("mc4_es", "spanish_optional", "allenai/c4", "es", enabled=False, max_gb=25),
    Source("oscar_es", "spanish_optional", "oscar-corpus/oscar", "unshuffled_deduplicated_es", enabled=False, max_gb=25),
]

def normalize_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()

def repeated_char_ratio(text: str) -> float:
    if not text:
        return 1.0
    runs = sum(len(m.group(0)) for m in re.finditer(r"(.)\1{7,}", text, flags=re.S))
    return runs / len(text)

def repeated_line_ratio(text: str) -> float:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 4:
        return 0.0
    unique = len(set(lines))
    return 1.0 - (unique / len(lines))

def looks_spanish(text: str) -> bool:
    if len(text) < 200:
        return False
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

def spanish_quality(text: str) -> bool:
    if not (300 <= len(text) <= 2_000_000):
        return False
    if not looks_spanish(text):
        return False
    if repeated_char_ratio(text) > 0.015 or repeated_line_ratio(text) > 0.35:
        return False
    url_count = len(re.findall(r"https?://|www\.", text[:50000], flags=re.I))
    if url_count > 40:
        return False
    return True

def infer_code_language(row: dict[str, Any]) -> str:
    meta = row.get("metadata") or {}
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            meta = {}
    for key in ("language", "programming_language", "lang"):
        value = row.get(key) or (meta.get(key) if isinstance(meta, dict) else None)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    path = ""
    if isinstance(meta, dict):
        path = str(meta.get("path") or meta.get("file_path") or meta.get("url") or "")
    path = str(row.get("path") or row.get("file_path") or path)
    lower = path.lower().split("?", 1)[0]
    for ext, lang in CODE_EXTENSIONS.items():
        if lower.endswith(ext):
            return lang
    return "unknown"

def code_quality(text: str, row: dict[str, Any]) -> bool:
    if not (80 <= len(text) <= 1_500_000):
        return False
    meta = row.get("metadata") or {}
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            meta = {}
    if isinstance(meta, dict):
        if meta.get("is_vendor") is True or meta.get("is_generated") is True:
            return False
        path = str(meta.get("path") or meta.get("file_path") or "").lower()
        if any(part in path for part in ("/vendor/", "/dist/", "/build/", "/node_modules/", ".min.js", ".map")):
            return False
    if repeated_char_ratio(text) > 0.03 or repeated_line_ratio(text) > 0.55:
        return False
    lines = text.splitlines()
    if lines:
        long_lines = sum(1 for ln in lines if len(ln) > 1200)
        if long_lines / len(lines) > 0.20:
            return False
    symbolish = sum(not (ch.isalnum() or ch.isspace()) for ch in text[:12000])
    if symbolish / max(1, min(len(text), 12000)) > 0.65:
        return False
    return True

class Deduper:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS seen (h TEXT PRIMARY KEY)")
        self.pending = 0

    def add(self, text: str) -> bool:
        h = hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()
        try:
            self.db.execute("INSERT INTO seen(h) VALUES(?)", (h,))
        except sqlite3.IntegrityError:
            return False
        self.pending += 1
        if self.pending >= 1000:
            self.db.commit()
            self.pending = 0
        return True

    def close(self) -> None:
        self.db.commit()
        self.db.close()

def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def save_state(path: Path, state: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

def extract_text(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    if isinstance(value, str):
        return value
    for candidate in ("content", "document", "body"):
        value = row.get(candidate)
        if isinstance(value, str):
            return value
    return ""

def stream_source(source: Source, args: argparse.Namespace, root: Path, deduper: Deduper) -> dict[str, Any]:
    from datasets import load_dataset

    target_dir = root / ("castellano" if source.group.startswith("spanish") else "codigo")
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / f"{source.key}.jsonl"
    state_path = root / "_state" / f"{source.key}.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = load_state(state_path)
    seen_rows = int(state.get("seen_rows", 0))
    written = int(state.get("written", 0))
    written_bytes = int(state.get("written_bytes", 0))
    rejected = int(state.get("rejected", 0))
    duplicates = int(state.get("duplicates", 0))

    cap_gb = args.max_gb if args.max_gb is not None else source.max_gb
    cap_bytes = int(cap_gb * (1024**3)) if cap_gb and cap_gb > 0 else 0

    print(f"\n[{source.key}] cargando {source.dataset} config={source.config!r}")
    try:
        ds = load_dataset(
            source.dataset,
            source.config,
            split="train",
            streaming=True,
            token=args.hf_token,
        )
    except Exception as exc:
        if source.gated:
            print(f"[{source.key}] omitido: dataset gated/non accesible ({exc})")
            return {"status":"skipped_gated","error":str(exc)}
        raise

    if seen_rows:
        ds = ds.skip(seen_rows)
        print(f"[{source.key}] retomando tras {seen_rows:,} filas xa inspeccionadas")

    mode = "a" if out_path.exists() else "w"
    started = time.time()
    with out_path.open(mode, encoding="utf-8", newline="\n") as fh:
        for row in ds:
            seen_rows += 1
            text = normalize_text(extract_text(row, source.text_field))
            ok = spanish_quality(text) if source.group.startswith("spanish") else code_quality(text, row)
            if not ok:
                rejected += 1
            elif not deduper.add(text):
                duplicates += 1
            else:
                record: dict[str, Any] = {
                    "text": text,
                    "source": source.key,
                    "dataset": source.dataset,
                }
                if source.group == "code":
                    record["language"] = infer_code_language(row)
                    meta = row.get("metadata")
                    if isinstance(meta, dict):
                        record["license"] = meta.get("license")
                        record["path"] = meta.get("path") or meta.get("file_path")
                        record["url"] = meta.get("url")
                raw = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                fh.write(raw)
                written += 1
                written_bytes += len(raw.encode("utf-8"))

            if seen_rows % args.checkpoint_every == 0:
                fh.flush()
                save_state(state_path, {
                    "seen_rows":seen_rows,"written":written,"written_bytes":written_bytes,
                    "rejected":rejected,"duplicates":duplicates,
                })
                elapsed = max(1.0, time.time() - started)
                print(
                    f"[{source.key}] vistos={seen_rows:,} escritos={written:,} "
                    f"GB={written_bytes/(1024**3):.2f} rexeitados={rejected:,} "
                    f"duplicados={duplicates:,} ritmo={seen_rows/elapsed:.1f} filas/s"
                )

            if cap_bytes and written_bytes >= cap_bytes:
                print(f"[{source.key}] límite alcanzado: {cap_gb:.1f} GB")
                break

    save_state(state_path, {
        "seen_rows":seen_rows,"written":written,"written_bytes":written_bytes,
        "rejected":rejected,"duplicates":duplicates,
    })
    return {
        "status":"ok","seen_rows":seen_rows,"written":written,
        "written_bytes":written_bytes,"rejected":rejected,"duplicates":duplicates,
        "output":str(out_path),
    }

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Descarga e filtra corpus ES + código en streaming.")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--groups", nargs="+", choices=["spanish","code","all"], default=["all"])
    p.add_argument("--sources", nargs="*", help="Claves concretas; sobrescribe --groups.")
    p.add_argument("--include-fallback-web", action="store_true", help="Activa mC4 ES e OSCAR ES ademais de CulturaX.")
    p.add_argument("--include-experimental", action="store_true", help="Activa fontes boas pero actualmente menos robustas, como Spanish-PD-Books.")
    p.add_argument("--max-gb", type=float, default=None, help="Límite por fonte. 0 = sen límite.")
    p.add_argument("--checkpoint-every", type=int, default=5000)
    p.add_argument("--hf-token", default=os.getenv("HF_TOKEN"))
    p.add_argument("--list-sources", action="store_true")
    return p.parse_args()

def choose_sources(args: argparse.Namespace) -> list[Source]:
    sources = list(SOURCES)
    if args.include_fallback_web or args.include_experimental:
        enabled_keys = set()
        if args.include_fallback_web:
            enabled_keys.update({"mc4_es", "oscar_es"})
        if args.include_experimental:
            enabled_keys.update({"spanish_pd_books", "spanish_pd_newspapers"})
        sources = [
            Source(s.key, s.group, s.dataset, s.config, s.text_field, s.gated, True, s.max_gb)
            if s.key in enabled_keys else s
            for s in sources
        ]
    if args.sources:
        wanted = set(args.sources)
        return [s for s in sources if s.key in wanted]
    groups = {"spanish","code"} if "all" in args.groups else set(args.groups)
    return [
        s for s in sources
        if s.enabled and (("spanish" in groups and s.group.startswith("spanish")) or s.group in groups)
    ]

def main() -> int:
    args = parse_args()
    if args.list_sources:
        for s in SOURCES:
            print(f"{s.key:20} {s.group:16} {s.dataset} {s.config or ''} default={s.enabled}")
        return 0

    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    cache = root / "_hf_cache"
    os.environ.setdefault("HF_HOME", str(cache))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache / "datasets"))
    os.environ.setdefault("HF_HUB_CACHE", str(cache / "hub"))

    selected = choose_sources(args)
    if not selected:
        print("Non hai fontes seleccionadas.", file=sys.stderr)
        return 2

    manifest = {
        "root":str(root),
        "sources":[s.__dict__ for s in selected],
        "notes":[
            "CulturaX pode requirir aceptar condicións en Hugging Face e HF_TOKEN.",
            "OSCAR/mC4 están desactivados por defecto para evitar solapamento masivo con CulturaX.",
            "Spanish-PD-Books e Spanish-PD-Newspapers quedan opt-in: son fontes de dominio público de alto valor, pero poden requirir limpeza adicional de OCR/esquema.",
            "A deduplicación implementada é exacta por hash; non substitúe near-dedup semántica global.",
            "Revisa as licenzas/metadatos antes de redistribuír ou publicar o corpus resultante.",
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    deduper = Deduper(root / "_state" / "dedup.sqlite3")
    results: dict[str, Any] = {}
    try:
        for source in selected:
            try:
                results[source.key] = stream_source(source, args, root, deduper)
            except KeyboardInterrupt:
                print("\nInterrompido. O estado xa gardado permite retomar.", file=sys.stderr)
                break
            except Exception as exc:
                print(f"[{source.key}] ERRO: {exc}", file=sys.stderr)
                results[source.key] = {"status":"error","error":str(exc)}
    finally:
        deduper.close()

    (root / "run_summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0 if all(v.get("status") in {"ok","skipped_gated"} for v in results.values()) else 1

if __name__ == "__main__":
    raise SystemExit(main())
