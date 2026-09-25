import importlib.util
import json
import sys
from pathlib import Path


def load_module(path_str, name):
    path = Path(path_str)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def test_audit_accepts_clean_spanish_jsonl(tmp_path):
    m = load_module("scripts/audit_curated_corpus.py", "audit_curated_corpus")
    d = tmp_path / "castellano"
    d.mkdir()
    p = d / "sample.jsonl"
    text = ("La inteligencia artificial puede ayudar a las personas cuando se usa con cuidado. "
            "Este documento está escrito en español y contiene información útil, clara y bien estructurada. ") * 8
    with p.open("w", encoding="utf-8") as fh:
        for i in range(40):
            row = {"text": text + f" Documento {i}.", "source": "test"}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = m.inspect_file(p, sample_size=100)
    assert report["eligible"]
    assert report["metrics"]["spanish_signal_rate"] >= 0.95
    assert report["metrics"]["sample_exact_duplicate_rate"] <= 0.01


def test_audit_flags_duplicates_and_malformed(tmp_path):
    m = load_module("scripts/audit_curated_corpus.py", "audit_curated_corpus")
    d = tmp_path / "codigo"
    d.mkdir()
    p = d / "sample.jsonl"
    row = {"text": "def f(x):\n    return x + 1\n", "source": "test", "language": "python"}
    with p.open("w", encoding="utf-8") as fh:
        for _ in range(20):
            fh.write(json.dumps(row) + "\n")
        fh.write("{not-json}\n")
    report = m.inspect_file(p, sample_size=100)
    assert not report["eligible"]
    assert report["malformed_rows"] == 1
    assert report["metrics"]["sample_exact_duplicate_rate"] > 0.01


def test_experimental_sources_include_public_domain_newspapers():
    m = load_module("scripts/download_curated_corpus.py", "download_curated_corpus")
    class Args:
        include_fallback_web = False
        include_experimental = True
        sources = None
        groups = ["all"]
    keys = {s.key for s in m.choose_sources(Args())}
    assert "spanish_pd_books" in keys
    assert "spanish_pd_newspapers" in keys
