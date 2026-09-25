import importlib.util
from pathlib import Path


def load_module():
    path = Path("scripts/download_curated_corpus.py")
    spec = importlib.util.spec_from_file_location("download_curated_corpus", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_spanish_filter_accepts_normal_spanish():
    m = load_module()
    text = ("La inteligencia artificial puede ayudar a las personas cuando se usa con cuidado. "
            "Este texto está escrito en español y contiene frases normales, información útil y "
            "una estructura razonable para un corpus de entrenamiento. ") * 8
    assert m.spanish_quality(text)


def test_spanish_filter_rejects_noise():
    m = load_module()
    assert not m.spanish_quality("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" * 30)


def test_code_filter_rejects_vendor_and_minified_paths():
    m = load_module()
    code = "def suma(a, b):\n    return a + b\n" * 20
    assert m.code_quality(code, {"metadata": {"path": "src/math.py"}})
    assert not m.code_quality(code, {"metadata": {"path": "vendor/math.py", "is_vendor": True}})


def test_code_language_from_extension():
    m = load_module()
    assert m.infer_code_language({"metadata": {"path": "src/main.py"}}) == "python"
    assert m.infer_code_language({"metadata": {"path": "web/app.tsx"}}) == "typescript"


def test_default_sources_avoid_duplicate_web_fallbacks():
    m = load_module()
    class Args:
        include_fallback_web = False
        sources = None
        groups = ["all"]
    keys = {s.key for s in m.choose_sources(Args())}
    assert "culturax_es" in keys
    assert "mc4_es" not in keys
    assert "oscar_es" not in keys
    assert "stackv2_edu" in keys
