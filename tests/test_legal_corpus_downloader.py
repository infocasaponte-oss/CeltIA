import importlib.util
import sys
from pathlib import Path


def load_module():
    path = Path("scripts/download_legal_corpus.py")
    spec = importlib.util.spec_from_file_location("download_legal_corpus", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def test_norm_text_extracts_xml_text():
    m = load_module()
    xml = b"<root><titulo>Ley de prueba</titulo><p>Articulo primero.</p></root>"
    text = m.norm_text(xml)
    assert "Ley de prueba" in text
    assert "Articulo primero." in text


def test_list_id_pattern_contract():
    m = load_module()
    import re
    assert re.fullmatch(r"BOE-A-\d{4}-\d+", "BOE-A-2024-12345")
    assert not re.fullmatch(r"BOE-A-\d{4}-\d+", "OTRO-2024-1")
