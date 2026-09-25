#!/usr/bin/env python3
"""Descarga corpus jurídico oficial en español desde BOE OpenData.

Por defecto obtiene legislación consolidada y guarda un JSONL por norma.
No descarga jurisprudencia de terceros ni scrapea webs privadas.
"""

from __future__ import annotations
import argparse, json, re, ssl, time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

ROOT = Path(r"D:\corpus_llm_grande")
BASE = "https://www.boe.es/datosabiertos/api/legislacion-consolidada"

def build_ssl_context(ca_bundle: str | None = None) -> ssl.SSLContext:
    if ca_bundle:
        return ssl.create_default_context(cafile=ca_bundle)
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()

def fetch(url: str, ssl_context: ssl.SSLContext | None = None) -> bytes:
    req = Request(url, headers={"User-Agent":"CeltIA-corpus/1.0"})
    with urlopen(req, timeout=60, context=ssl_context) as r:
        return r.read()

def text_of(elem):
    return " ".join("".join(elem.itertext()).split())

def list_ids(limit: int, ssl_context: ssl.SSLContext | None = None) -> list[str]:
    url = BASE + "?" + urlencode({"limit": -1})
    root = ET.fromstring(fetch(url, ssl_context))
    ids = []
    for e in root.iter():
        t = (e.text or "").strip()
        if re.fullmatch(r"BOE-A-\d{4}-\d+", t):
            ids.append(t)
    seen = []
    for x in ids:
        if x not in seen:
            seen.append(x)
    return seen[:limit] if limit > 0 else seen

def norm_text(xml_bytes: bytes) -> str:
    root = ET.fromstring(xml_bytes)
    text = text_of(root)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--limit", type=int, default=0, help="0 = todas las normas disponibles")
    p.add_argument("--sleep", type=float, default=0.15)
    p.add_argument("--ca-bundle", default=None, help="Ruta a un bundle CA PEM. Por defecto usa certifi si está instalado.")
    args=p.parse_args()
    outdir=args.root/"derecho"
    outdir.mkdir(parents=True, exist_ok=True)
    out=outdir/"boe_legislacion_consolidada.jsonl"
    state=outdir/"boe_state.json"
    done=set()
    if state.exists():
        try: done=set(json.loads(state.read_text(encoding="utf-8")).get("done",[]))
        except Exception: pass
    ssl_context = build_ssl_context(args.ca_bundle)
    ids=list_ids(args.limit, ssl_context)
    with out.open("a",encoding="utf-8",newline="\n") as fh:
        for i,boe_id in enumerate(ids,1):
            if boe_id in done: continue
            try:
                meta=norm_text(fetch(f"{BASE}/id/{boe_id}/metadatos", ssl_context))
                body=norm_text(fetch(f"{BASE}/id/{boe_id}/texto", ssl_context))
                if len(body)<200:
                    continue
                row={
                    "text": body,
                    "source":"boe_legislacion_consolidada",
                    "document_id":boe_id,
                    "metadata_text":meta,
                    "jurisdiction":"ES",
                    "document_type":"legislation",
                    "language":"es",
                    "official_source":True,
                }
                fh.write(json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n")
                fh.flush()
                done.add(boe_id)
                state.write_text(json.dumps({"done":sorted(done)},ensure_ascii=False),encoding="utf-8")
                print(f"[{i}/{len(ids)}] {boe_id}")
                time.sleep(args.sleep)
            except KeyboardInterrupt:
                break
            except Exception as exc:
                print(f"[WARN] {boe_id}: {exc}")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
