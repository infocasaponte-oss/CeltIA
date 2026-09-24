from pathlib import Path
import json

ALLOWED={"fast","think","code","agent","long"}

def test_route_benchmark_schema():
    path=Path("benchmarks/decision_routes.jsonl")
    rows=[json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert rows
    assert all(isinstance(r["text"],str) and r["text"].strip() for r in rows)
    assert all(r["expected"] in ALLOWED for r in rows)
    assert len({r["text"] for r in rows}) == len(rows)
