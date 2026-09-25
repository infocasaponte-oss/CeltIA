from pathlib import Path
import json

ALLOWED={"fast","think","code","agent","long"}

def test_route_benchmark_schema():
    path=Path("benchmarks/decision_routes.jsonl")
    rows=[json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(rows) == 120
    assert all(isinstance(r["text"],str) and r["text"].strip() for r in rows)
    assert all(r["expected"] in ALLOWED for r in rows)
    assert all(r.get("ood") is False for r in rows)
    counts={label: sum(r["expected"] == label for r in rows) for label in ALLOWED}
    assert counts == {label: 24 for label in ALLOWED}
    assert len({r["text"] for r in rows}) == len(rows)


def test_route_ood_benchmark_schema():
    path=Path("benchmarks/decision_routes_ood.jsonl")
    rows=[json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(rows) == 80
    assert all(isinstance(r["text"],str) and r["text"].strip() for r in rows)
    assert all(r.get("expected") is None for r in rows)
    assert all(r.get("ood") is True for r in rows)
    assert len({r["text"] for r in rows}) == len(rows)



def test_combined_decision_benchmark_has_200_unique_texts():
    paths=(
        Path("benchmarks/decision_routes.jsonl"),
        Path("benchmarks/decision_routes_ood.jsonl"),
    )
    rows=[
        json.loads(line)
        for path in paths
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 200
    assert len({row["text"] for row in rows}) == 200
