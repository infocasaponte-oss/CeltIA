import json
from pathlib import Path

ROUTES={"fast","think","code","agent","long"}
VALIDATION=Path("benchmarks/holdout/decision_routes_holdout.jsonl")
PROMOTION=(
    Path("benchmarks/decision_routes.jsonl"),
    Path("benchmarks/decision_routes_ood.jsonl"),
)


def _rows(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_holdout_validation_schema_and_balance():
    rows=_rows(VALIDATION)
    assert len(rows) == 80
    assert len({row["text"] for row in rows}) == 80
    assert all(isinstance(row["text"],str) and row["text"].strip() for row in rows)

    ood=[row for row in rows if row["ood"] is True]
    in_domain=[row for row in rows if row["ood"] is False]
    assert len(ood) == 40
    assert len(in_domain) == 40
    assert all(row["expected"] is None for row in ood)
    assert all(row["expected"] in ROUTES for row in in_domain)

    counts={route:sum(row["expected"] == route for row in in_domain) for route in ROUTES}
    assert counts == {route:8 for route in ROUTES}


def test_holdout_validation_is_disjoint_from_promotion_evidence():
    validation_texts={row["text"] for row in _rows(VALIDATION)}
    promotion_texts={
        row["text"]
        for path in PROMOTION
        for row in _rows(path)
    }
    assert validation_texts.isdisjoint(promotion_texts)
