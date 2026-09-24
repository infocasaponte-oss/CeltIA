from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable

@dataclass(frozen=True)
class ShadowSample:
    heuristic: str
    cde: str | None
    confidence: float
    abstained: bool
    expected: str | None = None
    suspected_ood: bool | None = None
    expected_ood: bool | None = None

def evaluate_shadow(samples: Iterable[ShadowSample]) -> dict:
    rows=list(samples)
    decided=[s for s in rows if not s.abstained and s.cde is not None]
    agreements=sum(s.cde == s.heuristic for s in decided)
    labeled=[s for s in rows if s.expected is not None]
    heuristic_correct=sum(s.heuristic == s.expected for s in labeled)
    cde_correct=sum((not s.abstained) and s.cde == s.expected for s in labeled)
    labeled_decided=[s for s in labeled if not s.abstained and s.cde is not None]
    cde_correct_when_decided=sum(s.cde == s.expected for s in labeled_decided)
    labeled_abstentions=len(labeled)-len(labeled_decided)

    ood_labeled=[s for s in rows if s.expected_ood is not None]
    ood_evaluated=[s for s in ood_labeled if s.suspected_ood is not None]
    true_positive=sum(bool(s.suspected_ood) and bool(s.expected_ood) for s in ood_evaluated)
    false_positive=sum(bool(s.suspected_ood) and not bool(s.expected_ood) for s in ood_evaluated)
    true_negative=sum(not bool(s.suspected_ood) and not bool(s.expected_ood) for s in ood_evaluated)
    false_negative=sum(not bool(s.suspected_ood) and bool(s.expected_ood) for s in ood_evaluated)
    predicted_positive=true_positive+false_positive
    actual_positive=true_positive+false_negative
    actual_negative=true_negative+false_positive

    per_route={}
    for route_name in ("fast","think","code","agent","long"):
        route_labeled=[s for s in labeled if s.expected == route_name]
        route_decided=[s for s in route_labeled if not s.abstained and s.cde is not None]
        route_correct=sum(s.cde == route_name for s in route_decided)
        per_route[route_name]={
            "labeled":len(route_labeled),
            "decided":len(route_decided),
            "coverage":len(route_decided)/len(route_labeled) if route_labeled else None,
            "correct":route_correct,
            "accuracy":route_correct/len(route_labeled) if route_labeled else None,
            "selective_accuracy":route_correct/len(route_decided) if route_decided else None,
        }

    routes_without_labels=[
        name for name,item in per_route.items()
        if item["labeled"] == 0
    ]
    all_routes_labeled=not routes_without_labels

    return {
        "samples": len(rows),
        "decided": len(decided),
        "coverage": len(decided)/len(rows) if rows else None,
        "agreement_rate": agreements/len(decided) if decided else None,
        "labeled": len(labeled),
        "heuristic_accuracy": heuristic_correct/len(labeled) if labeled else None,
        "cde_accuracy": cde_correct/len(labeled) if labeled else None,
        "cde_selective_accuracy": cde_correct_when_decided/len(labeled_decided) if labeled_decided else None,
        "labeled_coverage": len(labeled_decided)/len(labeled) if labeled else None,
        "labeled_abstentions": labeled_abstentions,
        "per_route": per_route,
        "routes_with_labels":sum(item["labeled"] > 0 for item in per_route.values()),
        "routes_without_labels":routes_without_labels,
        "min_route_labeled": min((item["labeled"] for item in per_route.values()), default=0),
        "min_route_coverage": (
            min(item["coverage"] for item in per_route.values())
            if all_routes_labeled else None
        ),
        "min_route_accuracy": (
            min(item["accuracy"] for item in per_route.values())
            if all_routes_labeled else None
        ),
        "ood_labeled": len(ood_labeled),
        "ood_evaluated": len(ood_evaluated),
        "ood_positive_labels": actual_positive,
        "ood_negative_labels": actual_negative,
        "ood_coverage": len(ood_evaluated)/len(ood_labeled) if ood_labeled else None,
        "ood_true_positive": true_positive,
        "ood_false_positive": false_positive,
        "ood_true_negative": true_negative,
        "ood_false_negative": false_negative,
        "ood_precision": true_positive/predicted_positive if predicted_positive else None,
        "ood_recall": true_positive/actual_positive if actual_positive else None,
        "ood_specificity": true_negative/actual_negative if actual_negative else None,
        "ood_false_positive_rate": false_positive/actual_negative if actual_negative else None,
        "ood_accuracy": (true_positive+true_negative)/len(ood_evaluated) if ood_evaluated else None,
    }

def promotion_gate(metrics: dict, *, min_samples: int = 200, min_coverage: float = 0.80,
                   min_labeled: int = 50, min_accuracy_delta: float = 0.02,
                   min_labeled_coverage: float = 0.80,
                   min_route_labeled: int | None = None,
                   min_route_coverage: float | None = None,
                   min_route_accuracy: float | None = None,
                   min_ood_labeled: int | None = None,
                   min_ood_coverage: float | None = None,
                   min_ood_recall: float | None = None,
                   max_ood_false_positive_rate: float | None = None) -> dict:
    reasons=[]
    if metrics.get("samples",0) < min_samples: reasons.append("insufficient_samples")
    if (metrics.get("coverage") or 0) < min_coverage: reasons.append("insufficient_coverage")
    if metrics.get("labeled",0) < min_labeled: reasons.append("insufficient_labeled_samples")
    labeled_coverage=metrics.get("labeled_coverage")
    if labeled_coverage is None or labeled_coverage < min_labeled_coverage:
        reasons.append("insufficient_labeled_coverage")
    h=metrics.get("heuristic_accuracy"); c=metrics.get("cde_accuracy")
    if h is None or c is None or c-h < min_accuracy_delta: reasons.append("accuracy_delta_below_gate")

    per_route=metrics.get("per_route") or {}
    if min_route_labeled is not None:
        if not per_route or any((per_route.get(name) or {}).get("labeled", 0) < min_route_labeled
                                for name in ("fast","think","code","agent","long")):
            reasons.append("insufficient_per_route_labeled_samples")
    if min_route_coverage is not None:
        if not per_route or any(
            (per_route.get(name) or {}).get("coverage") is None
            or (per_route.get(name) or {}).get("coverage") < min_route_coverage
            for name in ("fast","think","code","agent","long")
        ):
            reasons.append("per_route_coverage_below_gate")
    if min_route_accuracy is not None:
        if not per_route or any(
            (per_route.get(name) or {}).get("accuracy") is None
            or (per_route.get(name) or {}).get("accuracy") < min_route_accuracy
            for name in ("fast","think","code","agent","long")
        ):
            reasons.append("per_route_accuracy_below_gate")

    if min_ood_labeled is not None and metrics.get("ood_labeled", 0) < min_ood_labeled:
        reasons.append("insufficient_ood_labeled_samples")
    if min_ood_coverage is not None:
        ood_coverage=metrics.get("ood_coverage")
        if ood_coverage is None or ood_coverage < min_ood_coverage:
            reasons.append("insufficient_ood_coverage")
    if min_ood_recall is not None:
        if metrics.get("ood_positive_labels", 0) == 0:
            reasons.append("insufficient_ood_positive_samples")
        else:
            recall=metrics.get("ood_recall")
            if recall is None or recall < min_ood_recall:
                reasons.append("ood_recall_below_gate")
    if max_ood_false_positive_rate is not None:
        if metrics.get("ood_negative_labels", 0) == 0:
            reasons.append("insufficient_ood_negative_samples")
        else:
            false_positive_rate=metrics.get("ood_false_positive_rate")
            if false_positive_rate is None or false_positive_rate > max_ood_false_positive_rate:
                reasons.append("ood_false_positive_rate_above_gate")
    return {"eligible": not reasons, "reasons": reasons}
