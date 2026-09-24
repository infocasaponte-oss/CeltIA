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
        "ood_labeled": len(ood_labeled),
        "ood_evaluated": len(ood_evaluated),
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
                   min_labeled_coverage: float = 0.80) -> dict:
    reasons=[]
    if metrics.get("samples",0) < min_samples: reasons.append("insufficient_samples")
    if (metrics.get("coverage") or 0) < min_coverage: reasons.append("insufficient_coverage")
    if metrics.get("labeled",0) < min_labeled: reasons.append("insufficient_labeled_samples")
    labeled_coverage=metrics.get("labeled_coverage")
    if labeled_coverage is None or labeled_coverage < min_labeled_coverage:
        reasons.append("insufficient_labeled_coverage")
    h=metrics.get("heuristic_accuracy"); c=metrics.get("cde_accuracy")
    if h is None or c is None or c-h < min_accuracy_delta: reasons.append("accuracy_delta_below_gate")
    return {"eligible": not reasons, "reasons": reasons}
