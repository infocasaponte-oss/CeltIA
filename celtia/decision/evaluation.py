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
