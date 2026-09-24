from __future__ import annotations
from collections.abc import Mapping, Sequence

def multiclass_brier(distributions: Sequence[Mapping[str,float]], targets: Sequence[str]) -> float:
    if len(distributions) != len(targets) or not distributions:
        raise ValueError("distributions and targets must be non-empty and aligned")
    total=0.0
    for dist,target in zip(distributions,targets,strict=True):
        if target not in dist:
            raise ValueError("target missing from distribution")
        s=sum(float(v) for v in dist.values())
        if abs(s-1.0) > 1e-6:
            raise ValueError("distribution must sum to one")
        total += sum((float(p)-(1.0 if label==target else 0.0))**2 for label,p in dist.items())
    return total/len(distributions)

def top_label_calibration(distributions: Sequence[Mapping[str,float]], targets: Sequence[str]) -> tuple[list[float],list[bool]]:
    if len(distributions) != len(targets):
        raise ValueError("distributions and targets must be aligned")
    confidence=[]; correct=[]
    for dist,target in zip(distributions,targets,strict=True):
        if not dist:
            raise ValueError("empty distribution")
        label=max(dist,key=dist.get)
        confidence.append(float(dist[label])); correct.append(label==target)
    return confidence,correct
