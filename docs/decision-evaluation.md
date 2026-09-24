# CDE shadow evaluation and promotion gate

CDE routing starts in shadow mode. The production heuristic remains authoritative while CeltIA records only route names, confidence, abstention and agreement. User prompt text is not stored in the decision-shadow table.

The admin report exposes sample count, agreement, abstention, mean confidence, mean normalized entropy, mean decision margin, abstention-reason counts and common disagreement pairs.

Promotion to controlled routing must not be based on agreement alone. A labeled evaluation set is required. The offline evaluation helper compares both the existing heuristic and CDE against expected routes and applies explicit minimum sample, coverage, labeled-sample and accuracy-delta gates.

Default CLI gate:
- at least 200 total samples;
- at least 80% CDE decision coverage;
- at least 50 labeled routing samples;
- at least 80% coverage on labeled routing samples;
- CDE labeled routing accuracy at least 2 percentage points above the heuristic;
- at least 20 OOD-labeled samples spanning in-domain negatives and OOD positives;
- at least 80% OOD-signal coverage;
- at least 80% OOD recall;
- at most 20% OOD false-positive rate.

The reusable `promotion_gate()` function keeps OOD thresholds optional for backwards-compatible programmatic use. The evaluation CLI supplies the OOD thresholds above by default so a controlled-routing eligibility check cannot pass using routing accuracy alone.

Passing the gate means eligible for a controlled experiment, not automatic activation. Tool authorization and security policy remain independent of routing.


Evaluation also reports selective accuracy: accuracy only on labeled samples where CDE actually returns a decision. This is reported alongside total labeled accuracy and labeled coverage so abstention cannot hide errors or inflate the promotion result.


## Option-order evaluation harness

`celtia.decision.order_eval` can evaluate a scorer across a bounded, deterministic set of candidate permutations. It always includes the canonical order and reverse order, then adds rotations up to `max_orders` (default 8), avoiding factorial growth for large candidate sets.

The report keeps distributions label-aligned and records maximum probability deviation plus whether the winning label changes across orders. This is an evaluation path only; production decisions are not multiplied by permutation testing.


Shadow telemetry stores only routing metadata and uncertainty diagnostics. It does not persist the user prompt. Existing databases are migrated in place by adding nullable `abstention_reason`, `normalized_entropy`, and `margin` columns.


## Labeled OOD evaluation

Decision results expose `suspected_ood` separately from `abstention_reason`. This matters when low confidence and OOD are both true: the abstention reason keeps its deterministic precedence, while evaluation can still count the independent OOD signal.

`evaluate_shadow` now reports OOD label coverage plus true/false positives and negatives, precision, recall, specificity, false-positive rate and accuracy whenever samples provide `expected_ood` and `suspected_ood`. Route labels and OOD labels are independent, so an in-domain route sample can contribute to both routing accuracy and OOD specificity.

`benchmarks/decision_routes_ood.jsonl` now contains 40 adversarial seed cases spanning malformed, injection-like, candidate-label and route-manipulation inputs. The in-domain routing benchmark contains 60 balanced cases (12 each for `fast`, `think`, `code`, `agent` and `long`). The normal routing benchmark is explicitly labeled `"ood": false`, while OOD rows use `"expected": null, "ood": true`. The evaluator accepts multiple repeatable `--dataset` arguments so both sets can be measured together, rejects duplicate texts across files, and consumes optional `suspected_ood` values from CDE-result JSONL. Both sets are schema- and cardinality-validated in CI. Together they are materially broader than the original smoke set, but still below the 200-sample promotion floor and should not be treated as sufficient production evidence.

Example combined evaluation:

```bash
PYTHONPATH=. python scripts/evaluate_decision_routes.py \
  --dataset benchmarks/decision_routes.jsonl \
  --dataset benchmarks/decision_routes_ood.jsonl \
  --cde-results results/cde_routes.jsonl \
  --require-eligible
```


## Promotion-result integrity

Promotion-result ingestion is strict. Each result row must have a unique non-empty benchmark text, a finite confidence in `[0,1]`, a real boolean `abstained`, an optional boolean `suspected_ood`, and a route consistent with abstention state: decided rows require one of `fast/think/code/agent/long`, while abstained rows require `cde: null`. Duplicate result texts, NaN/infinite confidence, string booleans, unknown routes and results for texts outside the selected benchmark datasets are rejected instead of being coerced or silently ignored.
