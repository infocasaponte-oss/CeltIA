# CDE shadow evaluation and promotion gate

CDE routing starts in shadow mode. The production heuristic remains authoritative while CeltIA records only route names, confidence, abstention and agreement. User prompt text is not stored in the decision-shadow table.

The admin report exposes sample count, agreement, abstention, mean confidence, mean normalized entropy, mean decision margin, abstention-reason counts, common disagreement pairs, and aggregate shadow prompt/completion/total tokens plus average tokens per sample.

Promotion to controlled routing must not be based on agreement alone. A labeled evaluation set is required. The offline evaluation helper compares both the existing heuristic and CDE against expected routes and applies explicit minimum sample, coverage, labeled-sample and accuracy-delta gates.

Default CLI gate:
- at least 200 total samples;
- at least 80% CDE decision coverage;
- at least 50 labeled routing samples;
- at least 80% coverage on labeled routing samples;
- CDE labeled routing accuracy at least 2 percentage points above the heuristic;
- at least 10 labeled samples for every individual route;
- at least 70% CDE decision coverage on every individual route;
- at least 60% total accuracy on every individual route (abstentions count against this value);
- at least 20 OOD-labeled samples spanning in-domain negatives and OOD positives;
- at least 80% OOD-signal coverage;
- at least 80% OOD recall;
- at most 20% OOD false-positive rate.

The reusable `promotion_gate()` function keeps per-route and OOD thresholds optional for backwards-compatible programmatic use. The evaluation CLI supplies both sets of thresholds above by default. Aggregate accuracy therefore cannot hide a collapsed or heavily abstaining route, and a controlled-routing eligibility check cannot pass using routing accuracy alone.

Passing the gate means eligible for a controlled experiment, not automatic activation. Tool authorization and security policy remain independent of routing.


Evaluation also reports selective accuracy: accuracy only on labeled samples where CDE actually returns a decision. This is reported alongside total labeled accuracy and labeled coverage so abstention cannot hide errors or inflate the promotion result. The same distinction is now reported per route, together with route-specific labeled count and coverage; promotion uses the non-selective per-route accuracy floor.


## Option-order evaluation harness

`celtia.decision.order_eval` can evaluate a scorer across a bounded, deterministic set of candidate permutations. It always includes the canonical order and reverse order, then adds rotations up to `max_orders` (default 8), avoiding factorial growth for large candidate sets.

The report keeps distributions label-aligned and records maximum probability deviation plus whether the winning label changes across orders. This is an evaluation path only; production decisions are not multiplied by permutation testing.


Shadow telemetry stores only routing metadata, uncertainty diagnostics and aggregate token counts. It does not persist the user prompt. Shadow inference usage is recorded for operator cost visibility but is not added to end-user billing/quota accounting. Existing databases are migrated in place by adding nullable diagnostics and token-count columns.


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


## Collecting real CDE results

`scripts/collect_decision_eval_results.py` runs the same route-decision shape used by shadow routing against the configured CeltIA LLM stack and writes strict JSONL suitable for the offline evaluator. By default it combines the 60 in-domain routing cases and 40 OOD cases. It records the CDE route, confidence, abstention, explicit `suspected_ood`, uncertainty diagnostics, heuristic route and benchmark labels.

Because the collector uses `build_llm()`, it follows the active CeltIA provider configuration: when the hosted primary is enabled and credentialed, running the collector can send benchmark prompts to that provider and consume billable model usage. CI only smoke-tests `--help`; it never executes live model calls.

Example:

```bash
PYTHONPATH=. python scripts/collect_decision_eval_results.py \
  --output results/cde_routes.jsonl \
  --sleep-seconds 0.25

PYTHONPATH=. python scripts/evaluate_decision_routes.py \
  --dataset benchmarks/decision_routes.jsonl \
  --dataset benchmarks/decision_routes_ood.jsonl \
  --cde-results results/cde_routes.jsonl
```

Use `--resume` to continue an interrupted collection without re-running already valid rows, or `--limit N` for a deterministic small pilot. Existing resume files are validated before reuse so malformed or out-of-dataset rows are not silently carried forward. Collection now checkpoints atomically: by default every 10 newly collected rows it writes and fsyncs a complete uniquely named temporary JSONL, atomically replaces the destination, and fsyncs the parent directory where the platform supports directory descriptors. A failed replacement preserves the previous result file and cleans the temporary checkpoint. Unique temporary names also avoid accidental temp-file collisions between overlapping processes; this is crash-safety hardening, not a multi-writer locking protocol. `--checkpoint-every N` can tune the durability/cost trade-off from 1 to 100 rows.

Route summaries are complete: if any route has no labels, `min_route_coverage` and `min_route_accuracy` are `null`, with `routes_with_labels` and `routes_without_labels` exposing the missing evidence.
