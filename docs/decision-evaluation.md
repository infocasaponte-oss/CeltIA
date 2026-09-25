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

`benchmarks/decision_routes_ood.jsonl` now contains 80 adversarial cases spanning malformed, injection-like, candidate-label and route-manipulation inputs. The in-domain routing benchmark contains 120 balanced cases (24 each for `fast`, `think`, `code`, `agent` and `long`). The normal routing benchmark is explicitly labeled `"ood": false`, while OOD rows use `"expected": null, "ood": true`. The evaluator accepts multiple repeatable `--dataset` arguments so both sets can be measured together, rejects duplicate texts across files, and consumes optional `suspected_ood` values from CDE-result JSONL. Both sets are schema- and cardinality-validated in CI. Together they now reach the 200-sample promotion floor structurally. This only removes the sample-count blocker: promotion still requires a real CDE result artifact whose coverage, routing accuracy, per-route metrics and OOD behavior satisfy every gate.

For local live collection when the provider key already lives in `.env` or `.env.local`, use the repository helper:

```bash
PYTHONPATH=. python scripts/run_cde_live_evidence.py
```

The helper refuses to collect from a tracked-dirty checkout, verifies that CeltIA can resolve a live-provider credential without printing it, runs the full 200-case collector, and then evaluates the promotion gate. If the gate fails, the JSONL and v4 manifest are intentionally kept for inspection. Use `--resume` to continue a compatible interrupted run.

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

`scripts/collect_decision_eval_results.py` runs the same route-decision shape used by shadow routing against the configured CeltIA LLM stack and writes strict JSONL suitable for the offline evaluator. By default it combines the 120 in-domain routing cases and 80 OOD cases. It records the CDE route, confidence, abstention, explicit `suspected_ood`, uncertainty diagnostics, heuristic route and benchmark labels.

Because the collector uses `build_llm()`, it follows the active CeltIA provider configuration: when the hosted primary is enabled and credentialed, running the collector can send benchmark prompts to that provider and consume billable model usage. CI only smoke-tests `--help`; it never executes live model calls.
A guarded GitHub Actions workflow, `.github/workflows/cde-live-evidence.yml`, is available for deliberate live collection. Before this PR is merged, it can run on the feature branch only when the pushed head commit message contains `[run-live-cde]`; ordinary pushes create no live job. After the workflow exists on the default branch it can also be started with `workflow_dispatch`, which requires the operator to type `RUN_LIVE_CDE`. In both modes it fails before model calls when none of `LLM_PRIMARY_API_KEY`, `XAI_API_KEY` or `IMAGE_API_KEY` is configured. Successful runs archive the result JSONL, v4 manifest, evaluator output and gate exit code as the `cde-live-evidence` artifact.


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

Promotion threshold configuration is validated before evaluation: sample-count floors must be non-negative integers, probability/rate thresholds must be finite values in `[0,1]`, and the allowed accuracy delta is finite and bounded to `[-1,1]`. Invalid programmatic or CLI-derived gate configuration fails closed instead of producing a misleading eligibility result.

The live collector summary includes a versioned provenance manifest with a UTC collection timestamp, SHA-256 digest over the exact selected dataset files (including their path/order), backend client/model identifiers available from the runtime, and the active CDE policy settings. This makes a saved evaluation run attributable to its evidence and decision configuration without storing API keys or other credentials.

The collector persists that provenance next to the result file as `<output>.manifest.json` using the same temporary-file, fsync, and atomic-replace discipline as result checkpoints. The command summary reports `manifest_output`, so evaluation evidence can be archived together with the exact provenance record instead of relying on terminal output.

Resume is provenance-safe: if a result file already exists, `--resume` requires its sidecar manifest and verifies dataset digest, backend identity, decision policy, runtime execution limits, code revision/cleanliness and selection provenance before reusing any row. Missing, incompatible, changed-dataset, changed-backend, changed-policy, changed-runtime or changed-code provenance fails closed. A successful resumed manifest records the original collection timestamp in `resumed_from_collected_at`.

Promotion evaluation now enforces the same evidence boundary: whenever `--cde-results` is supplied, the evaluator requires the adjacent version-4 manifest and recomputes the digest of the selected datasets before calculating metrics or eligibility. Results collected against different or subsequently modified benchmark evidence are rejected rather than scored.

Fresh collection is overwrite-safe and interruption-resumable: if either the output JSONL or its sidecar already exists, the collector requires `--resume` instead of replacing prior evidence implicitly. For a new output path it atomically materializes an empty JSONL first, records its SHA-256 and `result_rows: 0`, and then persists the matching `collecting` manifest before the first model call. An interruption before row one therefore still leaves a cryptographically bound checkpoint that can be resumed safely.

Each collected result row now records `models_used`, derived from the actual model metadata returned by the decision backend for that row. This complements the manifest's configured primary/fallback model identities and makes mixed-provider or fallback evaluation runs auditable at row granularity.

Completed evaluation manifests are now cryptographically bound to the exact result JSONL as well as the benchmark evidence. Finalization records `status: complete`, `results_sha256` and `result_rows`; promotion evaluation refuses incomplete manifests or result files whose bytes no longer match that digest. Safe resume likewise verifies the digest before reusing a completed result file, while interrupted `status: collecting` checkpoints remain resumable.

Evaluation evidence schema v4 is the hardened provenance contract used by the collector and evaluator. It requires selection scope, result binding, row-level model attribution, runtime execution limits, code revision/cleanliness and redacted backend endpoint identity in addition to the earlier digest/status fields. Collector and evaluator import the same `RESULT_FORMAT_VERSION`, so provenance compatibility cannot drift through duplicated version constants. Older manifests, including v3 artifacts that predate these required fields, fail closed and must be recollected or explicitly migrated rather than being silently accepted under the stronger evidence contract. A v3 manifest must not be upgraded by editing `format_version` alone: required v4 provenance such as selection scope, runtime limits, code cleanliness and endpoint identity may be unknowable after the fact. The safe default is recollection; an explicit migration is only valid when every required v4 field can be independently established from trustworthy contemporaneous evidence.

CDE configuration now fails at settings load, before API/runtime initialization, when probability thresholds, token/question limits, timeouts or aggregate output-budget constraints are invalid. The same bounds remain enforced inside the runtime as defense in depth.

Candidate-aware output budgets are now bound to the exact serialized scorer request instead of being consumed from a positional iterator. That preserves the intended per-question budget even if scorer call ordering changes later, and the tests require the larger candidate set to receive the larger budget.

The `/v1/decide` response preserves the runtime `usage.models` list alongside token counts, so callers can observe which backend model identities actually served structured decisions without relying on configured-provider assumptions.

Promotion evidence now verifies the manifest's `result_rows` against the actual number of non-empty JSONL rows in addition to the SHA-256 binding. Invalid row-count types or count mismatches fail closed, making manifest consistency explicit rather than treating the count as descriptive metadata only.


## Rollout evidence hardening

Dataset provenance hashing now streams file contents in fixed-size chunks while preserving the exact digest algorithm introduced for schema v3 (path, NUL separator, file bytes, NUL separator, in dataset order). Schema v4 deliberately keeps that byte-level digest unchanged, avoiding unnecessary hash churn while strengthening the manifest contract around it.

Repeated resume operations preserve the timestamp of the original collection in `resumed_from_collected_at` instead of replacing it with the immediately preceding resume timestamp. The current `collected_at` still identifies the latest completed materialization, while the root collection provenance remains stable across an arbitrary resume chain.


Schema-v4 promotion and resume paths require each result row to carry a non-empty `models_used` list of unique non-empty strings. Every reported model must also be declared by backend provenance, so a v4 artifact cannot claim row-level model attribution while silently omitting or substituting the serving model.


Promotion now treats the full schema-v4 manifest as a contract, not only its digests: the exact ordered dataset path list must match the evaluator inputs, `collected_at` must be present, and backend/policy provenance must be structured objects. Both dataset and result digests must also be canonical lowercase 64-character SHA-256 values. Structurally incomplete v4 manifests fail closed before metrics are calculated. Backend provenance now requires the runtime client identity plus explicit configured model slots (`model`, `primary_model`, `fallback_model`), with model identities either null or non-empty strings. It also records a redacted endpoint identity for direct, primary and fallback clients plus their local/hosted flags. Endpoint provenance keeps scheme, host and port, while the path is represented only by a SHA-256 digest; embedded credentials, raw paths, query strings and fragments are not persisted. This still distinguishes different backend bases without exposing tenant- or deployment-specific path material. Policy provenance requires all five CDE decision-policy fields with finite, range-valid numeric values and a real boolean OOD-rejection flag. At least one backend model identity must be declared. Promotion and resume also require every result row to report a non-empty `models_used` list, and every reported model must be one of those declared backend identities; syntactically valid rows attributed to an unrelated model fail closed. The collector enforces the same relation before appending a row to any JSONL checkpoint, so invalid model provenance is rejected before result evidence is persisted. Numeric policy fields must also be actual JSON numbers, not numeric-looking strings, matching the runtime manifest exactly rather than relying on coercion. Runtime execution provenance is recorded separately from decision policy: question limits, per-question and aggregate output budgets, aggregate prompt budget, per-call timeout, and request timeout are all validated with the same bounds enforced by `CeltIADecisionRuntime`. Resume requires the full runtime provenance object to match the current process, preventing evidence chains from silently spanning operational configuration changes.

Resume now applies the same evidence boundary. Completed manifests are revalidated through the promotion-manifest validator before any existing row is reused, including result digest and row-count checks. Interrupted `collecting` manifests must still carry the exact ordered dataset list, `collected_at`, structured backend/policy provenance and a canonical dataset SHA-256; malformed or unknown-status sidecars fail closed instead of being treated as resumable checkpoints.

Interrupted checkpoints are now bound to their partial JSONL too. Every persisted checkpoint refreshes `results_sha256` and `result_rows` in the `collecting` manifest, and resume verifies both before loading any existing row. A modified or truncated partial result file therefore fails closed instead of being accepted merely because its rows remain syntactically valid. Fresh collection now begins by atomically materializing an empty JSONL checkpoint and binding it as `result_rows=0` with its SHA-256 before the first model call. This makes failures before row one resumable. Resume also rejects one-sided/orphaned state where only the results file or only the manifest exists.

Collection scope is explicit in schema v4 through `selection.dataset_rows`, `selection.selected_rows` and `selection.limit`. Completed artifacts must contain exactly one result row per selected benchmark row. `--limit` remains valid for pilots and resumable collection, but promotion requires `selected_rows == dataset_rows`, so a deterministic prefix run cannot be mistaken for full-dataset evidence even if its observed routing coverage would otherwise clear the gate. A completed pilot can still be resumed without `--limit`; existing rows are reused, only the remaining benchmark rows are collected, and the final manifest records a full selection. Resume also verifies that every previously collected row belongs to the exact deterministic prefix declared by the prior selection, so a partial artifact cannot swap in different benchmark rows while preserving only the same cardinality.

Collection timestamps are validated as timezone-aware ISO-8601 values rather than arbitrary non-empty strings. Promotion and resume reject malformed or timezone-naive `collected_at` values, preserving an unambiguous audit timeline across hosts and environments. When present, `resumed_from_collected_at` is held to the same requirement so a resume chain cannot preserve malformed root chronology. Resume chronology is also ordered: a root `resumed_from_collected_at` later than the current `collected_at` is rejected as impossible provenance. The manifest also records `code_revision`: the exact Git commit SHA when available, otherwise explicit `null`. Resume requires the stored revision to match the current runtime revision, preventing evidence chains from silently spanning code changes. When a revision is available, `code_dirty` records whether tracked files differ from that commit; resume requires the same clean/dirty state. If Git cannot resolve the revision, both fields are explicitly `null`. Untracked files are ignored so collector outputs themselves do not invalidate a legitimate resume. Collection and resume may operate on dirty code for diagnostic/pilot work, but promotion validation requires a resolved clean revision (`code_dirty == false`), because a dirty checkout is not reproducible from the recorded commit alone.

### Leak-free routing evidence

Routing evidence must compare CDE against the legacy heuristic without exposing the heuristic decision to the scorer. Shadow routing and the live collector therefore pass only the visible user message plus explicit input-size metadata (`input_chars` and `long_context_chars`) into the CDE. The legacy heuristic route is computed separately and recorded only in the result row for comparison.

The five routing modes use a trusted system-level contract shared by shadow and collection: `fast` for direct low-deliberation work, `think` for reasoning/planning/trade-offs, `code` for implementation/debugging/code-centric output, `agent` for external/current/tool-using work, and `long` for genuinely large effective input context. Benchmark rows labeled `long` carry explicit `input_chars` above the production long-context threshold, so the scorer is evaluated on the same size signal production can know instead of inferring long-context needs from words such as “long document”.

Any live evidence collected before this contract change remains useful as historical diagnostic evidence, but it must not be compared as if it were generated by the same scorer contract. New promotion evidence must be recollected from the current clean revision and current dataset digest.
