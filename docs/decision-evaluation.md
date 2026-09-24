# CDE shadow evaluation and promotion gate

CDE routing starts in shadow mode. The production heuristic remains authoritative while CeltIA records only route names, confidence, abstention and agreement. User prompt text is not stored in the decision-shadow table.

The admin report exposes sample count, agreement, abstention, mean confidence, mean normalized entropy, mean decision margin, abstention-reason counts and common disagreement pairs.

Promotion to controlled routing must not be based on agreement alone. A labeled evaluation set is required. The offline evaluation helper compares both the existing heuristic and CDE against expected routes and applies explicit minimum sample, coverage, labeled-sample and accuracy-delta gates.

Default gate:
- at least 200 total samples;
- at least 80% CDE decision coverage;
- at least 50 labeled samples;
- at least 80% coverage on labeled samples;
- CDE labeled accuracy at least 2 percentage points above the heuristic.

Passing the gate means eligible for a controlled experiment, not automatic activation. Tool authorization and security policy remain independent of routing.


Evaluation also reports selective accuracy: accuracy only on labeled samples where CDE actually returns a decision. This is reported alongside total labeled accuracy and labeled coverage so abstention cannot hide errors or inflate the promotion result.


## Option-order evaluation harness

`celtia.decision.order_eval` can evaluate a scorer across a bounded, deterministic set of candidate permutations. It always includes the canonical order and reverse order, then adds rotations up to `max_orders` (default 8), avoiding factorial growth for large candidate sets.

The report keeps distributions label-aligned and records maximum probability deviation plus whether the winning label changes across orders. This is an evaluation path only; production decisions are not multiplied by permutation testing.


Shadow telemetry stores only routing metadata and uncertainty diagnostics. It does not persist the user prompt. Existing databases are migrated in place by adding nullable `abstention_reason`, `normalized_entropy`, and `margin` columns.
