# CeltIA Decision Engine (CDE) v0

## Goal
Provide a small, model-agnostic structured-decision primitive for CeltIA: boolean, choice and ordinal score decisions with calibrated probabilities, explicit abstention and expected-value output for ordinal scores.

## Clean implementation boundary
CDE is an independently authored CeltIA component. It does not copy source code, weights, training outputs, APIs, class names, or model-specific implementation from external decision-model projects. General ML ideas such as candidate scoring, softmax probabilities, caching, calibration and abstention are implemented behind CeltIA-owned interfaces.

## Contract
A request contains shared context plus questions. Each question exposes a finite candidate set. A CandidateScorer returns one finite logit per candidate. The engine applies temperature scaling, softmax, and returns the highest-probability candidate or abstains below a confidence threshold.

## v0 invariants
- no generative parsing required by the core;
- model/backbone stays behind CandidateScorer;
- full probability distribution returned;
- ordinal score results also expose the probability-weighted expected score;
- results expose observable abstention diagnostics (`low_confidence` / `suspected_ood`), normalized entropy and decision margin without exposing hidden reasoning;
- `/v1/decide` uses the same gateway concurrency/rate-limit slot as other model-backed requests;
- request-cost bounds are operator-configurable: maximum questions per request (hard-capped at 32), model output tokens per scoring call (64..2048), an aggregate output-token budget divided across all questions, an aggregate serialized prompt-size budget checked before any model call, a per-call timeout (default 30s), and a total request timeout (default 120s);
- model usage from decision scoring is aggregated per request and fed into CeltIA usage/quota/billing accounting;
- API schema and runtime both enforce bounded request/question/option sizes plus full per-type semantics (choice cardinality/uniqueness and valid score ranges); invalid definitions/output fail closed;
- type-specific fields are mutually exclusive: boolean rejects options/bounds, choice rejects score bounds, and score rejects choice options;
- the LLM scorer serializes context/question/candidates as untrusted JSON data and rejects non-serializable context instead of coercing it;
- model-facing score output uses opaque compact candidate IDs (`c0`, `c1`, ...) rather than echoing candidate text, keeping structured output bounded and reducing injection surface;
- scorer JSON parsing is strict about envelope shape, candidate IDs, finite numeric values and duplicate JSON keys;
- low confidence can abstain;
- sync and async engines share the same optional OOD rejection semantics (entropy/margin thresholds);
- policy thresholds are bounded to [0,1] and scorer logits must be finite numeric values (booleans are rejected);
- runtime configuration is validated at construction (finite thresholds/temperature, boolean OOD flag, cost and timeout bounds) so misconfiguration fails before serving decisions;
- direct runtime callers get normalized validation errors for malformed question containers or missing required question fields;
- deterministic tests need no GPU/model downloads.

## Prefix/state caching benchmark
CDE now includes a deterministic pre-tokenization benchmark that renders the exact scorer messages and measures their longest shared character prefix. The report exposes total, reusable and unique serialized characters plus a reuse fraction. CI exercises a multi-question 12k-character shared-context case and requires at least 75% structural reuse.

This is deliberately a structural proxy, not a performance claim: it does not assume tokenizer boundaries, KV-cache compatibility, backend prefix-cache behavior or latency savings. Production caching remains disabled until real-backend evidence is collected.

The repository also includes a real-backend harness. It sends the exact CDE scorer messages for two same-size workloads: one reuses a large shared context across questions, while the control changes the context near the beginning of every request. It reports median/p95 latency, shared-vs-control median reduction, logical usage, and cached prompt tokens when the OpenAI-compatible backend exposes `prompt_tokens_details.cached_tokens`. Demo/offline fallback is rejected rather than benchmarked.

Run both layers with:

```bash
PYTHONPATH=. python scripts/benchmark_decision_prefix.py --context-chars 12000 --questions 8
PYTHONPATH=. python scripts/benchmark_decision_backend_prefix.py --context-chars 12000 --questions 8
```

Backend results are environment-specific and are not a CI performance gate. For evidence before runtime enablement, run the backend harness in both workload orders and on a production-equivalent backend with prefix caching disabled/enabled where the backend supports that control.

## Next milestones
1. Collect backend token/latency/memory evidence with the new harness before enabling caching in the runtime.
2. Expand labeled routing and OOD datasets beyond the initial smoke benchmark.
3. Evaluate LoRA/trained decision heads against the current LLM-scorer baseline.
4. Add controlled-routing rollout only after promotion-gate evidence is sufficient.

## Security
A decision is advisory until the caller policy authorizes an action. Tool execution and privileged actions remain behind CeltIA authorization and sandbox boundaries.
