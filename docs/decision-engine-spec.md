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
- model usage from decision scoring is aggregated per request and fed into CeltIA usage/quota/billing accounting;
- API schema and runtime both enforce bounded request/question/option sizes; invalid definitions/output fail closed;
- low confidence can abstain;
- sync and async engines share the same optional OOD rejection semantics (entropy/margin thresholds);
- deterministic tests need no GPU/model downloads.

## Next milestones
1. Benchmark prefix/state caching before adding it to the runtime.
2. Expand labeled routing and OOD datasets beyond the initial smoke benchmark.
3. Evaluate LoRA/trained decision heads against the current LLM-scorer baseline.
4. Add controlled-routing rollout only after promotion-gate evidence is sufficient.

## Security
A decision is advisory until the caller policy authorizes an action. Tool execution and privileged actions remain behind CeltIA authorization and sandbox boundaries.
