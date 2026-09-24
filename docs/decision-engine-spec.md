# CeltIA Decision Engine (CDE) v0

## Goal
Provide a small, model-agnostic structured-decision primitive for CeltIA: boolean, choice and ordinal score decisions with calibrated probabilities and explicit abstention.

## Clean implementation boundary
CDE is an independently authored CeltIA component. It does not copy source code, weights, training outputs, APIs, class names, or model-specific implementation from external decision-model projects. General ML ideas such as candidate scoring, softmax probabilities, caching, calibration and abstention are implemented behind CeltIA-owned interfaces.

## Contract
A request contains shared context plus questions. Each question exposes a finite candidate set. A CandidateScorer returns one finite logit per candidate. The engine applies temperature scaling, softmax, and returns the highest-probability candidate or abstains below a confidence threshold.

## v0 invariants
- no generative parsing required by the core;
- model/backbone stays behind CandidateScorer;
- full probability distribution returned;
- `/v1/decide` uses the same gateway concurrency/rate-limit slot as other model-backed requests;
- invalid definitions/output fail closed;
- low confidence can abstain;
- deterministic tests need no GPU/model downloads.

## Next milestones
1. Local-model scorer using the existing runtime.
2. NLL, Brier, ECE, option-order and OOD/abstention evaluation.
3. Prefix/state caching after benchmarking.
4. /v1/decide using existing authentication/authorization conventions.
5. Evaluate LoRA/trained decision heads against baseline.

## Security
A decision is advisory until the caller policy authorizes an action. Tool execution and privileged actions remain behind CeltIA authorization and sandbox boundaries.
