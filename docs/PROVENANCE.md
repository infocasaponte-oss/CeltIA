# Provenance: CeltIA Decision Engine

The CeltIA Decision Engine (CDE) code in `celtia/decision/` is an independently authored implementation created for CeltIA.

No third-party decision-engine source code, model weights, generated training traces, private chain-of-thought, or copied tests are included in CDE v0.

The design uses general machine-learning/software concepts including finite candidate scoring, probability normalization, confidence thresholds, calibration, caching and model adapters. Before adding any external dataset, model, adapter or source-code component, record its source, exact revision, license, redistribution/training permissions and required notices here or in a machine-readable manifest.

CeltIA repository licensing governs CeltIA-authored code. Third-party dependencies remain governed by their own licenses.


## Machine-readable training-source gate

CDE training inputs should be represented as `TrainingSource` records from `celtia.decision.provenance` before use. Each record carries a stable identifier, source, exact revision, declared license, SHA-256 digest, and explicit training/distribution permission flags.

`validate_training_sources(...)` fails closed on missing provenance, duplicate identifiers, invalid hashes, sources not permitted for training, and—when requested—sources that cannot be redistributed. The validator records declared permissions; it does not replace legal review of a third-party license.
