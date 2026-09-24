from celtia.decision.provenance import TrainingSource, validate_training_sources


def _source(**overrides):
    values = {
        "id": "example",
        "source": "https://example.invalid/dataset",
        "revision": "v1",
        "license": "Example-License",
        "sha256": "a" * 64,
        "allowed_for_training": True,
        "allowed_for_distribution": False,
    }
    values.update(overrides)
    return TrainingSource(**values)


def test_training_source_manifest_accepts_training_only_source():
    rows = validate_training_sources([_source()])
    assert rows[0].id == "example"


def test_training_source_manifest_rejects_unlicensed_training():
    try:
        validate_training_sources([_source(allowed_for_training=False)])
        assert False
    except ValueError as exc:
        assert "not permitted for training" in str(exc)


def test_training_source_manifest_enforces_distribution_permission():
    try:
        validate_training_sources([_source()], require_distribution=True)
        assert False
    except ValueError as exc:
        assert "not permitted for distribution" in str(exc)


def test_training_source_manifest_rejects_bad_hash_and_duplicate_ids():
    for rows in (
        [_source(sha256="bad")],
        [_source(), _source()],
    ):
        try:
            validate_training_sources(rows)
            assert False
        except ValueError:
            pass
