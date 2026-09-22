import pytest

from asr_evidence import AsrEvidence, inventory


def test_evidence_is_content_free_bounded_and_expires():
    evidence = AsrEvidence(max_samples=3, max_age_s=60)
    import time
    base = time.time()
    for i in range(5):
        evidence.observe("finalization_ms", 100 + i, at=base + i)
    evidence.observe("failure", 0, at=base + 4)
    got = inventory(backend="r2t2", model="m", model_revision="rev",
                    streaming="native", evidence=evidence)["asr"]
    assert got["finalization_ms"]["sample_count"] == 3
    assert got["finalization_ms"]["value"] == 103
    assert set(got["finalization_ms"]) == {
        "value", "sample_count", "measured_at", "ttl_s", "model_revision"}
    assert "text" not in repr(got).lower()
    assert evidence.observations("rev", now=base + 100) == {}


def test_evidence_rejects_unknown_metrics_and_unbounded_configuration():
    with pytest.raises(ValueError):
        AsrEvidence(max_samples=0)
    evidence = AsrEvidence()
    with pytest.raises(ValueError):
        evidence.observe("accuracy", .9)
