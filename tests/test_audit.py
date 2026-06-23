"""Tests for tamper-evident engagement digests."""

from datetime import UTC, datetime

from xai.audit import engagement_digest, verify
from xai.core.model import Engagement, Step

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _engagement() -> Engagement:
    return Engagement(
        id="e1",
        name="house_search",
        started_at=T0,
        metadata={"user_id": "u1"},
        steps=[Step(id="s1", engagement_id="e1", name="greet", started_at=T0)],
    )


def test_digest_is_deterministic():
    assert engagement_digest(_engagement()) == engagement_digest(_engagement())


def test_digest_changes_when_any_content_changes():
    a, b = _engagement(), _engagement()
    b.steps[0].name = "tampered"
    assert engagement_digest(a) != engagement_digest(b)


def test_verify_detects_tampering():
    eng = _engagement()
    digest = engagement_digest(eng)
    assert verify(eng, digest) is True

    eng.metadata["injected"] = "x"  # alter the recorded trail
    assert verify(eng, digest) is False
