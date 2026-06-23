"""Tamper-evidence for the audit trail — a deterministic digest of an engagement.

:func:`engagement_digest` is a SHA-256 fingerprint of an engagement's full content. Store
it when the engagement ends; later, recompute and compare to detect any alteration of the
recorded trail (a changed extraction, a deleted tool call, an injected step …).

Threat model: this detects accidental corruption and any tampering by a party that cannot
*also* rewrite the stored digest. For strong tamper-evidence, anchor the digest outside the
trace store — append it to write-once/WORM storage, a transparency log, or sign it — so the
fingerprint can't be rewritten alongside the data it protects.
"""

from __future__ import annotations

import hashlib
import json

from .core.model import Engagement


def engagement_digest(engagement: Engagement) -> str:
    """Return a deterministic SHA-256 hex digest of ``engagement``'s full content."""
    canonical = json.dumps(
        engagement.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify(engagement: Engagement, expected_digest: str) -> bool:
    """True if ``engagement`` still matches ``expected_digest`` (untampered)."""
    return engagement_digest(engagement) == expected_digest
