"""Retention — purge whole engagements past their retention window.

The trace is append-only *within* an engagement (an audit trail you don't rewrite);
retention operates at the engagement level — dropping entire old engagements once they age
out, which is the compliant way to bound how long user data is kept.
"""

from __future__ import annotations

from datetime import datetime, timedelta


def purge_before(store, cutoff: datetime) -> list[str]:
    """Purge every *completed* engagement that ended before ``cutoff``.

    Active engagements (no ``ended_at``) are never purged. Returns the purged ids.
    """
    purged: list[str] = []
    for summary in store.list_engagements():
        if summary.ended_at is not None and summary.ended_at < cutoff and store.purge(summary.id):
            purged.append(summary.id)
    return purged


class RetentionPolicy:
    """Keep engagements for at most ``max_age``; older ones are purged on :meth:`apply`."""

    def __init__(self, *, max_age: timedelta) -> None:
        self.max_age = max_age

    def apply(self, store, *, now: datetime) -> list[str]:
        """Purge engagements that ended more than ``max_age`` before ``now``."""
        return purge_before(store, now - self.max_age)
