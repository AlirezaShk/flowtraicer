"""The pluggable store interface.

A backend is append-only. It must support reconstructing a single engagement, listing
engagement summaries, and tailing newly-appended records for live monitoring.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from ..core.model import Engagement, EngagementStatus
from .records import Record


class EngagementSummary(BaseModel):
    """A lightweight listing row for the engagements index.

    Includes ``metadata`` (so journeys can be filtered by ``user_id``/``session_id``
    without re-fetching each engagement), the rolled-up ``total_tokens``, and ``dropped_at``.
    """

    id: str
    name: str
    status: EngagementStatus
    started_at: datetime
    ended_at: datetime | None = None
    step_count: int = 0
    total_tokens: int = 0
    dropped_at: str | None = None
    metadata: dict = {}

    @classmethod
    def from_engagement(cls, eng: Engagement) -> EngagementSummary:
        return cls(
            id=eng.id,
            name=eng.name,
            status=eng.status,
            started_at=eng.started_at,
            ended_at=eng.ended_at,
            step_count=len(eng.steps),
            total_tokens=eng.total_tokens,
            dropped_at=eng.dropped_at,
            metadata=eng.metadata,
        )


def matches(metadata: dict, where: dict | None) -> bool:
    """True if ``metadata`` contains every key/value in ``where`` (None matches all)."""
    if not where:
        return True
    return all(metadata.get(k) == v for k, v in where.items())


@runtime_checkable
class Store(Protocol):
    """Append-only trace store."""

    def append(self, record: Record) -> None:
        """Persist a record (append-only) and publish it to live subscribers."""

    def get_engagement(self, engagement_id: str) -> Engagement:
        """Reconstruct one engagement. Raises ``KeyError`` if unknown."""

    def list_engagements(self, where: dict | None = None) -> list[EngagementSummary]:
        """Return a summary per engagement (oldest first), optionally filtered by metadata."""

    def subscribe(self) -> AsyncIterator[Record]:
        """Yield records as they are appended (for live monitoring)."""
