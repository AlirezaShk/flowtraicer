"""The pluggable store interface.

A backend is append-only. It must support reconstructing a single engagement, listing
engagement summaries, and tailing newly-appended records for live monitoring.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from xai.core.model import Engagement, EngagementStatus
from xai.store.records import Record


class EngagementSummary(BaseModel):
    """A lightweight listing row for the engagements index."""

    id: str
    name: str
    status: EngagementStatus
    started_at: datetime
    ended_at: datetime | None = None
    step_count: int = 0

    @classmethod
    def from_engagement(cls, eng: Engagement) -> EngagementSummary:
        return cls(
            id=eng.id,
            name=eng.name,
            status=eng.status,
            started_at=eng.started_at,
            ended_at=eng.ended_at,
            step_count=len(eng.steps),
        )


@runtime_checkable
class Store(Protocol):
    """Append-only trace store."""

    def append(self, record: Record) -> None:
        """Persist a record (append-only) and publish it to live subscribers."""

    def get_engagement(self, engagement_id: str) -> Engagement:
        """Reconstruct one engagement. Raises ``KeyError`` if unknown."""

    def list_engagements(self) -> list[EngagementSummary]:
        """Return a summary per known engagement, oldest first."""

    def subscribe(self) -> AsyncIterator[Record]:
        """Yield records as they are appended (for live monitoring)."""
