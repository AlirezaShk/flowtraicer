"""Append-only delta records — the store's source of truth.

The whole engagement tree is reconstructed by folding an ordered stream of these
records (see :mod:`.reconstruct`). Keeping the log append-only gives audit
integrity and replay for free, and lets any backend reuse the same fold logic.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter

from ..core.model import (
    EngagementStatus,
    Extraction,
    IntentSwitch,
    StepEvent,
    StepStatus,
    Topology,
)


def _now() -> datetime:
    return datetime.now(UTC)


class _BaseRecord(BaseModel):
    engagement_id: str
    ts: datetime = Field(default_factory=_now)
    # Assigned by the store on append (monotonic ordering key); None until persisted.
    seq: int | None = None


class EngagementStarted(_BaseRecord):
    """An engagement began (carries its name, metadata, and static topology)."""

    type: Literal["engagement_started"] = "engagement_started"
    name: str
    metadata: dict = Field(default_factory=dict)
    topology: Topology = Field(default_factory=Topology)


class StepStarted(_BaseRecord):
    """A step (workflow node) began."""

    type: Literal["step_started"] = "step_started"
    step_id: str
    name: str
    tools: list[str] = Field(default_factory=list)
    is_global: bool = False
    parent_step_id: str | None = None


class EventRecorded(_BaseRecord):
    """An event occurred inside a step."""

    type: Literal["event_recorded"] = "event_recorded"
    event: StepEvent


class ExtractionRecorded(_BaseRecord):
    """A step produced a structured extraction."""

    type: Literal["extraction_recorded"] = "extraction_recorded"
    step_id: str
    extraction: Extraction


class IntentSwitched(_BaseRecord):
    """A global step re-routed the engagement's intent."""

    type: Literal["intent_switched"] = "intent_switched"
    intent_switch: IntentSwitch


class StepEnded(_BaseRecord):
    """A step finished (with its final status and measured duration)."""

    type: Literal["step_ended"] = "step_ended"
    step_id: str
    status: StepStatus = StepStatus.COMPLETED
    ended_at: datetime = Field(default_factory=_now)
    duration_ms: float | None = None


class EngagementEnded(_BaseRecord):
    """An engagement finished (with final status and, if abandoned, the drop-off step)."""

    type: Literal["engagement_ended"] = "engagement_ended"
    status: EngagementStatus = EngagementStatus.COMPLETED
    ended_at: datetime = Field(default_factory=_now)
    dropped_at: str | None = None


class EngagementStatusChanged(_BaseRecord):
    """A non-terminal engagement status transition (e.g. ACTIVE <-> PAUSED).

    Used by human-in-the-loop pause/resume: pausing records ``PAUSED`` (without ending the
    engagement); resuming records ``ACTIVE`` again. Terminal states are still recorded with
    :class:`EngagementEnded`. See ``docs/2026-06-25-checkpoint-resume-design.md``.
    """

    type: Literal["engagement_status_changed"] = "engagement_status_changed"
    status: EngagementStatus


Record = Annotated[
    EngagementStarted
    | StepStarted
    | EventRecorded
    | ExtractionRecorded
    | IntentSwitched
    | StepEnded
    | EngagementEnded
    | EngagementStatusChanged,
    Field(discriminator="type"),
]

#: Adapter used to (de)serialize any record to/from its concrete type.
RecordAdapter: TypeAdapter[Record] = TypeAdapter(Record)
