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
    type: Literal["engagement_started"] = "engagement_started"
    name: str
    metadata: dict = Field(default_factory=dict)
    topology: Topology = Field(default_factory=Topology)


class StepStarted(_BaseRecord):
    type: Literal["step_started"] = "step_started"
    step_id: str
    name: str
    tools: list[str] = Field(default_factory=list)
    is_global: bool = False
    parent_step_id: str | None = None


class EventRecorded(_BaseRecord):
    type: Literal["event_recorded"] = "event_recorded"
    event: StepEvent


class ExtractionRecorded(_BaseRecord):
    type: Literal["extraction_recorded"] = "extraction_recorded"
    step_id: str
    extraction: Extraction


class IntentSwitched(_BaseRecord):
    type: Literal["intent_switched"] = "intent_switched"
    intent_switch: IntentSwitch


class StepEnded(_BaseRecord):
    type: Literal["step_ended"] = "step_ended"
    step_id: str
    status: StepStatus = StepStatus.COMPLETED
    ended_at: datetime = Field(default_factory=_now)
    duration_ms: float | None = None


class EngagementEnded(_BaseRecord):
    type: Literal["engagement_ended"] = "engagement_ended"
    status: EngagementStatus = EngagementStatus.COMPLETED
    ended_at: datetime = Field(default_factory=_now)


Record = Annotated[
    EngagementStarted
    | StepStarted
    | EventRecorded
    | ExtractionRecorded
    | IntentSwitched
    | StepEnded
    | EngagementEnded,
    Field(discriminator="type"),
]

#: Adapter used to (de)serialize any record to/from its concrete type.
RecordAdapter: TypeAdapter[Record] = TypeAdapter(Record)
