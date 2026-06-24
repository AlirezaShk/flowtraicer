"""Build a temporal viewmodel from an engagement for the timeline view.

Pure, dependency-free transformation: every offset/duration is expressed in milliseconds
relative to the engagement start, so the viewer only has to scale to pixels. Keeping this
in Python means the (slightly fiddly) timing math is covered by the test suite rather than
living untested in the browser.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from .core.model import Engagement, StepStatus


def _ms_between(later: datetime, earlier: datetime) -> float:
    return (later - earlier).total_seconds() * 1000.0


class TimelineMark(BaseModel):
    """A point/interval inside the timeline (an event, or an intent switch)."""

    kind: str
    name: str
    offset_ms: float
    duration_ms: float | None = None


class TimelineLane(BaseModel):
    """One step rendered as a horizontal bar."""

    step_id: str
    name: str
    is_global: bool
    status: StepStatus
    offset_ms: float
    duration_ms: float
    total_tokens: int = 0
    events: list[TimelineMark] = Field(default_factory=list)


class TimelineView(BaseModel):
    """The full temporal layout of one engagement."""

    engagement_id: str
    total_ms: float
    total_tokens: int = 0
    lanes: list[TimelineLane] = Field(default_factory=list)
    intent_switches: list[TimelineMark] = Field(default_factory=list)


def build_timeline(engagement: Engagement) -> TimelineView:
    """Compute the timeline viewmodel for ``engagement`` (offsets relative to its start)."""
    t0 = engagement.started_at
    span = 0.0

    lanes: list[TimelineLane] = []
    for step in engagement.steps:
        offset = max(0.0, _ms_between(step.started_at, t0))
        if step.duration_ms is not None:
            duration = max(0.0, step.duration_ms)
        elif step.ended_at is not None:
            duration = max(0.0, _ms_between(step.ended_at, step.started_at))
        else:
            duration = 0.0
        span = max(span, offset + duration)

        marks: list[TimelineMark] = []
        for event in step.events:
            ev_offset = max(0.0, _ms_between(event.ts, t0))
            span = max(span, ev_offset + (event.duration_ms or 0.0))
            marks.append(
                TimelineMark(
                    kind=event.kind.value,
                    name=event.name,
                    offset_ms=ev_offset,
                    duration_ms=event.duration_ms,
                )
            )

        lanes.append(
            TimelineLane(
                step_id=step.id,
                name=step.name,
                is_global=step.is_global,
                status=step.status,
                offset_ms=offset,
                duration_ms=duration,
                total_tokens=step.total_tokens,
                events=marks,
            )
        )

    lanes.sort(key=lambda lane: lane.offset_ms)

    switches: list[TimelineMark] = []
    for switch in engagement.intent_switches:
        sw_offset = max(0.0, _ms_between(switch.ts, t0))
        span = max(span, sw_offset)
        label = f"{switch.from_step or 'start'} → {switch.to_step}"
        switches.append(TimelineMark(kind="intent_switch", name=label, offset_ms=sw_offset))

    return TimelineView(
        engagement_id=engagement.id,
        total_ms=span,
        total_tokens=engagement.total_tokens,
        lanes=lanes,
        intent_switches=switches,
    )
