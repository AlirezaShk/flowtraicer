"""Fold an ordered stream of records into an :class:`Engagement` tree."""

from __future__ import annotations

from collections.abc import Iterable

from ..core.model import Engagement, Step
from .records import (
    EngagementEnded,
    EngagementStarted,
    EventRecorded,
    ExtractionRecorded,
    IntentSwitched,
    Record,
    StepEnded,
    StepStarted,
)


def fold(records: Iterable[Record]) -> Engagement:
    """Replay ``records`` (in append order) into a fully-formed engagement.

    The first record must be an :class:`EngagementStarted`.
    """
    eng: Engagement | None = None
    steps_by_id: dict[str, Step] = {}

    for rec in records:
        if isinstance(rec, EngagementStarted):
            eng = Engagement(
                id=rec.engagement_id,
                name=rec.name,
                started_at=rec.ts,
                metadata=rec.metadata,
                topology=rec.topology,
            )
        elif eng is None:
            raise ValueError("record stream did not start with EngagementStarted")
        elif isinstance(rec, StepStarted):
            step = Step(
                id=rec.step_id,
                engagement_id=rec.engagement_id,
                name=rec.name,
                started_at=rec.ts,
                tools_available=rec.tools,
                is_global=rec.is_global,
                parent_step_id=rec.parent_step_id,
            )
            eng.steps.append(step)
            steps_by_id[step.id] = step
        elif isinstance(rec, EventRecorded):
            step = steps_by_id.get(rec.event.step_id)
            if step is not None:
                step.events.append(rec.event)
        elif isinstance(rec, ExtractionRecorded):
            step = steps_by_id.get(rec.step_id)
            if step is not None:
                step.extraction = rec.extraction
        elif isinstance(rec, IntentSwitched):
            eng.intent_switches.append(rec.intent_switch)
        elif isinstance(rec, StepEnded):
            step = steps_by_id.get(rec.step_id)
            if step is not None:
                step.status = rec.status
                step.ended_at = rec.ended_at
                step.duration_ms = rec.duration_ms
        elif isinstance(rec, EngagementEnded):
            eng.status = rec.status
            eng.ended_at = rec.ended_at

    if eng is None:
        raise ValueError("empty record stream")
    return eng
