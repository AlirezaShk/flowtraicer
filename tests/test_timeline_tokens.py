"""Timeline viewmodel surfaces per-lane and total token usage."""

from datetime import UTC, datetime, timedelta

from xai.core.model import Engagement, EventKind, Step, StepEvent, TokenUsage
from xai.timeline import build_timeline

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def test_timeline_includes_token_totals():
    eng = Engagement(name="e", started_at=T0)
    step = Step(
        engagement_id=eng.id,
        name="apply",
        started_at=T0,
        ended_at=T0 + timedelta(milliseconds=5),
        duration_ms=5,
    )
    step.events.append(
        StepEvent(
            step_id=step.id,
            kind=EventKind.LLM_CALL,
            name="gpt",
            ts=T0 + timedelta(milliseconds=1),
            tokens=TokenUsage(prompt=10, completion=5),
        )
    )
    eng.steps.append(step)

    tv = build_timeline(eng)
    assert tv.lanes[0].total_tokens == 15
    assert tv.total_tokens == 15
