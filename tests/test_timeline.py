"""Tests for the timeline viewmodel builder."""

from datetime import UTC, datetime, timedelta

from ft.core.model import (
    Engagement,
    EventKind,
    IntentSwitch,
    Step,
    StepEvent,
    StepStatus,
)
from ft.timeline import TimelineView, build_timeline

T0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)


def _ms(n: float) -> datetime:
    return T0 + timedelta(milliseconds=n)


def _engagement() -> Engagement:
    eng = Engagement(name="house_search", started_at=T0)
    greet = Step(
        engagement_id=eng.id,
        name="greet",
        started_at=T0,
        ended_at=_ms(10),
        duration_ms=10,
        status=StepStatus.COMPLETED,
    )
    greet.events.append(
        StepEvent(
            step_id=greet.id, kind=EventKind.TOOL_CALL, name="lookup", ts=_ms(3), duration_ms=2
        )
    )
    qualify = Step(
        engagement_id=eng.id,
        name="qualify",
        started_at=_ms(10),
        ended_at=_ms(25),
        duration_ms=15,
        is_global=True,
        status=StepStatus.COMPLETED,
    )
    eng.steps.extend([greet, qualify])
    eng.intent_switches.append(
        IntentSwitch(to_step="qualify", from_step="greet", reason="reroute", ts=_ms(12))
    )
    return eng


def test_build_timeline_total_span():
    tv = build_timeline(_engagement())
    assert isinstance(tv, TimelineView)
    assert tv.total_ms == 25  # T0 .. last ended_at (qualify @25)


def test_lane_offsets_and_durations_relative_to_start():
    tv = build_timeline(_engagement())
    greet, qualify = tv.lanes
    assert greet.name == "greet"
    assert greet.offset_ms == 0
    assert greet.duration_ms == 10
    assert qualify.offset_ms == 10
    assert qualify.duration_ms == 15
    assert qualify.is_global is True


def test_event_marks_are_offset_within_engagement():
    tv = build_timeline(_engagement())
    mark = tv.lanes[0].events[0]
    assert mark.kind == "tool_call"
    assert mark.name == "lookup"
    assert mark.offset_ms == 3
    assert mark.duration_ms == 2


def test_intent_switches_become_marks():
    tv = build_timeline(_engagement())
    assert len(tv.intent_switches) == 1
    sw = tv.intent_switches[0]
    assert sw.offset_ms == 12
    assert "greet" in sw.name and "qualify" in sw.name


def test_lanes_sorted_by_offset():
    eng = _engagement()
    eng.steps.reverse()  # feed out of order
    tv = build_timeline(eng)
    assert [lane.offset_ms for lane in tv.lanes] == [0, 10]


def test_step_without_duration_falls_back_to_wallclock():
    eng = Engagement(name="e", started_at=T0)
    eng.steps.append(
        Step(engagement_id=eng.id, name="s", started_at=T0, ended_at=_ms(7), duration_ms=None)
    )
    tv = build_timeline(eng)
    assert tv.lanes[0].duration_ms == 7


def test_running_step_without_end_has_zero_duration():
    eng = Engagement(name="e", started_at=T0)
    eng.steps.append(
        Step(engagement_id=eng.id, name="s", started_at=T0, ended_at=None, duration_ms=None)
    )
    tv = build_timeline(eng)
    assert tv.lanes[0].duration_ms == 0
    assert tv.total_ms >= 0
