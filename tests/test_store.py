"""Tests for the append-only store: record folding + SQLite backend + live subscribe."""

import pytest

from xai.core.model import (
    EngagementStatus,
    EventKind,
    Extraction,
    IntentSwitch,
    NodeDef,
    StepEvent,
    StepStatus,
    Topology,
)
from xai.store.reconstruct import fold
from xai.store.records import (
    EngagementEnded,
    EngagementStarted,
    EventRecorded,
    ExtractionRecorded,
    IntentSwitched,
    StepEnded,
    StepStarted,
)
from xai.store.sqlite import SQLiteStore


def _full_record_stream(engagement_id="e1", step_id="s1"):
    return [
        EngagementStarted(
            engagement_id=engagement_id,
            name="house_search",
            metadata={"user_id": "u-42"},
            topology=Topology(
                nodes=[NodeDef(name="greet"), NodeDef(name="escalate", is_global=True)]
            ),
        ),
        StepStarted(
            engagement_id=engagement_id,
            step_id=step_id,
            name="qualify",
            tools=["lookup_area"],
        ),
        EventRecorded(
            engagement_id=engagement_id,
            event=StepEvent(step_id=step_id, kind=EventKind.TOOL_CALL, name="lookup_area"),
        ),
        ExtractionRecorded(
            engagement_id=engagement_id,
            step_id=step_id,
            extraction=Extraction(schema_name="BudgetInfo", values={"budget": 120000}),
        ),
        StepEnded(
            engagement_id=engagement_id,
            step_id=step_id,
            status=StepStatus.COMPLETED,
            duration_ms=42.0,
        ),
        IntentSwitched(
            engagement_id=engagement_id,
            intent_switch=IntentSwitch(to_step="escalate", reason="handoff", from_step="qualify"),
        ),
        EngagementEnded(engagement_id=engagement_id, status=EngagementStatus.COMPLETED),
    ]


def test_fold_reconstructs_engagement_tree():
    eng = fold(_full_record_stream())

    assert eng.id == "e1"
    assert eng.name == "house_search"
    assert eng.metadata["user_id"] == "u-42"
    assert eng.status is EngagementStatus.COMPLETED
    assert eng.ended_at is not None
    assert len(eng.topology.nodes) == 2

    assert len(eng.steps) == 1
    step = eng.steps[0]
    assert step.name == "qualify"
    assert step.tools_available == ["lookup_area"]
    assert step.status is StepStatus.COMPLETED
    assert step.duration_ms == 42.0
    assert step.extraction.values["budget"] == 120000
    assert step.events[0].kind is EventKind.TOOL_CALL

    assert len(eng.intent_switches) == 1
    assert eng.intent_switches[0].to_step == "escalate"


def test_sqlite_append_and_get_roundtrip():
    store = SQLiteStore()  # in-memory by default
    for rec in _full_record_stream():
        store.append(rec)

    eng = store.get_engagement("e1")
    assert eng.name == "house_search"
    assert eng.steps[0].extraction.values["budget"] == 120000
    assert eng.status is EngagementStatus.COMPLETED


def test_sqlite_list_engagements_summaries():
    store = SQLiteStore()
    for rec in _full_record_stream(engagement_id="e1"):
        store.append(rec)
    for rec in _full_record_stream(engagement_id="e2"):
        store.append(rec)

    summaries = store.list_engagements()
    ids = {s.id for s in summaries}
    assert ids == {"e1", "e2"}
    by_id = {s.id: s for s in summaries}
    assert by_id["e1"].name == "house_search"
    assert by_id["e1"].step_count == 1
    assert by_id["e1"].status is EngagementStatus.COMPLETED


def test_get_unknown_engagement_raises():
    store = SQLiteStore()
    with pytest.raises(KeyError):
        store.get_engagement("nope")


async def test_subscribe_receives_appended_records():
    store = SQLiteStore()
    received = []

    async def consume():
        async for rec in store.subscribe():
            received.append(rec)
            if len(received) == 2:
                break

    import asyncio

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)  # let the subscriber register
    store.append(EngagementStarted(engagement_id="e9", name="x"))
    store.append(EngagementEnded(engagement_id="e9", status=EngagementStatus.COMPLETED))
    await asyncio.wait_for(task, timeout=1.0)

    assert received[0].engagement_id == "e9"
    assert isinstance(received[1], EngagementEnded)
