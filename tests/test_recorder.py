"""Tests for the recorder — the fail-open emit contract."""

from xai.core.model import EventKind, Extraction, StepStatus
from xai.recorder import Recorder
from xai.store.sqlite import SQLiteStore


def test_recorder_drives_a_full_engagement():
    store = SQLiteStore()
    rec = Recorder(store)

    eid = rec.start_engagement("house_search", metadata={"user_id": "u-7"})
    sid = rec.start_step(eid, "qualify", tools=["lookup_area"])
    rec.record_event(sid, EventKind.TOOL_CALL, "lookup_area", payload={"area": "Shibuya"})
    rec.record_extraction(sid, Extraction(schema_name="BudgetInfo", values={"budget": 95000}))
    rec.end_step(sid, StepStatus.COMPLETED, duration_ms=12.5)
    rec.record_intent_switch(eid, to_step="escalate", reason="handoff", from_step="qualify")
    rec.end_engagement(eid)

    eng = store.get_engagement(eid)
    assert eng.name == "house_search"
    assert eng.metadata["user_id"] == "u-7"
    step = eng.steps[0]
    assert step.name == "qualify"
    assert step.tools_available == ["lookup_area"]
    assert step.status is StepStatus.COMPLETED
    assert step.duration_ms == 12.5
    assert step.extraction.values["budget"] == 95000
    tool_events = [e for e in step.events if e.kind is EventKind.TOOL_CALL]
    assert tool_events[0].payload == {"area": "Shibuya"}
    assert eng.intent_switches[0].to_step == "escalate"


def test_record_event_resolves_engagement_from_step():
    store = SQLiteStore()
    rec = Recorder(store)
    eid = rec.start_engagement("e")
    sid = rec.start_step(eid, "greet")
    # No engagement_id passed — recorder must resolve it from the step.
    rec.record_event(sid, EventKind.LOG, "hello")
    eng = store.get_engagement(eid)
    assert eng.steps[0].events[0].name == "hello"


class _FailingStore:
    def append(self, record):
        raise RuntimeError("disk on fire")


def test_recorder_is_fail_open():
    rec = Recorder(_FailingStore())
    # None of these may raise — instrumentation must never crash the agent.
    eid = rec.start_engagement("e")
    assert eid
    sid = rec.start_step(eid, "greet")
    assert sid
    rec.record_event(sid, EventKind.LOG, "x")
    rec.record_extraction(sid, Extraction(schema_name="S"))
    rec.record_intent_switch(eid, to_step="z", reason="r")
    rec.end_step(sid)
    rec.end_engagement(eid)
