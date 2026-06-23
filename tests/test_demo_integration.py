"""End-to-end integration test: the shipped demo agent, captured by xai."""

from xai.core.model import EngagementStatus, EventKind, StepStatus
from xai.examples.demo_agent import GLOBAL_NODES, run_demo
from xai.recorder import Recorder
from xai.store.sqlite import SQLiteStore


async def test_demo_happy_path_is_fully_captured():
    store = SQLiteStore()
    rec = Recorder(store)
    eid = await run_demo(rec, route="search")

    eng = store.get_engagement(eid)
    assert eng.status is EngagementStatus.COMPLETED
    assert [s.name for s in eng.steps] == ["greet", "qualify", "search"]
    assert all(s.status is StepStatus.COMPLETED for s in eng.steps)

    qualify = next(s for s in eng.steps if s.name == "qualify")
    assert qualify.extraction is not None
    assert "budget" in qualify.extraction.values
    assert any(e.kind is EventKind.TOOL_CALL for e in qualify.events)

    # Every step has a positive measured duration.
    assert all(s.duration_ms is not None and s.duration_ms >= 0 for s in eng.steps)
    assert eng.intent_switches == []


async def test_demo_escalation_path_records_intent_switch():
    store = SQLiteStore()
    rec = Recorder(store)
    eid = await run_demo(rec, route="escalate")

    eng = store.get_engagement(eid)
    assert "escalate" in [s.name for s in eng.steps]
    escalate = next(s for s in eng.steps if s.name == "escalate")
    assert escalate.is_global is True
    assert escalate.name in GLOBAL_NODES

    assert len(eng.intent_switches) == 1
    assert eng.intent_switches[0].to_step == "escalate"
    assert eng.intent_switches[0].from_step == "qualify"
