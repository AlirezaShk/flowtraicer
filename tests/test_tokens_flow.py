"""Tests for token recording + abandonment through recorder and the LangGraph runner."""

from operator import add
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from ft.core.model import EngagementStatus, EventKind, StepStatus
from ft.langgraph_adapter import run_instrumented
from ft.recorder import Recorder
from ft.store.sqlite import SQLiteStore


def test_recorder_record_llm_call():
    store = SQLiteStore()
    rec = Recorder(store)
    eid = rec.start_engagement("e")
    sid = rec.start_step(eid, "apply")
    rec.record_llm_call(sid, "gpt-4o", prompt=100, completion=20, duration_ms=5, model="gpt-4o")
    rec.end_step(sid)
    rec.end_engagement(eid)

    eng = store.get_engagement(eid)
    step = eng.steps[0]
    ev = next(e for e in step.events if e.kind is EventKind.LLM_CALL)
    assert ev.tokens.total == 120
    assert ev.payload["model"] == "gpt-4o"
    assert ev.duration_ms == 5
    assert step.total_tokens == 120
    assert eng.total_tokens == 120


class _State(TypedDict):
    msgs: Annotated[list, add]
    llm_calls: Annotated[list, add]
    events: Annotated[list, add]
    go: bool


def _graph():
    def intake(s):
        return {
            "msgs": ["hi"],
            "llm_calls": [{"name": "gpt", "prompt_tokens": 10, "completion_tokens": 4}],
            "events": [{"kind": "log", "name": "intake_note", "payload": {"x": 1}}],
        }

    def submitted(s):
        return {"msgs": ["done"]}

    def router(s):
        return "submitted" if s.get("go") else "end"

    g = StateGraph(_State)
    g.add_node("intake", intake)
    g.add_node("submitted", submitted)
    g.add_edge(START, "intake")
    g.add_conditional_edges("intake", router, {"submitted": "submitted", "end": END})
    g.add_edge("submitted", END)
    return g.compile()


async def test_runner_drains_llm_calls_into_token_events():
    store = SQLiteStore()
    rec = Recorder(store)
    eid = await run_instrumented(
        _graph(), {"msgs": [], "llm_calls": [], "events": [], "go": True}, rec
    )

    eng = store.get_engagement(eid)
    intake = next(s for s in eng.steps if s.name == "intake")
    llm = next(e for e in intake.events if e.kind is EventKind.LLM_CALL)
    assert llm.tokens.total == 14
    assert intake.total_tokens == 14
    assert eng.total_tokens == 14


async def test_runner_drains_generic_events():
    store = SQLiteStore()
    rec = Recorder(store)
    eid = await run_instrumented(
        _graph(), {"msgs": [], "llm_calls": [], "events": [], "go": True}, rec
    )

    eng = store.get_engagement(eid)
    intake = next(s for s in eng.steps if s.name == "intake")
    log = next(e for e in intake.events if e.kind is EventKind.LOG)
    assert log.name == "intake_note"
    assert log.payload == {"x": 1}


async def test_runner_marks_abandoned_when_goal_not_reached():
    store = SQLiteStore()
    rec = Recorder(store)
    eid = await run_instrumented(
        _graph(),
        {"msgs": [], "llm_calls": [], "events": [], "go": False},
        rec,
        goal_nodes={"submitted"},
    )

    eng = store.get_engagement(eid)
    assert eng.status is EngagementStatus.ABANDONED
    assert eng.dropped_at == "intake"
    assert [s.name for s in eng.steps] == ["intake"]


async def test_runner_completed_when_goal_reached():
    store = SQLiteStore()
    rec = Recorder(store)
    eid = await run_instrumented(
        _graph(),
        {"msgs": [], "llm_calls": [], "events": [], "go": True},
        rec,
        goal_nodes={"submitted"},
    )

    eng = store.get_engagement(eid)
    assert eng.status is EngagementStatus.COMPLETED
    assert eng.dropped_at is None
    assert eng.steps[-1].status is StepStatus.COMPLETED
