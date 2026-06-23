"""Tests for the Workflow orchestration DSL (sugar over LangGraph + run_instrumented)."""

from operator import add
from typing import Annotated, TypedDict

import pytest

from xai.core.model import EngagementStatus, EventKind
from xai.langgraph_adapter import read_topology
from xai.orchestration import Workflow
from xai.recorder import Recorder
from xai.store.sqlite import SQLiteStore


class _State(TypedDict):
    messages: Annotated[list, add]
    tool_calls: Annotated[list, add]
    route: str


def _workflow() -> Workflow:
    wf = Workflow("demo", state_schema=_State, goal_nodes={"finish"})

    @wf.step
    def start(state):
        return {"messages": ["hi"]}

    @wf.step(tools=["search"])
    def work(state):
        return {"messages": ["working"], "tool_calls": [{"name": "search", "payload": {}}]}

    @wf.global_step(tools=["page_human"])
    def escalate(state):
        return {"messages": ["connecting a human"]}

    @wf.step
    def finish(state):
        return {"messages": ["done"]}

    def router(state):
        return "escalate" if state.get("route") == "escalate" else "finish"

    wf.entry("start")
    wf.edge("start", "work")
    wf.branch("work", router, {"escalate": "escalate", "finish": "finish"})
    wf.finish("finish")
    wf.finish("escalate")
    return wf


@pytest.mark.asyncio
async def test_dsl_runs_and_records_steps_tools_and_goal():
    store = SQLiteStore()
    eid = await _workflow().run(
        {"messages": [], "tool_calls": [], "route": "finish"}, Recorder(store)
    )

    eng = store.get_engagement(eid)
    assert [s.name for s in eng.steps] == ["start", "work", "finish"]
    assert eng.status is EngagementStatus.COMPLETED
    work = next(s for s in eng.steps if s.name == "work")
    assert work.tools_available == ["search"]
    assert any(e.kind is EventKind.TOOL_CALL and e.name == "search" for e in work.events)


@pytest.mark.asyncio
async def test_dsl_global_step_records_intent_switch_and_abandonment():
    store = SQLiteStore()
    eid = await _workflow().run(
        {"messages": [], "tool_calls": [], "route": "escalate"}, Recorder(store)
    )

    eng = store.get_engagement(eid)
    escalate = next(s for s in eng.steps if s.name == "escalate")
    assert escalate.is_global is True
    assert len(eng.intent_switches) == 1
    assert eng.status is EngagementStatus.ABANDONED  # goal 'finish' not reached
    assert eng.dropped_at == "escalate"


def test_dsl_exposes_topology_metadata():
    wf = _workflow()
    topo = read_topology(wf.compile(), global_nodes=wf.global_nodes, node_tools=wf.node_tools)
    by_name = {n.name: n for n in topo.nodes}
    assert by_name["escalate"].is_global is True
    assert by_name["work"].tools == ["search"]
