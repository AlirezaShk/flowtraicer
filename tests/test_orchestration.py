"""Tests for the Workflow orchestration DSL (sugar over LangGraph + run_instrumented)."""

from operator import add
from types import SimpleNamespace
from typing import Annotated, TypedDict

import pytest

from xai.core.model import EngagementStatus, EventKind
from xai.langgraph_adapter import TraceState, read_topology
from xai.llm import LiteLLMClient
from xai.orchestration import Workflow
from xai.recorder import Recorder
from xai.store.sqlite import SQLiteStore


class _State(TypedDict):
    messages: Annotated[list, add]
    tool_calls: Annotated[list, add]
    route: str


def _stub_llm() -> LiteLLMClient:
    async def _acompletion(**kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="hi there"))],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3, total_tokens=8),
        )

    return LiteLLMClient(provider="stub", model="m", _acompletion=_acompletion)


class _LLMState(TraceState):
    messages: Annotated[list, add]


@pytest.mark.asyncio
async def test_workflow_injects_llm_and_auto_records_tokens():
    wf = Workflow("talk", state_schema=_LLMState, llm=_stub_llm())

    @wf.step
    async def talk(state, ctx):
        text = await ctx.llm("say hi")  # injected client; token cost auto-recorded
        return {"messages": [text]}

    wf.entry("talk")
    wf.finish("talk")

    store = SQLiteStore()
    eid = await wf.run({"messages": []}, Recorder(store))

    step = store.get_engagement(eid).steps[0]
    llm_events = [e for e in step.events if e.kind is EventKind.LLM_CALL]
    assert len(llm_events) == 1
    assert step.total_tokens == 8


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
