"""Tests for the LangGraph adapter: topology reading + instrumented run."""

from operator import add
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from ft.core.model import EventKind, StepStatus
from ft.langgraph_adapter import read_topology, run_instrumented
from ft.recorder import Recorder
from ft.store.sqlite import SQLiteStore


class _State(TypedDict):
    msgs: Annotated[list, add]
    tool_calls: Annotated[list, add]
    extraction: dict
    route: str


def _build_graph():
    def greet(s):
        return {"msgs": ["greet"]}

    def qualify(s):
        return {
            "msgs": ["qualify"],
            "tool_calls": [{"name": "lookup_area", "payload": {"area": "Shibuya"}}],
            "extraction": {"schema_name": "BudgetInfo", "values": {"budget": 95000}},
        }

    def search(s):
        return {"msgs": ["search"]}

    def escalate(s):
        return {"msgs": ["escalate"]}

    def router(s):
        return "escalate" if s.get("route") == "escalate" else "search"

    g = StateGraph(_State)
    for n, f in [
        ("greet", greet),
        ("qualify", qualify),
        ("search", search),
        ("escalate", escalate),
    ]:
        g.add_node(n, f)
    g.add_edge(START, "greet")
    g.add_edge("greet", "qualify")
    g.add_conditional_edges("qualify", router, {"search": "search", "escalate": "escalate"})
    g.add_edge("search", END)
    g.add_edge("escalate", END)
    return g.compile()


def test_read_topology_excludes_synthetic_and_marks_global():
    topo = read_topology(
        _build_graph(),
        global_nodes={"escalate"},
        node_tools={"qualify": ["lookup_area"]},
    )
    names = {n.name for n in topo.nodes}
    assert names == {"greet", "qualify", "search", "escalate"}
    assert "__start__" not in names and "__end__" not in names

    by_name = {n.name: n for n in topo.nodes}
    assert by_name["escalate"].is_global is True
    assert by_name["qualify"].tools == ["lookup_area"]

    edge_pairs = {(e.source, e.target) for e in topo.edges}
    assert ("greet", "qualify") in edge_pairs
    assert ("qualify", "escalate") in edge_pairs


async def test_run_instrumented_captures_steps_events_and_extraction():
    store = SQLiteStore()
    rec = Recorder(store)
    eid = await run_instrumented(
        _build_graph(),
        {"msgs": [], "tool_calls": [], "route": "search"},
        rec,
        name="house_search",
        node_tools={"qualify": ["lookup_area"]},
    )

    eng = store.get_engagement(eid)
    step_names = [s.name for s in eng.steps]
    assert step_names == ["greet", "qualify", "search"]
    assert all(s.status is StepStatus.COMPLETED for s in eng.steps)

    qualify = next(s for s in eng.steps if s.name == "qualify")
    assert qualify.tools_available == ["lookup_area"]
    assert qualify.extraction.values["budget"] == 95000
    tool_events = [e for e in qualify.events if e.kind is EventKind.TOOL_CALL]
    assert tool_events[0].name == "lookup_area"
    assert tool_events[0].payload == {"area": "Shibuya"}

    assert eng.topology.nodes  # topology was attached
    assert eng.intent_switches == []


async def test_run_instrumented_records_intent_switch_into_global_step():
    store = SQLiteStore()
    rec = Recorder(store)
    eid = await run_instrumented(
        _build_graph(),
        {"msgs": [], "tool_calls": [], "route": "escalate"},
        rec,
        global_nodes={"escalate"},
    )

    eng = store.get_engagement(eid)
    assert [s.name for s in eng.steps] == ["greet", "qualify", "escalate"]
    escalate = next(s for s in eng.steps if s.name == "escalate")
    assert escalate.is_global is True

    assert len(eng.intent_switches) == 1
    switch = eng.intent_switches[0]
    assert switch.to_step == "escalate"
    assert switch.from_step == "qualify"
