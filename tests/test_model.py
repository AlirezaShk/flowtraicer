"""Tests for the core data model: Engagement -> Step -> Events."""

from xai.core.model import (
    EdgeDef,
    Engagement,
    EngagementStatus,
    EventKind,
    Extraction,
    IntentSwitch,
    NodeDef,
    Step,
    StepEvent,
    StepStatus,
    Topology,
)


def test_event_has_autoid_and_timestamp():
    ev = StepEvent(step_id="s1", kind=EventKind.TOOL_CALL, name="search_properties")
    assert ev.id
    assert ev.ts is not None
    assert ev.kind is EventKind.TOOL_CALL
    assert ev.payload == {}
    assert ev.error is None


def test_step_defaults_to_running_and_not_global():
    step = Step(engagement_id="e1", name="qualify")
    assert step.id
    assert step.status is StepStatus.RUNNING
    assert step.is_global is False
    assert step.tools_available == []
    assert step.events == []
    assert step.extraction is None


def test_step_carries_per_step_tools_and_extraction():
    extraction = Extraction(
        schema_name="BudgetInfo",
        json_schema={"type": "object", "properties": {"budget": {"type": "integer"}}},
        values={"budget": 120000},
        confidence=0.92,
        valid=True,
    )
    step = Step(
        engagement_id="e1",
        name="qualify",
        tools_available=["lookup_area"],
        extraction=extraction,
    )
    step.events.append(StepEvent(step_id=step.id, kind=EventKind.EXTRACTION, name="BudgetInfo"))
    assert step.tools_available == ["lookup_area"]
    assert step.extraction.values["budget"] == 120000
    assert step.events[0].kind is EventKind.EXTRACTION


def test_intent_switch_records_reroute():
    sw = IntentSwitch(to_step="escalate", reason="user asked for a human", from_step="search")
    assert sw.id
    assert sw.to_step == "escalate"
    assert sw.from_step == "search"
    assert sw.ts is not None


def test_topology_nodes_and_edges():
    topo = Topology(
        nodes=[
            NodeDef(name="greet", tools=[]),
            NodeDef(name="escalate", is_global=True, tools=["page_human"]),
        ],
        edges=[EdgeDef(source="greet", target="qualify")],
    )
    assert topo.nodes[1].is_global is True
    assert topo.edges[0].source == "greet"


def test_engagement_assembles_full_tree():
    eng = Engagement(
        name="house_search",
        metadata={"user_id": "u-42"},
        topology=Topology(nodes=[NodeDef(name="greet")], edges=[]),
    )
    eng.steps.append(Step(engagement_id=eng.id, name="greet"))
    eng.intent_switches.append(IntentSwitch(to_step="escalate", reason="handoff"))
    assert eng.status is EngagementStatus.ACTIVE
    assert eng.metadata["user_id"] == "u-42"
    assert eng.steps[0].name == "greet"
    assert eng.intent_switches[0].to_step == "escalate"
    assert eng.ended_at is None
