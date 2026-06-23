"""A tiny self-contained LangGraph agent that exercises every part of the model.

Workflow::

    greet -> qualify -> (router) -> search        (happy path)
                                 \\-> escalate     (global re-route)

``qualify`` writes a ``tool_calls`` entry and an ``extraction`` to state, so the captured
trace contains a tool-call event and a per-step extraction. ``escalate`` is a *global*
step: routing into it records an intent switch.

No LLM/network is used — node logic is deterministic, so this runs anywhere (CI-safe).
"""

from __future__ import annotations

from operator import add
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from xai.langgraph_adapter import run_instrumented
from xai.recorder import Recorder

#: Nodes that re-route the workflow's intent when entered.
GLOBAL_NODES = {"escalate"}

#: Per-step tool lists (the graph itself doesn't expose these).
NODE_TOOLS = {"qualify": ["lookup_area"], "search": ["search_properties"]}


class DemoState(TypedDict):
    msgs: Annotated[list, add]
    tool_calls: Annotated[list, add]
    extraction: dict
    route: str


def _greet(state: DemoState) -> dict:
    return {"msgs": ["Hi! Tell me what kind of place you're looking for."]}


def _qualify(state: DemoState) -> dict:
    return {
        "msgs": ["Got it — budget around ¥95,000 in Shibuya."],
        "tool_calls": [{"name": "lookup_area", "payload": {"area": "Shibuya"}}],
        "extraction": {
            "schema_name": "BudgetInfo",
            "json_schema": {
                "type": "object",
                "properties": {"budget": {"type": "integer"}, "area": {"type": "string"}},
            },
            "values": {"budget": 95000, "area": "Shibuya"},
            "confidence": 0.9,
        },
    }


def _search(state: DemoState) -> dict:
    return {
        "msgs": ["I found 3 matching rooms."],
        "tool_calls": [{"name": "search_properties", "payload": {"results": 3}}],
    }


def _escalate(state: DemoState) -> dict:
    return {"msgs": ["Connecting you to a human agent."]}


def _router(state: DemoState) -> str:
    return "escalate" if state.get("route") == "escalate" else "search"


def build_demo_graph():
    """Compile and return the demo LangGraph."""
    g = StateGraph(DemoState)
    g.add_node("greet", _greet)
    g.add_node("qualify", _qualify)
    g.add_node("search", _search)
    g.add_node("escalate", _escalate)
    g.add_edge(START, "greet")
    g.add_edge("greet", "qualify")
    g.add_conditional_edges("qualify", _router, {"search": "search", "escalate": "escalate"})
    g.add_edge("search", END)
    g.add_edge("escalate", END)
    return g.compile()


async def run_demo(recorder: Recorder, *, route: str = "search") -> str:
    """Run the demo agent under instrumentation; return the engagement id.

    ``route="escalate"`` drives the global re-route path.
    """
    return await run_instrumented(
        build_demo_graph(),
        {"msgs": [], "tool_calls": [], "route": route},
        recorder,
        name=f"demo:{route}",
        metadata={"demo": True, "route": route},
        global_nodes=GLOBAL_NODES,
        node_tools=NODE_TOOLS,
    )
