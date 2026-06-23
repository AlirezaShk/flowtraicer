"""A tiny self-contained LangGraph agent that exercises every part of the model.

Workflow::

    greet -> qualify -> (router) -> search        (happy path)
                                 \\-> escalate     (global re-route)

``qualify`` runs the Instructor-powered :class:`~xai.extraction.Extractor` to pull a
``BudgetInfo`` schema, then writes ``result.as_record()`` to state under ``extraction`` —
the record-via-state path the runner records automatically. ``escalate`` is a *global*
step: routing into it records an intent switch.

No LLM/network is used — a deterministic stub client stands in for a real provider, so this
runs anywhere (CI-safe). In a real agent you would build the extractor with
``Extractor.from_provider("openai/gpt-4o-mini")`` instead.
"""

from __future__ import annotations

from operator import add
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from xai.extraction import Extractor
from xai.langgraph_adapter import run_instrumented
from xai.recorder import Recorder


class BudgetInfo(BaseModel):
    """The per-step schema extracted in ``qualify``."""

    budget: int
    area: str


class _StubExtractionClient:
    """Deterministic stand-in for an Instructor client, so the demo runs offline.

    Swap ``Extractor(_StubExtractionClient())`` for ``Extractor.from_provider(...)`` to
    extract with a real LLM.
    """

    def create(self, *, response_model, messages, **kwargs):
        return response_model(budget=95000, area="Shibuya")


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
    extractor = Extractor(_StubExtractionClient())
    result = extractor.extract(BudgetInfo, "I'm looking in Shibuya, budget about ¥95,000 a month.")
    return {
        "msgs": [f"Got it — ¥{result.value.budget:,} in {result.value.area}."],
        "tool_calls": [{"name": "lookup_area", "payload": {"area": result.value.area}}],
        "extraction": result.as_record().model_dump(),
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
