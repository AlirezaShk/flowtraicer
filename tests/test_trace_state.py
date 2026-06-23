"""TraceState — a reusable base state carrying the xai-drained channels."""

from operator import add
from typing import Annotated

import pytest
from langgraph.graph import END, START, StateGraph

from xai.core.model import EventKind
from xai.langgraph_adapter import TraceState, run_instrumented
from xai.recorder import Recorder
from xai.store.sqlite import SQLiteStore


class _MyState(TraceState):
    """A user state that just extends TraceState with domain fields."""

    messages: Annotated[list, add]
    flag: bool


def _graph():
    def node(state):
        return {
            "messages": ["hi"],
            "tool_calls": [{"name": "search", "payload": {"q": 1}}],
            "llm_calls": [{"name": "m", "prompt_tokens": 3, "completion_tokens": 2}],
            "events": [{"kind": "log", "name": "noted"}],
            "extraction": {"schema_name": "S", "values": {"x": 1}},
        }

    g = StateGraph(_MyState)
    g.add_node("node", node)
    g.add_edge(START, "node")
    g.add_edge("node", END)
    return g.compile()


@pytest.mark.asyncio
async def test_inherited_channels_are_drained_by_the_runner():
    store = SQLiteStore()
    # Note: the xai channels are NOT passed in the initial input — they come from the base.
    eid = await run_instrumented(_graph(), {"messages": [], "flag": True}, Recorder(store))

    step = store.get_engagement(eid).steps[0]
    kinds = {e.kind for e in step.events}
    assert EventKind.TOOL_CALL in kinds
    assert EventKind.LLM_CALL in kinds
    assert EventKind.LOG in kinds
    assert step.total_tokens == 5
    assert step.extraction.values == {"x": 1}
