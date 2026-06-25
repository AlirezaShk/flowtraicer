"""ctx.emit from inside an AgentTool.handler (NEEDS.md #B / #15).

SP4 wraps the app's tools as ``ft.agent.AgentTool``s whose ``handler(args, ctx)`` must push tool
render-events (cards) to the caller via ``ctx.emit(...)`` DURING the ``ctx.run_tools`` agentic loop.
These tests lock that:

1. The ``ctx`` passed to a tool handler IS the running step context and exposes ``emit``.
2. Emits from within ``run_tools`` surface as ``emit`` StreamEvents during ``stream`` /
   ``stream_resume`` (interleaved with the agentic loop).
3. The same emit is harmless (a no-op) under non-streaming ``run`` / ``start`` / ``resume``.
"""

from operator import add
from typing import Annotated

import pytest

from ft.agent import AgentTool
from ft.core.model import TokenUsage
from ft.langgraph_adapter import TraceState
from ft.llm import LLMResult, ToolRequest
from ft.orchestration import Workflow
from ft.recorder import Recorder
from ft.store.sqlite import SQLiteStore


class _State(TraceState):
    messages: Annotated[list, add]


class _ScriptedToolLLM:
    """Emits a scripted sequence of tool-call rounds, then a final answer."""

    def __init__(self, script: list) -> None:
        self._script = list(script)

    async def acomplete(self, messages, *, tools=None, **overrides) -> LLMResult:
        step = self._script.pop(0)
        if isinstance(step, list):
            return LLMResult("", TokenUsage(prompt=5, completion=2), "fake", tool_calls=step)
        return LLMResult(step, TokenUsage(prompt=3, completion=4), "fake")


def _emit_card_tool(card_type: str) -> AgentTool:
    def handler(args, ctx):
        ctx.emit({"type": card_type, "data": args})  # push a render card mid-loop
        return {"ok": True}

    return AgentTool(
        name="compare_schools",
        description="Compare schools and render a comparison card.",
        parameters={"type": "object", "properties": {"a": {"type": "string"}}},
        handler=handler,
    )


def _workflow_with_emitting_tool() -> Workflow:
    wf = Workflow("emit_handler", state_schema=_State)

    @wf.agent_step(tools=[_emit_card_tool("school_comparison")], max_iterations=4)
    async def qa(state, ctx):
        return {"messages": [await ctx.run_tools(state["messages"])]}

    wf.entry("qa")
    wf.finish("qa")
    return wf


@pytest.mark.asyncio
async def test_handler_emit_surfaces_as_emit_stream_event():
    llm = _ScriptedToolLLM(
        script=[[ToolRequest(name="compare_schools", args={"a": "Tokyo"})], "Tokyo wins."]
    )
    wf = _workflow_with_emitting_tool()
    recorder = Recorder(SQLiteStore())

    events = [
        ev async for ev in wf.stream({"messages": ["compare"]}, recorder, thread_id="e1", llm=llm)
    ]

    emits = [e for e in events if e.kind == "emit"]
    assert emits, "ctx.emit from inside the tool handler must surface as an emit StreamEvent"
    assert emits[0].data == {"type": "school_comparison", "data": {"a": "Tokyo"}}
    assert events[-1].kind == "completed"


@pytest.mark.asyncio
async def test_handler_emit_is_noop_under_non_streaming_run():
    """Calling ctx.emit from a handler under run() (no stream) is a harmless no-op (no crash)."""
    llm = _ScriptedToolLLM(
        script=[[ToolRequest(name="compare_schools", args={"a": "Osaka"})], "done"]
    )
    wf = _workflow_with_emitting_tool()
    recorder = Recorder(SQLiteStore())

    # run() (non-streaming) must complete normally despite the handler calling ctx.emit.
    engagement_id = await wf.run({"messages": ["compare"]}, recorder, llm=llm)
    eng = recorder._store.get_engagement(engagement_id)
    qa_step = next(s for s in eng.steps if s.name == "qa")
    # The tool still ran (recorded as a tool_call); the emit was simply a no-op.
    assert any(e.name == "compare_schools" for e in qa_step.events)


@pytest.mark.asyncio
async def test_handler_emit_is_noop_under_non_streaming_start():
    """ctx.emit from a handler under start() (non-streaming, checkpointed) is also a no-op."""
    llm = _ScriptedToolLLM(
        script=[[ToolRequest(name="compare_schools", args={"a": "Kyoto"})], "done"]
    )
    wf = _workflow_with_emitting_tool()
    recorder = Recorder(SQLiteStore())

    turn = await wf.start({"messages": ["compare"]}, recorder, thread_id="e2", llm=llm)
    assert turn.status == "completed"  # no crash; emit was a no-op
