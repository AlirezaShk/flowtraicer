"""Incremental streaming of a turn (NEEDS.md #2).

``wf.stream(...)`` / ``wf.stream_resume(...)`` are async generators that yield events as the turn
executes (step_started, token, emit, step_finished), terminating on the SAME paused/completed
boundary as ``start``/``resume``. The terminal event carries the resulting ``WorkflowTurn``.
"""

from operator import add
from typing import Annotated

import pytest

from ft.core.model import TokenUsage
from ft.langgraph_adapter import TraceState
from ft.llm import LLMResult
from ft.orchestration import Workflow
from ft.recorder import Recorder
from ft.store.sqlite import SQLiteStore


class _State(TraceState):
    messages: Annotated[list, add]


class _StreamingLLM:
    """A fake client supporting streamed completions: yields token chunks, totals the usage."""

    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks

    async def acomplete(self, messages, **overrides) -> LLMResult:
        text = "".join(self._chunks)
        return LLMResult(text=text, tokens=TokenUsage(prompt=4, completion=3), model="fake")

    async def astream(self, messages, **overrides):
        """Yield text chunks, then a final LLMResult carrying token usage."""
        for c in self._chunks:
            yield c
        yield LLMResult(
            text="".join(self._chunks), tokens=TokenUsage(prompt=4, completion=3), model="fake"
        )


@pytest.mark.asyncio
async def test_stream_yields_tokens_and_terminal_completed():
    llm = _StreamingLLM(chunks=["Hel", "lo ", "world"])

    wf = Workflow("stream", state_schema=_State)

    @wf.step
    async def answer(state, ctx):
        text = await ctx.llm("say hello", stream=True)
        return {"messages": [text]}

    wf.entry("answer")
    wf.finish("answer")

    recorder = Recorder(SQLiteStore())
    events = []
    async for ev in wf.stream({"messages": []}, recorder, thread_id="s1", llm=llm):
        events.append(ev)

    kinds = [e.kind for e in events]
    assert "step_started" in kinds
    assert "token" in kinds
    assert "step_finished" in kinds
    assert kinds[-1] == "completed"

    # Tokens streamed through the iterator reconstruct the answer.
    tokens = [e.data["text"] for e in events if e.kind == "token"]
    assert "".join(tokens) == "Hello world"

    # The terminal event carries the resulting WorkflowTurn with per-turn token usage.
    terminal = events[-1]
    assert terminal.turn is not None
    assert terminal.turn.status == "completed"
    assert terminal.turn.token_usage.total == 7  # 4 + 3, still accounted into the step trace

    # Tokens are still recorded into the step trace (llm_call event).
    eng = recorder._store.get_engagement(terminal.turn.engagement_id)
    step = next(s for s in eng.steps if s.name == "answer")
    assert step.total_tokens == 7


@pytest.mark.asyncio
async def test_stream_pauses_with_emit_and_terminal_paused():
    llm = _StreamingLLM(chunks=["x"])

    wf = Workflow("stream2", state_schema=_State)

    @wf.step
    async def qualify(state, ctx):
        reply = await ctx.pause(awaiting="confirm", payload={"card": "qualification"})
        return {"messages": [f"r={reply}"]}

    @wf.step
    async def answer(state, ctx):
        return {"messages": ["done"]}

    wf.entry("qualify")
    wf.edge("qualify", "answer")
    wf.finish("answer")

    recorder = Recorder(SQLiteStore())
    events = []
    async for ev in wf.stream({"messages": []}, recorder, thread_id="s2", llm=llm):
        events.append(ev)

    terminal = events[-1]
    assert terminal.kind == "paused"
    assert terminal.turn.is_paused
    assert terminal.turn.awaiting == "confirm"
    assert terminal.turn.interrupt == {"card": "qualification"}


@pytest.mark.asyncio
async def test_stream_resume_continues_same_engagement():
    llm = _StreamingLLM(chunks=["ok"])

    wf = Workflow("stream3", state_schema=_State)

    @wf.step
    async def qualify(state, ctx):
        reply = await ctx.pause(awaiting="confirm", payload={"card": 1})
        return {"messages": [f"r={reply}"]}

    @wf.step
    async def answer(state, ctx):
        await ctx.llm("answer", stream=True)
        return {"messages": ["done"]}

    wf.entry("qualify")
    wf.edge("qualify", "answer")
    wf.finish("answer")

    recorder = Recorder(SQLiteStore())
    start_events = [
        ev async for ev in wf.stream({"messages": []}, recorder, thread_id="s3", llm=llm)
    ]
    started_turn = start_events[-1].turn
    assert started_turn.is_paused

    resume_events = [
        ev
        async for ev in wf.stream_resume(
            thread_id="s3", recorder=recorder, input={"confirm": True}, llm=llm
        )
    ]
    terminal = resume_events[-1]
    assert terminal.kind == "completed"
    assert terminal.turn.engagement_id == started_turn.engagement_id  # SAME engagement
    # The answer node's streamed tokens surfaced on the resume stream.
    assert any(e.kind == "token" for e in resume_events)


@pytest.mark.asyncio
async def test_stream_event_carries_emit_when_node_emits_midturn():
    """A node can ctx.emit(...) a render payload mid-turn (without pausing); it surfaces as an
    'emit' stream event."""
    llm = _StreamingLLM(chunks=["a"])

    wf = Workflow("stream4", state_schema=_State)

    @wf.step
    async def answer(state, ctx):
        ctx.emit({"type": "message_form", "form": {"kind": "info"}})
        return {"messages": ["done"]}

    wf.entry("answer")
    wf.finish("answer")

    recorder = Recorder(SQLiteStore())
    events = [ev async for ev in wf.stream({"messages": []}, recorder, thread_id="s4", llm=llm)]
    emits = [e for e in events if e.kind == "emit"]
    assert emits and emits[0].data == {"type": "message_form", "form": {"kind": "info"}}
    assert events[-1].kind == "completed"
