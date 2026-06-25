"""Streaming a turn incrementally + per-turn tokens + chained pauses (NEEDS.md #2, #8, #13).

``wf.stream(...)`` / ``wf.stream_resume(...)`` are async generators yielding :class:`StreamEvent`s
as the turn executes — ``step_started`` / ``token`` / ``emit`` / ``step_finished`` — and ending on
exactly one terminal ``paused`` / ``completed`` event whose ``ev.turn`` is the resulting
:class:`WorkflowTurn` (carrying ``token_usage`` for THIS turn). This maps 1:1 onto a FastAPI
``StreamingResponse`` NDJSON contract.

This example also shows **chained pauses**: the flow pauses twice (two cards), driven by a
``while turn.is_paused: stream_resume(...)`` loop.

Run it::

    python -m ft.examples.streaming_turn

No network is used — a tiny streaming fake stands in for a real provider.
"""

from __future__ import annotations

import asyncio
from operator import add
from typing import Annotated

from ..core.model import TokenUsage
from ..langgraph_adapter import TraceState
from ..llm import LLMResult
from ..orchestration import Workflow
from ..recorder import Recorder
from ..store.sqlite import SQLiteStore


class State(TraceState):
    messages: Annotated[list, add]


class _StreamingLLM:
    """A fake streaming client: ``astream`` yields text chunks then a final usage-carrying result.

    A real client's ``astream`` awaits its provider's streaming API per chunk.
    """

    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks

    async def acomplete(self, messages, **overrides) -> LLMResult:
        return LLMResult("".join(self._chunks), TokenUsage(prompt=18, completion=12), "fake")

    async def astream(self, messages, **overrides):
        for c in self._chunks:
            await asyncio.sleep(0)  # cooperative — real clients await the network here
            yield c
        yield LLMResult("".join(self._chunks), TokenUsage(prompt=18, completion=12), "fake")


def build_workflow() -> Workflow:
    wf = Workflow("streaming_chat", state_schema=State)

    @wf.step
    async def qualify(state, ctx):
        # Card #1 — pause for the user's confirmation.
        reply = await ctx.pause(awaiting="qualification_confirm", payload={"card": "qualification"})
        return {"messages": [f"qualified={reply}"]}

    @wf.step
    async def pick_area(state, ctx):
        # Card #2 — a chained pause at a DIFFERENT node, on the next turn.
        area = await ctx.pause(awaiting="area_confirm", payload={"card": "pick_area"})
        return {"messages": [f"area={area}"]}

    @wf.step
    async def answer(state, ctx):
        # Streamed answer — each chunk surfaces as a `token` StreamEvent.
        text = await ctx.llm("Summarise the plan for the applicant.", stream=True)
        return {"messages": [text]}

    wf.entry("qualify")
    wf.edge("qualify", "pick_area")
    wf.edge("pick_area", "answer")
    wf.finish("answer")
    return wf


async def drain(agen) -> list:
    return [ev async for ev in agen]


async def main() -> None:
    wf = build_workflow()
    recorder = Recorder(SQLiteStore())
    llm = _StreamingLLM(chunks=["Here ", "is ", "your ", "plan."])
    session = "chat-stream-1"

    # Turn 1 — stream until the first card pauses.
    events = await drain(wf.stream({"messages": []}, recorder, thread_id=session, llm=llm))
    turn = events[-1].turn
    print("turn 1 ->", events[-1].kind, "| awaiting:", turn.awaiting, "| card:", turn.interrupt)

    # Drive the N-card flow as a loop; each stream_resume may itself re-pause (chained pauses).
    replies = iter([{"confirm": True}, {"area": "Shibuya"}])
    while turn.is_paused:
        events = await drain(
            wf.stream_resume(thread_id=session, recorder=recorder, input=next(replies), llm=llm)
        )
        for ev in events:
            if ev.kind == "token":
                print("   token:", repr(ev.data["text"]))
        turn = events[-1].turn
        print("resume ->", events[-1].kind, "| awaiting:", turn.awaiting)

    # The terminal turn carries THIS turn's token usage (for charging a per-turn budget).
    print("final:", turn.status, "| this-turn tokens:", turn.token_usage.total)
    eng = recorder._store.get_engagement(turn.engagement_id)
    print("engagement total tokens (all turns):", eng.total_tokens)


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())
