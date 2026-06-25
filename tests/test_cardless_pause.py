"""Cardless 'await-user' pause + re-entrant loop (NEEDS.md #A / #14).

Shape A of the SP4 cutover maps the WHOLE chat session to ONE FT engagement: after an agentic
``assist`` step answers, the flow pauses purely to hand the turn back to the user (no card) and
resumes into ``assist`` again — chained pauses at the same node, the engagement never finishing.

These tests lock the two guarantees the app relies on:

1. ``ctx.pause(awaiting=…, payload=None)`` (or payload omitted) yields ``turn.interrupt is None``
   and the streaming terminal ``paused`` event carries no card — so the app emits no
   ``message_form`` line, just the streamed answer + usage.
2. An engagement can stay PAUSED and be resumed MANY times at the SAME node (re-entrant loop),
   never reaching a goal/finish, with per-turn ``token_usage`` scoped to only that turn's steps.
"""

from operator import add
from typing import Annotated

import pytest

from ft.core.model import EngagementStatus, TokenUsage
from ft.langgraph_adapter import TraceState
from ft.llm import LLMResult
from ft.orchestration import Workflow
from ft.recorder import Recorder
from ft.store.sqlite import SQLiteStore


class _State(TraceState):
    messages: Annotated[list, add]
    qualified: bool


class _StreamingLLM:
    """A fake streaming client: ``astream`` yields chunks then a final usage-carrying result."""

    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks

    async def acomplete(self, messages, **overrides) -> LLMResult:
        return LLMResult("".join(self._chunks), TokenUsage(prompt=5, completion=3), "fake")

    async def astream(self, messages, **overrides):
        for c in self._chunks:
            yield c
        yield LLMResult("".join(self._chunks), TokenUsage(prompt=5, completion=3), "fake")


def _await_user_workflow() -> Workflow:
    """qualify ─► assist ─► await_user (cardless pause) ─► assist ─► …  (never finishes)."""
    wf = Workflow("await_user_loop", state_schema=_State)

    @wf.step
    async def qualify(state, ctx):
        reply = await ctx.pause(awaiting="qualification_confirm", payload={"card": "qualification"})
        confirmed = bool((reply or {}).get("confirm"))
        return {"qualified": confirmed, "messages": [f"qualified={confirmed}"]}

    @wf.step
    async def assist(state, ctx):
        # The streamed answer is AFTER the last pause (the await_user pause precedes re-entry),
        # so it streams/charges exactly once per turn (replay rule).
        text = await ctx.llm("answer the user", stream=True)
        return {"messages": [text]}

    @wf.step
    async def await_user(state, ctx):
        # Cardless pause: purely hands the turn back to the user; no card to render.
        await ctx.pause(awaiting="user_turn")
        return {}

    def after_qualify(state) -> str:
        return "assist" if state.get("qualified") else "decline"

    @wf.step
    async def decline(state, ctx):
        return {"messages": ["no problem"]}

    wf.entry("qualify")
    wf.branch("qualify", after_qualify, {"assist": "assist", "decline": "decline"})
    wf.edge("assist", "await_user")
    wf.edge("await_user", "assist")  # re-entrant loop: resume goes back into assist
    wf.finish("decline")
    return wf


@pytest.mark.asyncio
async def test_cardless_pause_yields_interrupt_none():
    """ctx.pause(payload=None) → turn.interrupt is None, status paused, awaiting carried."""
    wf = Workflow("cardless", state_schema=_State)

    @wf.step
    async def await_user(state, ctx):
        await ctx.pause(awaiting="user_turn")  # payload omitted == None
        return {}

    wf.entry("await_user")
    wf.finish("await_user")

    recorder = Recorder(SQLiteStore())
    turn = await wf.start({"messages": [], "qualified": True}, recorder, thread_id="c1")

    assert turn.is_paused
    assert turn.awaiting == "user_turn"
    assert turn.interrupt is None


@pytest.mark.asyncio
async def test_cardless_pause_streaming_terminal_carries_no_card():
    """The streaming terminal `paused` event carries no card (turn.interrupt is None)."""
    llm = _StreamingLLM(chunks=["hi"])
    wf = Workflow("cardless_stream", state_schema=_State)

    @wf.step
    async def assist(state, ctx):
        text = await ctx.llm("answer", stream=True)
        return {"messages": [text]}

    @wf.step
    async def await_user(state, ctx):
        await ctx.pause(awaiting="user_turn")
        return {}

    wf.entry("assist")
    wf.edge("assist", "await_user")
    wf.finish("await_user")  # finish satisfies the graph; pause is what actually parks it

    recorder = Recorder(SQLiteStore())
    events = [
        ev
        async for ev in wf.stream(
            {"messages": [], "qualified": True}, recorder, thread_id="c2", llm=llm
        )
    ]
    terminal = events[-1]
    assert terminal.kind == "paused"
    assert terminal.turn.is_paused
    assert terminal.turn.awaiting == "user_turn"
    assert terminal.turn.interrupt is None  # no card → app emits no message_form line
    # The assist answer DID stream before the pause this turn.
    assert "".join(e.data["text"] for e in events if e.kind == "token") == "hi"


@pytest.mark.asyncio
async def test_reentrant_loop_pauses_at_same_node_many_times_one_engagement():
    """An engagement stays PAUSED and resumes MANY times into the same agentic step (shape A)."""
    llm = _StreamingLLM(chunks=["ans"])
    wf = _await_user_workflow()
    recorder = Recorder(SQLiteStore())
    session = "loop-1"

    # Turn 1: qualify pauses for the confirmation card.
    turn = await wf.start(
        {"messages": [], "qualified": False}, recorder, thread_id=session, llm=llm
    )
    assert turn.is_paused and turn.awaiting == "qualification_confirm"
    assert turn.interrupt == {"card": "qualification"}
    engagement_id = turn.engagement_id

    # Turn 2: confirm → assist answers → await_user pauses (cardless).
    turn = await wf.resume(thread_id=session, recorder=recorder, input={"confirm": True}, llm=llm)
    assert turn.is_paused
    assert turn.awaiting == "user_turn"
    assert turn.interrupt is None
    assert turn.engagement_id == engagement_id  # SAME engagement

    # Turns 3..N: resume the cardless pause repeatedly — re-enters assist, re-pauses at await_user.
    for _ in range(4):
        turn = await wf.resume(thread_id=session, recorder=recorder, input=None, llm=llm)
        assert turn.is_paused, "the loop never finishes — engagement stays PAUSED"
        assert turn.awaiting == "user_turn"
        assert turn.interrupt is None
        assert turn.engagement_id == engagement_id  # still ONE engagement across all turns

    # The engagement is PAUSED (never ended) — it would be never purged by retention.
    summary = recorder._store.get_engagement(engagement_id)
    assert summary.status == EngagementStatus.PAUSED


@pytest.mark.asyncio
async def test_reentrant_loop_per_turn_token_accounting_is_scoped_to_that_turn():
    """Each resume's turn.token_usage counts only that turn's steps (assist's one llm round)."""
    llm = _StreamingLLM(chunks=["x"])  # each astream → prompt=5, completion=3 → total 8
    wf = _await_user_workflow()
    recorder = Recorder(SQLiteStore())
    session = "loop-tokens"

    # Turn 1 (qualify) — no llm call, so this turn's tokens are 0.
    turn = await wf.start(
        {"messages": [], "qualified": False}, recorder, thread_id=session, llm=llm
    )
    assert turn.token_usage.total == 0

    # Turn 2 (confirm → assist runs one llm round) — exactly that round's tokens.
    turn = await wf.resume(thread_id=session, recorder=recorder, input={"confirm": True}, llm=llm)
    assert turn.token_usage.total == 8  # only THIS turn's assist round (5 + 3)

    # Turn 3 (resume the cardless pause → assist runs again) — again only this turn's tokens.
    turn = await wf.resume(thread_id=session, recorder=recorder, input=None, llm=llm)
    assert turn.token_usage.total == 8

    # The engagement total spans all turns (a different granularity than per-turn).
    eng = recorder._store.get_engagement(turn.engagement_id)
    assert eng.total_tokens == 16  # two assist rounds so far (turn 2 + turn 3)
