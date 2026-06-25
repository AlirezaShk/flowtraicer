"""Pin the resume/turn contract documented in NEEDS.md #11, #12, #13.

These tests turn three previously-ambiguous behaviours into guarantees:

* #11 — ``turn.interrupt`` is the **unwrapped** ``payload`` dict passed to ``ctx.pause(payload=…)``.
* #12 — resuming a **completed** thread raises ``ResumeError``; ``expect_awaiting=`` mismatch raises
  ``ResumeError``; ``resume(input=None)`` surfaces as ``None`` inside ``ctx.pause(...)``.
* #13 — a single ``resume`` may itself return ``status="paused"`` again at a different node
  (chained pauses), so an N-card flow is a ``while turn.is_paused: resume(...)`` loop.
"""

from operator import add
from typing import Annotated

import pytest

from ft.core.model import EngagementStatus
from ft.langgraph_adapter import TraceState
from ft.orchestration import ResumeError, Workflow
from ft.recorder import Recorder
from ft.store.sqlite import SQLiteStore


class _State(TraceState):
    messages: Annotated[list, add]


# ── #11: turn.interrupt is the unwrapped payload ──────────────────────────────────────


@pytest.mark.asyncio
async def test_interrupt_is_unwrapped_payload():
    """``turn.interrupt`` must be exactly the ``payload`` dict (not the wrapper)."""
    card = {"type": "qualification", "fields": [{"name": "confirm", "kind": "boolean"}]}

    wf = Workflow("c11", state_schema=_State)

    @wf.step
    async def qualify(state, ctx):
        await ctx.pause(awaiting="qualification_confirm", payload=card)
        return {"messages": ["done"]}

    wf.entry("qualify")
    wf.finish("qualify")

    turn = await wf.start({"messages": []}, Recorder(SQLiteStore()), thread_id="t11")
    assert turn.awaiting == "qualification_confirm"
    # The exact card, verbatim and unwrapped — no "awaiting"/"payload"/"value" wrapper key.
    assert turn.interrupt == card
    assert "awaiting" not in turn.interrupt
    assert "payload" not in turn.interrupt


# ── #12: resume on completed / stale / wrong-awaiting / None-input ─────────────────────


def _one_pause_workflow() -> Workflow:
    wf = Workflow("c12", state_schema=_State)

    @wf.step
    async def qualify(state, ctx):
        reply = await ctx.pause(awaiting="confirm", payload={"q": "ok?"})
        return {
            "messages": [f"reply={reply!r}"],
            "events": [
                {"kind": "log", "name": "got_reply", "payload": {"reply_is_none": reply is None}}
            ],
        }

    wf.entry("qualify")
    wf.finish("qualify")
    return wf


@pytest.mark.asyncio
async def test_resume_on_completed_thread_raises_resume_error():
    wf = _one_pause_workflow()
    recorder = Recorder(SQLiteStore())
    await wf.start({"messages": []}, recorder, thread_id="t12a")
    done = await wf.resume(thread_id="t12a", recorder=recorder, input={"confirm": True})
    assert done.status == "completed"

    # A double-submit / stale replay on the now-completed thread must raise (not silently re-run).
    with pytest.raises(ResumeError):
        await wf.resume(thread_id="t12a", recorder=recorder, input={"confirm": True})


@pytest.mark.asyncio
async def test_resume_unknown_thread_still_raises_resume_error():
    wf = _one_pause_workflow()
    recorder = Recorder(SQLiteStore())
    with pytest.raises(ResumeError):
        await wf.resume(thread_id="never-started", recorder=recorder, input={"x": 1})


@pytest.mark.asyncio
async def test_resume_expect_awaiting_mismatch_raises():
    wf = _one_pause_workflow()
    recorder = Recorder(SQLiteStore())
    await wf.start({"messages": []}, recorder, thread_id="t12b")
    # The thread is parked on "confirm"; asserting a different label must raise before delivering.
    with pytest.raises(ResumeError):
        await wf.resume(
            thread_id="t12b", recorder=recorder, input={"x": 1}, expect_awaiting="something_else"
        )
    # The correct label proceeds.
    done = await wf.resume(
        thread_id="t12b", recorder=recorder, input={"confirm": True}, expect_awaiting="confirm"
    )
    assert done.status == "completed"


@pytest.mark.asyncio
async def test_resume_with_none_input_surfaces_as_none():
    wf = _one_pause_workflow()
    recorder = Recorder(SQLiteStore())
    started = await wf.start({"messages": []}, recorder, thread_id="t12c")
    done = await wf.resume(thread_id="t12c", recorder=recorder)  # input defaults to None
    assert done.status == "completed"
    eng = recorder._store.get_engagement(started.engagement_id)
    got = [e for s in eng.steps for e in s.events if e.name == "got_reply"][-1]
    assert got.payload == {"reply_is_none": True}  # ctx.pause(...) returned None, not {}


# ── #13: chained pauses — a resume can re-pause at a different node ─────────────────────


def _two_card_workflow() -> Workflow:
    wf = Workflow("c13", state_schema=_State)

    @wf.step
    async def card_one(state, ctx):
        r = await ctx.pause(awaiting="card_one", payload={"card": 1})
        return {"messages": [f"one:{r}"]}

    @wf.step
    async def card_two(state, ctx):
        r = await ctx.pause(awaiting="card_two", payload={"card": 2})
        return {"messages": [f"two:{r}"]}

    @wf.step
    async def done(state, ctx):
        return {"messages": ["done"]}

    wf.entry("card_one")
    wf.edge("card_one", "card_two")
    wf.edge("card_two", "done")
    wf.finish("done")
    return wf


@pytest.mark.asyncio
async def test_resume_can_re_pause_at_a_different_node():
    wf = _two_card_workflow()
    recorder = Recorder(SQLiteStore())

    turn = await wf.start({"messages": []}, recorder, thread_id="t13")
    assert turn.is_paused and turn.awaiting == "card_one" and turn.interrupt == {"card": 1}

    # Drive the N-card flow as a loop; the second resume RE-PAUSES at card_two.
    seen = [turn.awaiting]
    guard = 0
    while turn.is_paused:
        guard += 1
        assert guard < 5
        turn = await wf.resume(thread_id="t13", recorder=recorder, input={"ack": True})
        if turn.is_paused:
            seen.append(turn.awaiting)

    assert seen == ["card_one", "card_two"]
    assert turn.status == "completed"
    eng = recorder._store.get_engagement(turn.engagement_id)
    assert eng.status is EngagementStatus.COMPLETED
