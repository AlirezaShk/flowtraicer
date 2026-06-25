"""Whole chat session = ONE FT engagement, via a cardless 'await-user' pause (NEEDS.md #14 / #A).

This is **shape A** of the SP4 school-chat cutover: instead of completing the engagement each turn,
the flow pauses after every agentic answer purely to **hand the turn back to the user** (no card),
then resumes back into the agentic step on the next user message — looping forever. The result is
one continuous ``PAUSED`` engagement spanning the entire conversation::

    qualify ─(yes)─► assist ─► await_user (pause, no card) ─► assist ─► await_user ─► …

Key guarantees this example demonstrates (locked by ``tests/test_cardless_pause.py``):

- ``ctx.pause(awaiting="user_turn")`` (no ``payload``) yields ``turn.interrupt is None`` — so the
  app emits no ``message_form`` line, only the streamed answer + a ``usage`` line.
- The engagement never finishes; it stays ``PAUSED`` and is resumed many times at the SAME node.
- Each turn's ``token_usage`` is scoped to only that turn's steps (charge a per-turn budget off it).
- The streamed answer lives in ``assist`` (AFTER the last pause), so tokens stream/charge once
  (the replay rule).

Run it::

    python -m ft.examples.session_loop

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
    qualified: bool


class _StreamingLLM:
    """A fake streaming client: ``astream`` yields chunks then a final usage-carrying result."""

    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks

    async def acomplete(self, messages, **overrides) -> LLMResult:
        return LLMResult("".join(self._chunks), TokenUsage(prompt=12, completion=8), "fake")

    async def astream(self, messages, **overrides):
        for c in self._chunks:
            await asyncio.sleep(0)  # cooperative — a real client awaits the network here
            yield c
        yield LLMResult("".join(self._chunks), TokenUsage(prompt=12, completion=8), "fake")


def build_workflow() -> Workflow:
    wf = Workflow("school_session", state_schema=State)

    @wf.step
    async def qualify(state, ctx):
        # The bug-fix gate: emit the financial-qualification card and pause for the confirm.
        card = {"type": "qualification", "first_year_cost_yen": 2_400_000}
        reply = await ctx.pause(awaiting="general_qualification_confirm", payload=card)
        confirmed = bool((reply or {}).get("confirm"))
        return {"qualified": confirmed, "messages": [f"qualified={confirmed}"]}

    @wf.step
    async def assist(state, ctx):
        # The open-ended Q&A answer. In SP4 this is an @wf.agent_step (ctx.run_tools); here a single
        # streamed ctx.llm stands in. It is AFTER the last pause, so it streams exactly once.
        text = await ctx.llm("Answer the user's question.", stream=True)
        return {"messages": [text]}

    @wf.step
    async def await_user(state, ctx):
        # CARDLESS pause: purely yields the turn back to the endpoint to await the next message.
        # turn.interrupt is None -> the app emits no message_form, only the assist answer + usage.
        await ctx.pause(awaiting="user_turn")
        return {}

    @wf.step
    async def decline(state, ctx):
        return {"messages": ["No problem — happy to answer any questions."]}

    def after_qualify(state) -> str:
        return "assist" if state.get("qualified") else "decline"

    wf.entry("qualify")
    wf.branch("qualify", after_qualify, {"assist": "assist", "decline": "decline"})
    wf.edge("assist", "await_user")
    wf.edge("await_user", "assist")  # re-entrant loop: each user message resumes back into assist
    wf.finish("decline")
    return wf


async def drain(agen) -> list:
    return [ev async for ev in agen]


async def main() -> None:
    wf = build_workflow()
    recorder = Recorder(SQLiteStore())
    llm = _StreamingLLM(chunks=["Here ", "is ", "the ", "answer."])
    session = "chat-session-A"  # your app's ChatSession id == the FT thread_id

    # ── Turn 1: first school intent → the qualification card pauses for the confirm. ──
    events = await drain(
        wf.stream({"messages": [], "qualified": False}, recorder, thread_id=session, llm=llm)
    )
    turn = events[-1].turn
    print("turn 1 ->", turn.status, "| awaiting:", turn.awaiting, "| card:", turn.interrupt)

    # ── Turn 2: the user confirms → assist answers (streamed) → await_user pauses (NO card). ──
    events = await drain(
        wf.stream_resume(thread_id=session, recorder=recorder, input={"confirm": True}, llm=llm)
    )
    turn = events[-1].turn
    answer = "".join(e.data["text"] for e in events if e.kind == "token")
    print(f"turn 2 -> {turn.status} | awaiting: {turn.awaiting} | card: {turn.interrupt!r}")
    print(f"         streamed answer: {answer!r} | this-turn tokens: {turn.token_usage.total}")

    # ── Turns 3..5: each subsequent user message resumes the cardless pause, re-entering assist. ──
    for n in range(3, 6):
        events = await drain(
            wf.stream_resume(thread_id=session, recorder=recorder, input=None, llm=llm)
        )
        turn = events[-1].turn
        print(
            f"turn {n} -> {turn.status} | awaiting: {turn.awaiting} "
            f"| card: {turn.interrupt!r} | this-turn tokens: {turn.token_usage.total}"
        )

    # The engagement never finished — it is one continuous PAUSED journey across all turns.
    eng = recorder._store.get_engagement(turn.engagement_id)
    print("\nengagement status:", eng.status.value, "(stays PAUSED; never purged by retention)")
    print("engagement total tokens (all turns):", eng.total_tokens)


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())
