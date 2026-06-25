"""Human-in-the-loop pause/resume across turns.

A two-node workflow where the first node emits a card and **pauses** for the user's reply, and a
later call **resumes** the SAME engagement with that reply. This is the shape a multi-turn chat
endpoint uses: one HTTP request = one ``start`` (turn 1) or ``resume`` (turn N), keyed by the chat
session id (``thread_id``).

Run it::

    python -m ft.examples.hitl_resume

No LLM/network is used. See ``docs/2026-06-25-checkpoint-resume-design.md`` for the design.
"""

from __future__ import annotations

import asyncio
from operator import add
from typing import Annotated

from ..langgraph_adapter import TraceState
from ..orchestration import Workflow
from ..recorder import Recorder
from ..store.sqlite import SQLiteStore


class State(TraceState):
    messages: Annotated[list, add]


def build_workflow() -> Workflow:
    # No checkpointer passed -> Workflow lazily uses an in-process MemorySaver (fine for a demo /
    # single worker). For cross-process resume, pass build_checkpointer("sqlite"/"postgres").
    wf = Workflow("qualification_chat", state_schema=State)

    @wf.step
    async def qualify(state, ctx):
        # Build the render-ready card, then pause and wait for the user's confirm.
        card = {"type": "qualification", "first_year_cost_yen": 2_400_000}
        reply = await ctx.pause(awaiting="qualification_confirm", payload=card)
        # On resume, `reply` is whatever resume(input=...) supplied — normalise both channels:
        confirmed = bool(reply.get("confirm")) or reply.get("text", "").lower() in {"yes", "y"}
        return {"messages": [f"qualified={confirmed}"]}

    @wf.step
    async def answer(state, ctx):
        return {"messages": ["Great — what area are you considering?"]}

    wf.entry("qualify")
    wf.edge("qualify", "answer")
    wf.finish("answer")
    return wf


async def main() -> None:
    wf = build_workflow()
    recorder = Recorder(SQLiteStore())
    session_id = "chat-session-123"  # your app's ChatSession id

    # --- turn 1: the user shows intent; run until the card pauses for input ---
    turn = await wf.start({"messages": []}, recorder, thread_id=session_id)
    print("turn 1:", turn.status, "| awaiting:", turn.awaiting, "| card:", turn.interrupt)
    #   -> render turn.interrupt to the user and stop the HTTP response here.

    # --- turn N (a later HTTP request): the user clicked "confirm" (or typed "yes") ---
    turn = await wf.resume(thread_id=session_id, recorder=recorder, input={"confirm": True})
    print("turn 2:", turn.status, "| engagement:", turn.engagement_id)

    # One continuous engagement across both turns:
    eng = recorder._store.get_engagement(turn.engagement_id)
    print("engagement status:", eng.status.value)
    print("steps:", [(s.name, s.status.value) for s in eng.steps])


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())
