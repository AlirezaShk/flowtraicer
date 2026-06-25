"""Cross-turn checkpoint/resume.

A node can ``ctx.pause(...)`` to emit a payload and suspend; ``Workflow.start`` returns a paused
turn; ``Workflow.resume`` continues the SAME engagement on a later (possibly fresh-process) call,
keyed by a stable ``thread_id``.
"""

from operator import add
from typing import Annotated

import pytest

from ft.checkpoint import build_checkpointer
from ft.core.model import EngagementStatus, StepStatus
from ft.langgraph_adapter import TraceState
from ft.orchestration import Workflow
from ft.recorder import Recorder
from ft.store.sqlite import SQLiteStore


class _State(TraceState):
    messages: Annotated[list, add]


def _two_node_workflow(checkpointer=None) -> Workflow:
    wf = Workflow("hitl", state_schema=_State, checkpointer=checkpointer)

    @wf.step
    async def a(state, ctx):
        reply = await ctx.pause(awaiting="confirm", payload={"q": "ok?"})
        return {
            "messages": [f"a:{reply}"],
            "events": [{"kind": "log", "name": "reply", "payload": {"reply": reply}}],
        }

    @wf.step
    async def b(state, ctx):
        return {"messages": ["b:done"]}

    wf.entry("a")
    wf.edge("a", "b")
    wf.finish("b")
    return wf


@pytest.mark.asyncio
async def test_start_pauses_and_returns_turn_with_payload():
    wf = _two_node_workflow()
    store = SQLiteStore()
    turn = await wf.start({"messages": []}, Recorder(store), thread_id="t1")

    assert turn.status == "paused"
    assert turn.is_paused is True
    assert turn.awaiting == "confirm"
    assert turn.interrupt == {"q": "ok?"}
    assert turn.engagement_id
    assert turn.thread_id == "t1"

    # The engagement exists and is not ended (it's parked waiting for input).
    eng = store.get_engagement(turn.engagement_id)
    assert eng.status is EngagementStatus.PAUSED
    a_step = next(s for s in eng.steps if s.name == "a")
    assert a_step.status is StepStatus.WAITING


@pytest.mark.asyncio
async def test_resume_completes_and_keeps_one_engagement():
    wf = _two_node_workflow()
    store = SQLiteStore()
    recorder = Recorder(store)

    started = await wf.start({"messages": []}, recorder, thread_id="t2")
    resumed = await wf.resume(thread_id="t2", recorder=recorder, input={"confirm": True})

    assert resumed.status == "completed"
    assert resumed.engagement_id == started.engagement_id  # SAME engagement

    # Exactly one engagement was created for this thread.
    matching = store.list_engagements(where={"ft_thread_id": "t2"})
    assert len(matching) == 1

    eng = store.get_engagement(resumed.engagement_id)
    assert eng.status is EngagementStatus.COMPLETED
    names = [s.name for s in eng.steps]
    assert names == ["a", "a", "b"]  # a (waited), a (resumed), b (completed)
    a_steps = [s for s in eng.steps if s.name == "a"]
    assert a_steps[0].status is StepStatus.WAITING
    assert a_steps[-1].status is StepStatus.COMPLETED
    b_step = next(s for s in eng.steps if s.name == "b")
    assert b_step.status is StepStatus.COMPLETED


@pytest.mark.asyncio
async def test_resume_delivers_input_to_paused_node():
    wf = _two_node_workflow()
    store = SQLiteStore()
    recorder = Recorder(store)

    await wf.start({"messages": []}, recorder, thread_id="t3")
    resumed = await wf.resume(thread_id="t3", recorder=recorder, input={"confirm": "YES"})

    eng = store.get_engagement(resumed.engagement_id)
    final_a = [s for s in eng.steps if s.name == "a"][-1]
    assert final_a.status is StepStatus.COMPLETED
    # ctx.pause returned exactly the value passed to resume(input=...).
    reply_events = [e for s in eng.steps for e in s.events if e.name == "reply"]
    assert reply_events, "node should have recorded the reply it received on resume"
    assert reply_events[-1].payload == {"reply": {"confirm": "YES"}}


@pytest.mark.asyncio
async def test_fresh_process_resume_round_trip():
    """Simulate a separate HTTP request / worker: a brand-new Workflow instance, same checkpointer
    + store + thread_id, resumes the engagement the first instance paused."""
    ckpt = build_checkpointer("memory")
    store = SQLiteStore()
    recorder = Recorder(store)

    wf_req1 = _two_node_workflow(checkpointer=ckpt)
    started = await wf_req1.start({"messages": []}, recorder, thread_id="t4")
    assert started.status == "paused"

    # A fresh Workflow object (new process), sharing only the durable checkpointer + store.
    wf_req2 = _two_node_workflow(checkpointer=ckpt)
    resumed = await wf_req2.resume(thread_id="t4", recorder=recorder, input={"confirm": True})

    assert resumed.status == "completed"
    assert resumed.engagement_id == started.engagement_id
    eng = store.get_engagement(resumed.engagement_id)
    assert eng.status is EngagementStatus.COMPLETED
    assert eng.steps[-1].name == "b"


@pytest.mark.asyncio
async def test_run_still_runs_to_completion():
    """Backward-compat: run() still returns a str engagement id and completes."""
    wf = Workflow("plain", state_schema=_State)

    @wf.step
    async def only(state, ctx):
        return {"messages": ["hi"]}

    wf.entry("only")
    wf.finish("only")

    store = SQLiteStore()
    eid = await wf.run({"messages": []}, Recorder(store))
    assert isinstance(eid, str)
    assert store.get_engagement(eid).status is EngagementStatus.COMPLETED
