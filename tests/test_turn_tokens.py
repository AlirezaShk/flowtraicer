"""Per-turn token accounting.

``WorkflowTurn.token_usage`` sums tokens for ONLY the steps advanced during THIS start/resume turn,
not the whole engagement (which spans the multi-turn session).
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


class _FakeLLM:
    """Returns a fixed token cost per call, so we can assert exact per-turn sums."""

    def __init__(self, prompt: int, completion: int) -> None:
        self._p = prompt
        self._c = completion

    async def acomplete(self, messages, **overrides) -> LLMResult:
        return LLMResult(
            text="ok", tokens=TokenUsage(prompt=self._p, completion=self._c), model="fake"
        )


@pytest.mark.asyncio
async def test_turn_token_usage_is_scoped_to_steps_advanced_this_turn():
    wf = Workflow("tok", state_schema=_State)

    @wf.step
    async def gate(state, ctx):
        await ctx.llm("score this")  # 10 + 5
        reply = await ctx.pause(awaiting="confirm", payload={"q": "ok?"})
        return {"messages": [f"r={reply}"]}

    @wf.step
    async def answer(state, ctx):
        await ctx.llm("answer this")  # 10 + 5
        await ctx.llm("answer again")  # 10 + 5
        return {"messages": ["done"]}

    wf.entry("gate")
    wf.edge("gate", "answer")
    wf.finish("answer")

    recorder = Recorder(SQLiteStore())
    llm = _FakeLLM(prompt=10, completion=5)

    # Turn 1: gate runs up to ctx.pause, which raises the interrupt control-flow signal — so the
    # node's partial state writes (its pre-pause ctx.llm) are DISCARDED by LangGraph and not yet
    # recorded. The pausing turn therefore accounts 0 tokens; the work is replayed (and recorded)
    # on resume. This is the documented replay rule for paused nodes.
    turn1 = await wf.start({"messages": []}, recorder, thread_id="t1", llm=llm)
    assert turn1.is_paused
    assert turn1.token_usage.total == 0  # pre-pause writes are discarded until the node replays

    # Turn 2: resume -> gate REPLAYS to completion (its ctx.llm runs = 15), then answer (2x15=30).
    turn2 = await wf.resume(thread_id="t1", recorder=recorder, input={"confirm": True}, llm=llm)
    assert turn2.status == "completed"
    assert turn2.token_usage.total == 45  # gate replay (15) + answer (30), THIS turn only
    assert turn2.token_usage.prompt == 30
    assert turn2.token_usage.completion == 15

    # The engagement total equals the resume turn's (the paused turn recorded nothing) = 45.
    eng = recorder._store.get_engagement(turn2.engagement_id)
    assert eng.total_tokens == 45


@pytest.mark.asyncio
async def test_turn_token_usage_zero_when_no_llm_calls():
    wf = Workflow("tok2", state_schema=_State)

    @wf.step
    async def only(state, ctx):
        return {"messages": ["hi"]}

    wf.entry("only")
    wf.finish("only")

    turn = await wf.start({"messages": []}, Recorder(SQLiteStore()), thread_id="z1")
    assert turn.status == "completed"
    assert turn.token_usage.total == 0
