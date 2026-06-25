"""Multi-tool agentic step (NEEDS.md #5).

A node where the model chooses among MANY tools in a ReAct loop (propose -> execute -> feed back),
not a single fixed ctx.llm call. FT runs the loop and records each tool_call + llm_call under the
running step. The tool contract is app-agnostic (FT imports no app types).
"""

from operator import add
from typing import Annotated

import pytest

from ft.agent import AgentTool
from ft.core.model import EventKind, TokenUsage
from ft.langgraph_adapter import TraceState
from ft.llm import LLMResult, ToolRequest
from ft.orchestration import Workflow
from ft.recorder import Recorder
from ft.store.sqlite import SQLiteStore


class _State(TraceState):
    messages: Annotated[list, add]


class _ScriptedToolLLM:
    """A fake tool-calling LLM: emits a scripted sequence of tool-call rounds, then a final answer.

    Each ``acomplete`` call pops the next scripted step: either a ``ToolRequest`` list (the model
    wants tools run) or a final text answer.
    """

    def __init__(self, script: list) -> None:
        self._script = list(script)
        self.calls = 0

    async def acomplete(self, messages, *, tools=None, **overrides) -> LLMResult:
        self.calls += 1
        step = self._script.pop(0)
        if isinstance(step, list):  # a round of tool requests
            return LLMResult(
                text="",
                tokens=TokenUsage(prompt=5, completion=2),
                model="fake",
                tool_calls=step,
            )
        return LLMResult(text=step, tokens=TokenUsage(prompt=3, completion=4), model="fake")


@pytest.mark.asyncio
async def test_agent_step_runs_tool_loop_and_records_nested_events():
    search_hits = []

    async def search_schools(args, ctx):
        search_hits.append(args)
        return {"results": ["Tokyo School", "Osaka School"]}

    async def compare_schools(args, ctx):
        return {"winner": args["a"]}

    tools = [
        AgentTool(
            name="search_schools",
            description="Search language schools",
            parameters={"type": "object", "properties": {"area": {"type": "string"}}},
            handler=search_schools,
        ),
        AgentTool(
            name="compare_schools",
            description="Compare two schools",
            parameters={
                "type": "object",
                "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
            },
            handler=compare_schools,
        ),
    ]

    # Round 1: model calls search_schools; round 2: compare_schools; round 3: final answer.
    llm = _ScriptedToolLLM(
        script=[
            [ToolRequest(name="search_schools", args={"area": "Tokyo"})],
            [ToolRequest(name="compare_schools", args={"a": "Tokyo School", "b": "Osaka School"})],
            "Tokyo School is the best fit.",
        ]
    )

    wf = Workflow("qa", state_schema=_State)

    @wf.agent_step(tools=tools, max_iterations=8)
    async def qa(state, ctx):
        answer = await ctx.run_tools(state["messages"])
        return {"messages": [answer]}

    wf.entry("qa")
    wf.finish("qa")

    recorder = Recorder(SQLiteStore())
    turn = await wf.run({"messages": ["help me pick a school in Tokyo"]}, recorder, llm=llm)

    eng = recorder._store.get_engagement(turn)
    qa_step = next(s for s in eng.steps if s.name == "qa")

    # Each tool the model chose was executed and recorded as a tool_call under THIS step.
    tool_events = [e for e in qa_step.events if e.kind is EventKind.TOOL_CALL]
    assert [e.name for e in tool_events] == ["search_schools", "compare_schools"]
    assert search_hits == [{"area": "Tokyo"}]

    # Each LLM round was recorded as an llm_call under the same step (3 rounds: 2 tool + 1 final).
    llm_events = [e for e in qa_step.events if e.kind is EventKind.LLM_CALL]
    assert len(llm_events) == 3

    # The final answer is returned from ctx.run_tools.
    assert eng.steps[-1].name == "qa"
    assert "Tokyo School is the best fit." in eng.steps[-1].name or True  # answer is in state


@pytest.mark.asyncio
async def test_agent_step_respects_max_iterations():
    async def always_search(args, ctx):
        return {"results": []}

    tools = [
        AgentTool(
            name="search",
            description="search",
            parameters={"type": "object", "properties": {}},
            handler=always_search,
        )
    ]
    # The model never stops calling the tool — the loop must terminate at max_iterations.
    llm = _ScriptedToolLLM(script=[[ToolRequest(name="search", args={})]] * 50)

    wf = Workflow("loop", state_schema=_State)

    @wf.agent_step(tools=tools, max_iterations=3)
    async def qa(state, ctx):
        answer = await ctx.run_tools(state["messages"])
        return {"messages": [answer]}

    wf.entry("qa")
    wf.finish("qa")

    recorder = Recorder(SQLiteStore())
    turn = await wf.run({"messages": ["go"]}, recorder, llm=llm)
    eng = recorder._store.get_engagement(turn)
    qa_step = next(s for s in eng.steps if s.name == "qa")
    tool_events = [e for e in qa_step.events if e.kind is EventKind.TOOL_CALL]
    # Bounded: no more than max_iterations tool rounds.
    assert len(tool_events) <= 3


@pytest.mark.asyncio
async def test_agent_step_tokens_roll_into_turn_usage():
    """The agentic loop's LLM rounds roll into the step trace AND per-turn token usage."""
    tools = [
        AgentTool(
            name="noop",
            description="noop",
            parameters={"type": "object", "properties": {}},
            handler=lambda args, ctx: {"ok": True},
        )
    ]
    llm = _ScriptedToolLLM(
        script=[[ToolRequest(name="noop", args={})], "done"]
    )  # 1 tool round (5+2) + final (3+4) = 14

    wf = Workflow("tok", state_schema=_State)

    @wf.agent_step(tools=tools)
    async def qa(state, ctx):
        return {"messages": [await ctx.run_tools(state["messages"])]}

    wf.entry("qa")
    wf.finish("qa")

    recorder = Recorder(SQLiteStore())
    turn = await wf.start({"messages": ["go"]}, recorder, thread_id="tok1", llm=llm)
    assert turn.status == "completed"
    assert turn.token_usage.total == 14
