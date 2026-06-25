"""Multi-tool agentic step.

A single FlowTraicer step where the model chooses among MANY tools in a ReAct loop
(propose -> execute -> feed back), instead of one fixed ``ctx.llm`` call. FT runs the loop via
``ctx.run_tools(...)`` and records each tool call + LLM round under the step.

The tool contract (:class:`ft.agent.AgentTool`) is app-agnostic: ``{name, description, parameters,
handler}`` where ``handler(args, ctx)`` returns a JSON-serializable result the model reads.

Run it::

    python -m ft.examples.agentic_step

No network is used — a tiny scripted tool-calling LLM stands in for a real provider.
"""

from __future__ import annotations

import asyncio
from operator import add
from typing import Annotated

from ..agent import AgentTool
from ..core.model import TokenUsage
from ..langgraph_adapter import TraceState
from ..llm import LLMResult, ToolRequest
from ..orchestration import Workflow
from ..recorder import Recorder
from ..store.sqlite import SQLiteStore


class State(TraceState):
    messages: Annotated[list, add]


# --- the tools (app-agnostic: each is name + description + json schema + async handler) ----------


async def search_schools(args: dict, ctx) -> dict:
    # A real handler would reach a service via ctx.deps; here we just echo.
    return {"results": ["Tokyo Intl. Academy", "Osaka Nihongo School"], "area": args.get("area")}


async def compare_schools(args: dict, ctx) -> dict:
    return {"recommendation": args.get("a"), "reason": "closer to the requested area"}


TOOLS = [
    AgentTool(
        name="search_schools",
        description="Search Japanese language schools by area/budget.",
        parameters={"type": "object", "properties": {"area": {"type": "string"}}},
        handler=search_schools,
    ),
    AgentTool(
        name="compare_schools",
        description="Compare two schools and recommend one.",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
        },
        handler=compare_schools,
    ),
]


class _ScriptedToolLLM:
    """Stands in for a real tool-calling client: a scripted sequence of rounds.

    A real client implements ``acomplete(messages, *, tools=...)`` and parses its provider's
    function-call response into ``LLMResult.tool_calls`` (a list of ``ToolRequest``).
    """

    def __init__(self, script: list) -> None:
        self._script = list(script)

    async def acomplete(self, messages, *, tools=None, **overrides) -> LLMResult:
        step = self._script.pop(0)
        if isinstance(step, list):
            return LLMResult("", TokenUsage(prompt=20, completion=8), "scripted", tool_calls=step)
        return LLMResult(step, TokenUsage(prompt=12, completion=30), "scripted")


def build_workflow() -> Workflow:
    wf = Workflow("school_qa", state_schema=State)

    @wf.agent_step(tools=TOOLS, max_iterations=8)
    async def qa(state, ctx):
        answer = await ctx.run_tools(state["messages"])
        return {"messages": [answer]}

    wf.entry("qa")
    wf.finish("qa")
    return wf


async def main() -> None:
    wf = build_workflow()
    recorder = Recorder(SQLiteStore())
    # The model: round 1 search, round 2 compare, round 3 final answer.
    llm = _ScriptedToolLLM(
        script=[
            [ToolRequest(name="search_schools", args={"area": "Shinjuku"})],
            [ToolRequest(name="compare_schools", args={"a": "Tokyo Intl.", "b": "Osaka Nihongo"})],
            "I recommend Tokyo Intl. Language Academy — it's closest to Shinjuku.",
        ]
    )

    eid = await wf.run({"messages": ["Find me a language school near Shinjuku"]}, recorder, llm=llm)

    eng = recorder._store.get_engagement(eid)
    qa_step = next(s for s in eng.steps if s.name == "qa")
    print("step:", qa_step.name, qa_step.status.value)
    print("tools called:", [e.name for e in qa_step.events if e.kind.value == "tool_call"])
    print("llm rounds:", len([e for e in qa_step.events if e.kind.value == "llm_call"]))
    print("step tokens:", qa_step.total_tokens)


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())
