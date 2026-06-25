"""App-agnostic tool contract for the multi-tool agentic step (NEEDS.md #5).

A :class:`AgentTool` binds a tool's *implementation* (an async/sync callable) to its name,
description, and JSON-schema parameters — the shape a tool-calling model needs to choose and call
it. :meth:`ft.orchestration.StepContext.run_tools` runs the propose -> execute -> feed-back loop:
the model proposes tool calls, FT executes each handler and records it as a ``tool_call`` (and each
model round as an ``llm_call``) under the running step, feeds results back, and repeats until the
model returns a final text answer or ``max_iterations`` is hit.

FlowTraicer imports **no application types** here — a tool handler receives the validated args dict
and the run's ``ctx`` (so it can reach request-scoped services via ``ctx.deps``) and returns any
JSON-serializable result the model can read.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

#: A tool handler: ``handler(args: dict, ctx: StepContext) -> result`` (sync or async). The result
#: must be JSON-serializable (it's fed back to the model). ``ctx`` is the running StepContext, so a
#: handler can reach request-scoped services via ``ctx.deps``.
ToolHandler = Callable[[dict, Any], "Any | Awaitable[Any]"]


@dataclass
class AgentTool:
    """A single executable tool the model may choose during an agentic step.

    :param name: the snake_case identifier the model calls.
    :param description: when-to-use guidance shown to the model.
    :param parameters: JSON schema for the tool's arguments (what the model fills in).
    :param handler: ``handler(args, ctx)`` (sync or async) returning a JSON-serializable result.
    """

    name: str
    description: str
    parameters: dict = field(default_factory=dict)
    handler: ToolHandler | None = None

    def spec(self) -> dict:
        """The provider-agnostic tool spec handed to the LLM client (``acomplete(tools=...)``)."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    async def invoke(self, args: dict, ctx: Any) -> Any:
        """Run the handler (awaiting it if async)."""
        if self.handler is None:
            raise ValueError(f"AgentTool {self.name!r} has no handler")
        result = self.handler(args, ctx)
        if inspect.isawaitable(result):
            result = await result
        return result
