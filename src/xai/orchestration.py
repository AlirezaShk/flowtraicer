"""A small declarative DSL for building instrumented LangGraph workflows.

Sugar over ``langgraph.StateGraph`` + :func:`xai.langgraph_adapter.run_instrumented`: declare
steps (with their tools), global steps (intent re-routes), goals, and edges once, and the
workflow compiles the graph and wires the per-step tools / global nodes / goal nodes into the
recorder for you — no separate ``node_tools=`` / ``global_nodes=`` / ``goal_nodes=`` bookkeeping.

```python
wf = Workflow("school_journey", state_schema=State, goal_nodes={"submit"})

@wf.step(tools=["search_schools"])
def school_selection(state): ...

@wf.global_step
def escalate(state): ...

wf.entry("intake")
wf.edge("intake", "school_selection")
wf.branch("school_selection", router, {"compare": "comparison", "apply": "consent"})
wf.finish("submit")

engagement_id = await wf.run(initial_state, recorder, metadata={"user_id": "u1"})
```
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from langgraph.graph import END, START, StateGraph

from .langgraph_adapter import run_instrumented
from .recorder import Recorder


class Workflow:
    """A declarative, instrumented LangGraph workflow."""

    def __init__(
        self,
        name: str,
        *,
        state_schema: type,
        goal_nodes: Iterable[str] = (),
    ) -> None:
        self.name = name
        self._state_schema = state_schema
        self._goal_nodes = set(goal_nodes)
        self._nodes: dict[str, Callable] = {}
        self._tools: dict[str, list[str]] = {}
        self._global: set[str] = set()
        self._entry: str | None = None
        self._edges: list[tuple[str, str]] = []
        self._branches: dict[str, tuple[Callable, dict]] = {}
        self._finish: set[str] = set()
        self._compiled: Any = None

    # -- declaration ----------------------------------------------------------

    def step(self, fn=None, *, name: str | None = None, tools: Iterable[str] = (), _global=False):
        """Register a node. Usable as ``@wf.step`` or ``@wf.step(tools=[...])``."""

        def register(func: Callable) -> Callable:
            node = name or func.__name__
            self._nodes[node] = func
            self._tools[node] = list(tools)
            if _global:
                self._global.add(node)
            return func

        return register(fn) if fn is not None else register

    def global_step(self, fn=None, *, name: str | None = None, tools: Iterable[str] = ()):
        """Register a *global* node (entering it records an intent switch)."""
        return self.step(fn, name=name, tools=tools, _global=True)

    def entry(self, node: str) -> None:
        """Set the workflow's start node."""
        self._entry = node

    def edge(self, source: str, target: str) -> None:
        """Add a direct edge ``source -> target``."""
        self._edges.append((source, target))

    def branch(self, source: str, router: Callable, mapping: dict[str, str]) -> None:
        """Add conditional edges from ``source`` (``router(state) -> mapping key``)."""
        self._branches[source] = (router, mapping)

    def finish(self, node: str) -> None:
        """Mark ``node`` as a terminal node (edge to END)."""
        self._finish.add(node)

    # -- introspection --------------------------------------------------------

    @property
    def global_nodes(self) -> set[str]:
        return set(self._global)

    @property
    def node_tools(self) -> dict[str, list[str]]:
        return dict(self._tools)

    @property
    def goal_nodes(self) -> set[str]:
        return set(self._goal_nodes)

    # -- build / run ----------------------------------------------------------

    def compile(self):
        """Compile to a LangGraph ``CompiledStateGraph`` (cached)."""
        if self._compiled is not None:
            return self._compiled
        graph = StateGraph(self._state_schema)
        for node, func in self._nodes.items():
            graph.add_node(node, func)
        if self._entry is not None:
            graph.add_edge(START, self._entry)
        for source, target in self._edges:
            graph.add_edge(source, target)
        for source, (router, mapping) in self._branches.items():
            graph.add_conditional_edges(source, router, mapping)
        for node in self._finish:
            graph.add_edge(node, END)
        self._compiled = graph.compile()
        return self._compiled

    async def run(
        self,
        input: Any,
        recorder: Recorder,
        *,
        name: str | None = None,
        metadata: dict | None = None,
        config: dict | None = None,
    ) -> str:
        """Run the workflow under instrumentation; return the engagement id."""
        return await run_instrumented(
            self.compile(),
            input,
            recorder,
            name=name or self.name,
            metadata=metadata,
            global_nodes=self._global,
            node_tools=self._tools,
            goal_nodes=self._goal_nodes,
            config=config,
        )
