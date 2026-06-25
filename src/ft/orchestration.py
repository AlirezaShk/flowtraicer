"""A small declarative DSL for building instrumented LangGraph workflows.

Sugar over ``langgraph.StateGraph`` + :func:`ft.langgraph_adapter.run_instrumented`: declare
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

import inspect
from collections.abc import Callable, Iterable
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from .core.model import EngagementStatus, TokenUsage
from .langgraph_adapter import run_instrumented
from .langgraph_adapter.runner import (
    read_parked_awaiting,
    run_instrumented_turn,
    stream_instrumented_turn,
)
from .llm import LLMClient, LLMResult
from .recorder import Recorder
from .registry import REGISTER


class ResumeError(RuntimeError):
    """Raised by :meth:`Workflow.resume` when the thread cannot be resumed as asked.

    The ``reason`` distinguishes the documented failure modes so a router can branch:

    * ``"no_resumable_engagement"`` — no in-flight engagement is bound to this ``thread_id``
      (never started, or already ``completed``/``abandoned``/``failed`` — e.g. a double-submit
      or a stale replay of an already-finished turn).
    * ``"not_paused"`` — the engagement exists but the graph is not parked at a pause
      (nothing is awaiting input on this thread right now).
    * ``"awaiting_mismatch"`` — ``expect_awaiting=`` was given and does not match the label the
      graph is actually parked on (a stale client delivering the wrong-turn input).
    """

    def __init__(self, message: str, *, reason: str, thread_id: str) -> None:
        super().__init__(message)
        self.reason = reason
        self.thread_id = thread_id


class StepContext:
    """Per-step context handed to a node that declares a second parameter.

    ``await ctx.llm(prompt)`` calls the run's LLM client and **records the token usage
    automatically** — the node never shapes ``llm_calls`` by hand. (The state must carry the
    ``llm_calls`` channel; extend :class:`ft.langgraph_adapter.TraceState`.)

    ``ctx.deps`` holds the per-run dependencies passed to :meth:`Workflow.run` (``deps=``),
    e.g. request-scoped services. This is how a single compiled workflow stays reusable while
    each run gets its own dependencies — nodes read ``ctx.deps[...]`` instead of closing over
    request state.
    """

    def __init__(
        self,
        llm: LLMClient | None,
        deps: Any = None,
        *,
        tools: list | None = None,
        max_iterations: int = 8,
        stream_writer: Any = None,
    ) -> None:
        self._llm = llm
        self.deps = deps if deps is not None else {}
        self._llm_calls: list[dict] = []
        self._tool_calls: list[dict] = []
        # Bound the agentic tool list/limit (set by @wf.agent_step) for ctx.run_tools.
        self._tools = list(tools or [])
        self._max_iterations = max_iterations
        # LangGraph custom-stream writer (set only under Workflow.stream); None outside streaming.
        self._stream_writer = stream_writer

    async def llm(self, prompt, *, model: str | None = None, stream: bool = False, **kwargs) -> str:
        """Run an LLM turn via the workflow's client; return the text, record the tokens.

        With ``stream=True`` (only meaningful under :meth:`Workflow.stream` /
        :meth:`Workflow.stream_resume`), the client's ``astream(...)`` is consumed and each text
        chunk is pushed to the stream iterator as a ``token`` event while the full text is
        accumulated; the final token usage is still recorded into the step trace (so streaming and
        non-streaming account tokens identically). Outside a streaming run, ``stream=True`` is
        ignored and a normal ``acomplete`` is made.
        """
        if self._llm is None:
            raise RuntimeError(
                "this Workflow has no llm client; pass Workflow(..., llm=...) to use ctx.llm"
            )
        if model is not None:
            kwargs["model"] = model
        writer = self._stream_writer
        if stream and writer is not None and hasattr(self._llm, "astream"):
            text_parts: list[str] = []
            result = None
            async for chunk in self._llm.astream(prompt, **kwargs):
                if isinstance(chunk, str):
                    text_parts.append(chunk)
                    writer({"_ft": "token", "text": chunk})
                else:  # the final LLMResult carrying token usage
                    result = chunk
            if result is None:  # client streamed only text; synthesize a usage-less result
                result = LLMResult(text="".join(text_parts), tokens=TokenUsage(), model="stream")
            self._llm_calls.append(result.as_llm_call())
            return result.text
        result = await self._llm.acomplete(prompt, **kwargs)
        self._llm_calls.append(result.as_llm_call())
        return result.text

    def emit(self, payload: dict) -> None:
        """Emit a render-ready payload to the **stream** iterator as an ``emit`` event, mid-turn.

        Only meaningful under :meth:`Workflow.stream` / :meth:`Workflow.stream_resume`; outside a
        streaming run it is a no-op. Use this to push a card to the caller *without* pausing (a card
        that coincides with a wait should use ``ctx.pause(payload=...)`` instead, which surfaces on
        ``turn.interrupt``)."""
        writer = self._stream_writer
        if writer is not None:
            writer({"_ft": "emit", "payload": payload})

    async def run_tools(self, messages, *, tools: list | None = None, **kwargs) -> str:
        """Run a bounded ReAct tool-calling loop and return the model's final text answer.

        The model (the run's ``ctx.llm`` client, which must accept ``tools=`` and may return
        ``LLMResult.tool_calls``) chooses among ``tools`` turn after turn; FT executes each chosen
        tool's handler and **records each as a ``tool_call`` and each model round as an ``llm_call``
        under the running step**, feeding tool results back, until the model returns a final answer
        (no more tool calls) or ``max_iterations`` rounds elapse. ``tools`` defaults to the list
        declared on ``@wf.agent_step(tools=...)``.

        Tools are :class:`ft.agent.AgentTool` (an app-agnostic ``{name, description, parameters,
        handler}`` contract); handlers receive ``(args, ctx)`` so they can reach request-scoped
        services via ``ctx.deps``. **Replay note:** under pause/resume a node re-runs from its top,
        so a tool with side effects will re-execute — keep tool side effects idempotent, or place
        the agentic step *after* any ``ctx.pause`` so the loop only runs once.
        """
        if self._llm is None:
            raise RuntimeError("this Workflow has no llm client; pass llm=... to use ctx.run_tools")
        tool_list = list(tools if tools is not None else self._tools)
        by_name = {t.name: t for t in tool_list}
        specs = [t.spec() for t in tool_list]
        # Normalize the conversation into a mutable message list we append tool results onto.
        convo: list[dict] = (
            [{"role": "user", "content": m} if isinstance(m, str) else dict(m) for m in messages]
            if isinstance(messages, list)
            else [{"role": "user", "content": messages}]
        )
        for _ in range(self._max_iterations):
            result = await self._llm.acomplete(convo, tools=specs, **kwargs)
            self._llm_calls.append(result.as_llm_call())
            requests = list(getattr(result, "tool_calls", None) or [])
            if not requests:
                return result.text
            # Record the model's tool-call request turn, then execute each tool.
            convo.append(
                {
                    "role": "assistant",
                    "content": result.text,
                    "tool_calls": [{"name": r.name, "args": r.args, "id": r.id} for r in requests],
                }
            )
            for req in requests:
                tool = by_name.get(req.name)
                if tool is None:
                    output = {"error": f"unknown tool {req.name!r}"}
                else:
                    output = await tool.invoke(req.args, self)
                self._tool_calls.append({"name": req.name, "payload": {"args": req.args}})
                convo.append(
                    {"role": "tool", "name": req.name, "tool_call_id": req.id, "content": output}
                )
        # Exhausted the iteration budget without a final answer — make one last non-tool call.
        final = await self._llm.acomplete(convo, **kwargs)
        self._llm_calls.append(final.as_llm_call())
        return final.text

    async def pause(self, *, awaiting: str, payload: dict | None = None):
        """Emit ``payload`` to the caller and **suspend** the workflow for human input.

        Pausing requires the run to be checkpointed, so it only works under
        :meth:`Workflow.start` / :meth:`Workflow.resume` (not :meth:`Workflow.run`). On the *next*
        :meth:`Workflow.resume`, this call **returns** the value passed as ``input=`` and the node
        re-runs from its top — so keep any work before ``ctx.pause`` idempotent.

        The pausing turn surfaces ``awaiting`` and ``payload`` on the returned
        :class:`WorkflowTurn` (``turn.awaiting`` / ``turn.interrupt``) for the caller to render and
        to collect the reply. See ``docs/2026-06-25-checkpoint-resume-design.md``.
        """
        # ``interrupt`` raises a control-flow signal under a checkpointer; on resume it returns
        # the value supplied to Command(resume=...). It's NOT awaitable, but pause is async so the
        # node body can ``await ctx.pause(...)`` symmetrically with ``await ctx.llm(...)``.
        return interrupt({"awaiting": awaiting, "payload": payload})

    def _drain(self) -> list[dict]:
        return list(self._llm_calls)

    def _drain_tools(self) -> list[dict]:
        return list(self._tool_calls)


class WorkflowTurn(BaseModel):
    """The outcome of one human-in-the-loop turn (:meth:`Workflow.start` / :meth:`Workflow.resume`).

    ``status`` is ``"paused"`` (a node called ``ctx.pause`` — render ``interrupt`` and collect input
    for the next :meth:`Workflow.resume`) or ``"completed"`` (the workflow finished this turn).
    The whole multi-turn journey is **one** engagement (``engagement_id``), keyed to your session by
    ``thread_id``. See ``docs/2026-06-25-checkpoint-resume-design.md``.
    """

    engagement_id: str
    thread_id: str
    status: str  # "paused" | "completed"
    #: The label the paused node is waiting on (its ``ctx.pause(awaiting=...)``). None if completed.
    awaiting: str | None = None
    #: The payload the paused node emitted (e.g. a card to render), **unwrapped** — exactly the dict
    #: you passed to ``ctx.pause(payload=...)``, so ``MessageForm.model_validate(turn.interrupt)``
    #: works directly. None when completed. The ``awaiting`` label lives on ``turn.awaiting``.
    interrupt: dict | None = None
    #: Token usage for ONLY the steps advanced during THIS turn (not the whole engagement, which
    #: spans the multi-turn session) — charge your per-turn budget off ``turn.token_usage.total``.
    token_usage: TokenUsage = Field(default_factory=TokenUsage)

    @property
    def is_paused(self) -> bool:
        return self.status == "paused"


class StreamEvent(BaseModel):
    """One incremental event from :meth:`Workflow.stream` / :meth:`Workflow.stream_resume`.

    ``kind`` is one of:

    * ``"step_started"`` / ``"step_finished"`` — a node entered/exited (``data["node"]``); map to
      your ``status`` NDJSON line.
    * ``"token"`` — a streamed text chunk (``data["text"]``, from ``ctx.llm(stream=True)``); map to
      your ``text_chunk`` line.
    * ``"emit"`` — a render-ready payload pushed mid-turn (``data`` is the payload, from
      ``ctx.emit(...)``); map to e.g. a ``message_form`` line.
    * ``"paused"`` / ``"completed"`` — the **terminal** event; ``turn`` carries the resulting
      :class:`WorkflowTurn` (``turn.interrupt`` / ``turn.awaiting`` on pause; ``turn.token_usage``
      for the ``usage`` line). Exactly one terminal event ends the stream.
    """

    kind: str
    data: dict = Field(default_factory=dict)
    #: Present only on the terminal ``"paused"`` / ``"completed"`` event.
    turn: WorkflowTurn | None = None


class Workflow:
    """A declarative, instrumented LangGraph workflow.

    **Build once, run many.** A ``Workflow`` is a reusable definition: construct it once
    (e.g. a module-level singleton), and its LangGraph is compiled lazily and cached. Each
    :meth:`run` call passes the per-request ``input`` plus optional ``llm`` / ``deps`` — so the
    nodes never close over request state, and you don't rebuild/recompile the graph per request.

    ``llm`` here is an optional default; prefer passing the (often request-scoped) client and
    services to :meth:`run` via ``llm=`` / ``deps=``.

    **Multi-turn (human-in-the-loop).** :meth:`run` runs to completion and returns the engagement
    id (``str``). For chat flows that pause for human input across HTTP turns, use :meth:`start` /
    :meth:`resume` (keyed by a stable ``thread_id``) with a node that calls ``ctx.pause(...)``; both
    return a :class:`WorkflowTurn`. Pausing requires a checkpointer — pass ``checkpointer=`` (see
    :func:`ft.checkpoint.build_checkpointer`) or let one default to an in-process ``MemorySaver``.
    """

    def __init__(
        self,
        name: str,
        *,
        state_schema: type,
        goal_nodes: Iterable[str] = (),
        llm: LLMClient | None = None,
        checkpointer: Any = None,
    ) -> None:
        self.name = name
        self._state_schema = state_schema
        self._llm = llm
        self._checkpointer = checkpointer
        self._goal_nodes = set(goal_nodes)
        self._nodes: dict[str, Callable] = {}
        self._tools: dict[str, list[str]] = {}
        #: node -> (AgentTool list, max_iterations) for @wf.agent_step nodes (ctx.run_tools).
        self._agent_tools: dict[str, tuple[list, int]] = {}
        self._global: set[str] = set()
        self._entry: str | None = None
        self._edges: list[tuple[str, str]] = []
        self._branches: dict[str, tuple[Callable, dict]] = {}
        self._finish: set[str] = set()
        self._compiled: Any = None
        self._compiled_ckpt: Any = None

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

    def agent_step(
        self,
        fn=None,
        *,
        name: str | None = None,
        tools: Iterable = (),
        max_iterations: int = 8,
        _global: bool = False,
    ):
        """Register a node that runs a multi-tool agentic loop via ``ctx.run_tools(...)``.

        ``tools`` is a list of :class:`ft.agent.AgentTool` (executable ``{name, description,
        parameters, handler}`` specs). Inside the node, ``await ctx.run_tools(messages)`` lets the
        model choose among them in a bounded (``max_iterations``) ReAct loop; FT records each tool
        call and each LLM round under this step. The tool *names* are also registered as the step's
        available-tools (for the trace/topology), exactly like ``@wf.step(tools=[...])``.

        Usable as ``@wf.agent_step(tools=IMPLS)`` or with a ``max_iterations=N`` cap.
        """
        tool_list = list(tools)

        def register(func: Callable) -> Callable:
            node = name or func.__name__
            self.step(func, name=node, tools=[t.name for t in tool_list], _global=_global)
            self._agent_tools[node] = (tool_list, max_iterations)
            return func

        return register(fn) if fn is not None else register

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

    def _wrap(self, node: str, func: Callable) -> Callable:
        """Wrap a node so it receives a :class:`StepContext` (if it declares one) and the
        LLM/tool calls it made via ``ctx`` are merged into the node's state writes.

        The wrapper reads the per-run ``llm``/``deps`` from LangGraph's ``config`` (set by
        :meth:`run`), falling back to the workflow's construction-time ``llm``. So one compiled
        graph serves many runs, each with its own dependencies. For ``@wf.agent_step`` nodes the
        ``ctx`` is preloaded with the node's agentic tools so ``ctx.run_tools()`` works."""
        wants_ctx = len(inspect.signature(func).parameters) >= 2
        agent_tools, max_iters = self._agent_tools.get(node, ([], 8))

        async def wrapped(state, config=None):
            configurable = (config or {}).get("configurable", {})
            # Resolution order: per-run llm > construction-time llm > global registry default.
            llm = configurable.get("_ft_llm", self._llm) or REGISTER.get_llm_provider()
            # The stream writer is active only under a streaming run; a no-op otherwise. Gating on
            # the per-run flag keeps non-streaming runs from touching LangGraph's stream machinery.
            writer = None
            if configurable.get("_ft_streaming"):
                from langgraph.config import get_stream_writer

                try:
                    writer = get_stream_writer()
                except RuntimeError:  # not in a streaming context
                    writer = None
            ctx = StepContext(
                llm,
                configurable.get("_ft_deps"),
                tools=agent_tools,
                max_iterations=max_iters,
                stream_writer=writer,
            )
            result = func(state, ctx) if wants_ctx else func(state)
            if inspect.isawaitable(result):
                result = await result
            result = dict(result or {})
            calls = ctx._drain()
            if calls:
                result["llm_calls"] = list(result.get("llm_calls", [])) + calls
            tool_calls = ctx._drain_tools()
            if tool_calls:
                result["tool_calls"] = list(result.get("tool_calls", [])) + tool_calls
            return result

        return wrapped

    def _build_graph(self) -> StateGraph:
        """Assemble the (uncompiled) ``StateGraph`` from the declared nodes/edges/branches."""
        graph = StateGraph(self._state_schema)
        for node, func in self._nodes.items():
            graph.add_node(node, self._wrap(node, func))
        if self._entry is not None:
            graph.add_edge(START, self._entry)
        for source, target in self._edges:
            graph.add_edge(source, target)
        for source, (router, mapping) in self._branches.items():
            graph.add_conditional_edges(source, router, mapping)
        for node in self._finish:
            graph.add_edge(node, END)
        return graph

    def compile(self):
        """Compile to a LangGraph ``CompiledStateGraph`` (cached, **no** checkpointer)."""
        if self._compiled is None:
            self._compiled = self._build_graph().compile()
        return self._compiled

    def _resolve_checkpointer(self, checkpointer: Any) -> Any:
        """Pick the checkpointer for a checkpointed run: explicit > ctor > a lazy process-wide
        ``MemorySaver`` (created once and reused so threads in one process share resume state)."""
        if checkpointer is not None:
            return checkpointer
        if self._checkpointer is None:
            from .checkpoint import build_checkpointer

            self._checkpointer = build_checkpointer("memory")
        return self._checkpointer

    def _compile_checkpointed(self, checkpointer: Any):
        """Compile (cached) a variant *with* a checkpointer, for :meth:`start` / :meth:`resume`."""
        ckpt = self._resolve_checkpointer(checkpointer)
        # Cache only when using the workflow's own checkpointer; explicit per-call ones recompile.
        if checkpointer is None and self._compiled_ckpt is not None:
            return self._compiled_ckpt
        compiled = self._build_graph().compile(checkpointer=ckpt)
        if checkpointer is None:
            self._compiled_ckpt = compiled
        return compiled

    async def run(
        self,
        input: Any,
        recorder: Recorder,
        *,
        name: str | None = None,
        metadata: dict | None = None,
        config: dict | None = None,
        llm: LLMClient | None = None,
        deps: Any = None,
    ) -> str:
        """Run the (reusably-compiled) workflow under instrumentation; return the engagement id.

        ``llm`` and ``deps`` are **per-run** dependencies — the LLM client and request-scoped
        objects the nodes read via ``ctx``. They override the construction-time ``llm`` for this
        run only, so a single workflow built once can serve every request. The compiled graph is
        cached; only ``input``/``llm``/``deps`` change per call.
        """
        configurable = dict((config or {}).get("configurable", {}))
        if llm is not None:
            configurable["_ft_llm"] = llm
        if deps is not None:
            configurable["_ft_deps"] = deps
        merged = {**(config or {}), "configurable": configurable} if configurable else config
        return await run_instrumented(
            self.compile(),
            input,
            recorder,
            name=name or self.name,
            metadata=metadata,
            global_nodes=self._global,
            node_tools=self._tools,
            goal_nodes=self._goal_nodes,
            config=merged,
        )

    # -- multi-turn (human-in-the-loop) ---------------------------------------

    #: Metadata key binding an engagement to its thread (= chat session id) for resume lookup.
    THREAD_META_KEY = "ft_thread_id"

    def _merge_config(self, thread_id, config, llm, deps, *, streaming: bool = False) -> dict:
        configurable = dict((config or {}).get("configurable", {}))
        configurable["thread_id"] = thread_id
        if llm is not None:
            configurable["_ft_llm"] = llm
        if deps is not None:
            configurable["_ft_deps"] = deps
        if streaming:
            configurable["_ft_streaming"] = True
        return {**(config or {}), "configurable": configurable}

    def _resume_command(self, compiled, merged, thread_id, *, input, expect_awaiting):
        """Validate a parked resume and build the LangGraph ``Command`` to deliver ``input``.

        Shared by :meth:`resume` and :meth:`stream_resume`. Raises :class:`ResumeError` on a
        not-parked / wrong-``awaiting`` thread; handles the ``input=None`` interrupt-id-map form."""
        paused, awaiting, interrupt_id = read_parked_awaiting(compiled, merged)
        if not paused:
            raise ResumeError(
                f"thread_id={thread_id!r} is not parked at a pause (nothing awaiting input)",
                reason="not_paused",
                thread_id=thread_id,
            )
        if expect_awaiting is not None and awaiting != expect_awaiting:
            raise ResumeError(
                f"thread_id={thread_id!r} is parked on {awaiting!r}, not {expect_awaiting!r}",
                reason="awaiting_mismatch",
                thread_id=thread_id,
            )
        # LangGraph rejects a bare Command(resume=None); deliver a None reply via the interrupt-id
        # -keyed resume map instead, so ctx.pause(...) returns None (not {}).
        if input is None and interrupt_id is not None:
            return Command(resume={interrupt_id: None})
        return Command(resume=input)

    def _turn(self, result, thread_id: str) -> WorkflowTurn:
        payload = result.interrupt_payload
        if payload is not None and not isinstance(payload, dict):
            payload = {"value": payload}
        return WorkflowTurn(
            engagement_id=result.engagement_id,
            thread_id=thread_id,
            status=result.status,
            awaiting=result.awaiting,
            interrupt=payload,
            token_usage=result.token_usage or TokenUsage(),
        )

    async def start(
        self,
        input: Any,
        recorder: Recorder,
        *,
        thread_id: str,
        name: str | None = None,
        metadata: dict | None = None,
        config: dict | None = None,
        llm: LLMClient | None = None,
        deps: Any = None,
        checkpointer: Any = None,
    ) -> WorkflowTurn:
        """Begin a **checkpointed** run keyed by ``thread_id`` (your chat session id).

        Runs until a node pauses (``ctx.pause(...)``) or the workflow completes, then returns a
        :class:`WorkflowTurn`. The engagement is tagged with ``metadata[THREAD_META_KEY]=thread_id``
        so :meth:`resume` can find and continue it. Per-turn ``llm`` / ``deps`` work exactly as in
        :meth:`run`.
        """
        compiled = self._compile_checkpointed(checkpointer)
        merged = self._merge_config(thread_id, config, llm, deps)
        meta = {self.THREAD_META_KEY: thread_id, **(metadata or {})}
        result = await run_instrumented_turn(
            compiled,
            input,
            recorder,
            name=name or self.name,
            metadata=meta,
            global_nodes=self._global,
            node_tools=self._tools,
            goal_nodes=self._goal_nodes,
            config=merged,
        )
        return self._turn(result, thread_id)

    async def resume(
        self,
        thread_id: str,
        recorder: Recorder,
        *,
        input: Any = None,
        expect_awaiting: str | None = None,
        config: dict | None = None,
        llm: LLMClient | None = None,
        deps: Any = None,
        checkpointer: Any = None,
    ) -> WorkflowTurn:
        """Continue the engagement paused for ``thread_id``; deliver ``input`` to the paused node.

        ``input`` becomes the return value of that node's ``ctx.pause(...)`` (the human's reply).
        ``input=None`` (the default — a bare resume) makes ``ctx.pause(...)`` return ``None``
        (not ``{}``), so guard with ``reply or {}`` if a node assumes a dict. Records the resumed
        steps under the **same** engagement (found via ``metadata[THREAD_META_KEY]``), so the
        journey is one continuous engagement across turns. A single ``resume`` may itself return
        ``status="paused"`` again at a *different* node (chained pauses) — drive an N-card flow as
        ``while turn.is_paused: turn = await wf.resume(...)``.

        ``expect_awaiting`` optionally asserts the graph is parked on that exact label before
        delivering ``input``; a mismatch raises ``ResumeError(reason="awaiting_mismatch")`` so a
        stale client cannot deliver wrong-turn input.

        :raises ResumeError: ``reason="no_resumable_engagement"`` if no in-flight engagement is
            bound to ``thread_id`` (never started, or already completed/abandoned/failed — a
            double-submit / stale replay); ``reason="not_paused"`` if it isn't parked at a pause;
            ``reason="awaiting_mismatch"`` if ``expect_awaiting`` doesn't match the parked label.
        """
        engagement_id = self._find_engagement_for_thread(recorder, thread_id)
        if engagement_id is None:
            raise ResumeError(
                f"no resumable engagement for thread_id={thread_id!r} "
                "(never started, or already completed/abandoned/failed)",
                reason="no_resumable_engagement",
                thread_id=thread_id,
            )
        compiled = self._compile_checkpointed(checkpointer)
        merged = self._merge_config(thread_id, config, llm, deps)
        resume_cmd = self._resume_command(
            compiled, merged, thread_id, input=input, expect_awaiting=expect_awaiting
        )
        result = await run_instrumented_turn(
            compiled,
            resume_cmd,
            recorder,
            engagement_id=engagement_id,
            global_nodes=self._global,
            node_tools=self._tools,
            goal_nodes=self._goal_nodes,
            config=merged,
        )
        return self._turn(result, thread_id)

    # -- streaming (incremental turns) ----------------------------------------

    async def stream(
        self,
        input: Any,
        recorder: Recorder,
        *,
        thread_id: str,
        name: str | None = None,
        metadata: dict | None = None,
        config: dict | None = None,
        llm: LLMClient | None = None,
        deps: Any = None,
        checkpointer: Any = None,
    ):
        """Like :meth:`start`, but an **async generator** yielding :class:`StreamEvent`s as the turn
        executes (``step_started`` / ``token`` / ``emit`` / ``step_finished``), ending with exactly
        one terminal ``paused`` / ``completed`` event whose ``ev.turn`` is a :class:`WorkflowTurn`.

        Token chunks (``ev.kind == "token"``) come from ``ctx.llm(prompt, stream=True)`` and are
        still rolled into the step trace and ``turn.token_usage``. Use it inside a FastAPI generator
        to produce an NDJSON stream without awaiting the whole turn first.
        """
        compiled = self._compile_checkpointed(checkpointer)
        merged = self._merge_config(thread_id, config, llm, deps, streaming=True)
        meta = {self.THREAD_META_KEY: thread_id, **(metadata or {})}
        async for ev in stream_instrumented_turn(
            compiled,
            input,
            recorder,
            name=name or self.name,
            metadata=meta,
            global_nodes=self._global,
            node_tools=self._tools,
            goal_nodes=self._goal_nodes,
            config=merged,
        ):
            yield self._stream_event(ev, thread_id)

    async def stream_resume(
        self,
        thread_id: str,
        recorder: Recorder,
        *,
        input: Any = None,
        expect_awaiting: str | None = None,
        config: dict | None = None,
        llm: LLMClient | None = None,
        deps: Any = None,
        checkpointer: Any = None,
    ):
        """Like :meth:`resume`, but an **async generator** yielding :class:`StreamEvent`s (see
        :meth:`stream`). Raises :class:`ResumeError` (same contract as :meth:`resume`) before
        yielding any event if the thread can't be resumed as asked."""
        engagement_id = self._find_engagement_for_thread(recorder, thread_id)
        if engagement_id is None:
            raise ResumeError(
                f"no resumable engagement for thread_id={thread_id!r} "
                "(never started, or already completed/abandoned/failed)",
                reason="no_resumable_engagement",
                thread_id=thread_id,
            )
        compiled = self._compile_checkpointed(checkpointer)
        merged = self._merge_config(thread_id, config, llm, deps, streaming=True)
        resume_cmd = self._resume_command(
            compiled, merged, thread_id, input=input, expect_awaiting=expect_awaiting
        )
        async for ev in stream_instrumented_turn(
            compiled,
            resume_cmd,
            recorder,
            engagement_id=engagement_id,
            global_nodes=self._global,
            node_tools=self._tools,
            goal_nodes=self._goal_nodes,
            config=merged,
        ):
            yield self._stream_event(ev, thread_id)

    def _stream_event(self, ev, thread_id: str) -> StreamEvent:
        """Map a runner ``StreamEvent`` to the public one, converting any terminal ``TurnResult``
        to a :class:`WorkflowTurn`."""
        turn = self._turn(ev.turn, thread_id) if ev.turn is not None else None
        return StreamEvent(kind=ev.kind, data=ev.data, turn=turn)

    @staticmethod
    def _find_engagement_for_thread(recorder: Recorder, thread_id: str) -> str | None:
        """Find the most-recent not-yet-completed engagement bound to ``thread_id`` in the store."""
        store = recorder._store
        matches = store.list_engagements(where={Workflow.THREAD_META_KEY: thread_id})
        terminal = {EngagementStatus.COMPLETED, EngagementStatus.ABANDONED, EngagementStatus.FAILED}
        # list_engagements is oldest-first; prefer the latest still-resumable one.
        for summary in reversed(matches):
            if summary.status not in terminal:
                return summary.id
        return None
