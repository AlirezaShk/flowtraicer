"""Drive a LangGraph run and record it into the FlowTraicer trace.

Consumes ``compiled.astream(stream_mode="debug")``, which emits a ``task`` chunk when a
node is entered and a ``task_result`` chunk when it exits. Each node becomes a
:class:`Step`; entering a *global* node records an :class:`IntentSwitch`.

To enrich a step, a node may write these conventional keys to graph state — the runner
records whatever it finds:

* ``tool_calls``: ``list[{"name": str, "payload"?: dict}]`` -> tool-call events (payload optional).
* ``llm_calls``: ``list[{"name": str, "prompt_tokens": int, "completion_tokens": int,
  "total_tokens"?: int, "duration_ms"?: float, "model"?: str}]`` -> llm-call events with
  token usage (so per-step / per-engagement token cost is captured).
* ``events``: ``list[{"kind": str, "name": str, "payload"?: dict, "duration_ms"?: float,
  "tokens"?: {...}, "error"?: str}]`` -> arbitrary typed events.
* ``extraction``: ``{"schema_name": str, "values": dict, ...}`` -> the step's
  :class:`Extraction`.

Passing ``goal_nodes`` marks the engagement ABANDONED (with ``dropped_at`` set to the last
step reached) if the run ends without entering any goal node — turning early exits into
first-class drop-offs.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..core.model import EngagementStatus, EventKind, Extraction, StepStatus, TokenUsage
from ..recorder import Recorder
from .topology import read_topology


def _as_dict(result: Any) -> dict:
    """Normalize a task_result ``result`` (dict, or list of (channel, value) pairs)."""
    if isinstance(result, dict):
        return result
    if isinstance(result, (list, tuple)):
        try:
            return dict(result)
        except (TypeError, ValueError):
            return {}
    return {}


async def run_instrumented(
    compiled,
    input: Any,
    recorder: Recorder,
    *,
    name: str = "engagement",
    metadata: dict | None = None,
    global_nodes: Iterable[str] = (),
    node_tools: Mapping[str, list[str]] | None = None,
    goal_nodes: Iterable[str] = (),
    config: dict | None = None,
) -> str:
    """Run ``compiled`` to completion, recording the engagement; return its id."""
    global_set = set(global_nodes)
    goal_set = set(goal_nodes)
    tools_map = dict(node_tools or {})

    topology = read_topology(compiled, global_nodes=global_set, node_tools=tools_map)
    engagement_id = recorder.start_engagement(name, metadata=metadata, topology=topology)

    # task id -> (step_id, node name, perf-counter start)
    open_tasks: dict[str, tuple[str, str, float]] = {}
    last_completed: str | None = None
    reached: set[str] = set()
    status = EngagementStatus.COMPLETED

    try:
        async for chunk in compiled.astream(input, stream_mode="debug", config=config):
            ctype = chunk.get("type")
            payload = chunk.get("payload", {})

            if ctype == "task":
                node = payload["name"]
                is_global = node in global_set
                if is_global:
                    recorder.record_intent_switch(
                        engagement_id,
                        to_step=node,
                        reason="global step entered",
                        from_step=last_completed,
                    )
                step_id = recorder.start_step(
                    engagement_id, node, tools=tools_map.get(node, []), is_global=is_global
                )
                open_tasks[payload["id"]] = (step_id, node, time.perf_counter())

            elif ctype == "task_result":
                opened = open_tasks.pop(payload.get("id"), None)
                if opened is None:
                    continue
                step_id, node, started = opened
                result = _as_dict(payload.get("result"))
                _record_step_writes(recorder, step_id, result)

                error = payload.get("error")
                duration_ms = (time.perf_counter() - started) * 1000.0
                recorder.end_step(
                    step_id,
                    StepStatus.FAILED if error else StepStatus.COMPLETED,
                    duration_ms=duration_ms,
                )
                if error:
                    recorder.record_event(step_id, EventKind.ERROR, "node_error", error=str(error))
                last_completed = node
                reached.add(node)
    except Exception as exc:  # the agent itself failed mid-run
        status = EngagementStatus.FAILED
        recorder.end_engagement(engagement_id, status, dropped_at=last_completed)
        raise exc

    dropped_at: str | None = None
    if goal_set and not (reached & goal_set):
        status = EngagementStatus.ABANDONED
        dropped_at = last_completed

    recorder.end_engagement(engagement_id, status, dropped_at=dropped_at)
    return engagement_id


@dataclass
class TurnResult:
    """The outcome of one checkpointed turn (``start``/``resume``).

    ``status`` is ``"paused"`` (a node called ``ctx.pause`` / ``interrupt``) or ``"completed"``.
    On a pause, ``awaiting`` / ``interrupt_payload`` carry the label and payload the paused node
    emitted (what to render and what input to collect for the next ``resume``).
    ``token_usage`` sums the tokens recorded by the steps advanced during **this** turn only.
    """

    engagement_id: str
    status: str  # "paused" | "completed"
    awaiting: str | None = None
    interrupt_payload: Any = None
    token_usage: TokenUsage | None = None


def _read_interrupt(compiled, config) -> tuple[bool, str | None, Any]:
    """Inspect the post-stream graph state: is it paused, and on what?

    Returns ``(paused, awaiting, payload)``. Paused iff there is a pending next node *and* a
    pending interrupt; the interrupt's value is the ``{"awaiting", "payload"}`` dict from
    ``ctx.pause``.
    """
    snapshot = compiled.get_state(config)
    interrupts = getattr(snapshot, "interrupts", ()) or ()
    if not (snapshot.next and interrupts):
        return False, None, None
    value = interrupts[0].value
    if isinstance(value, dict):
        return True, value.get("awaiting"), value.get("payload")
    return True, None, value


def read_parked_awaiting(compiled, config) -> tuple[bool, str | None, str | None]:
    """Return ``(is_paused, awaiting, interrupt_id)`` for the checkpointed graph at ``config``,
    **without** advancing it. Used by :meth:`ft.orchestration.Workflow.resume` to validate a resume
    before delivering input (``expect_awaiting=`` mismatch detection) and to build a ``Command``
    that can deliver ``None`` (LangGraph rejects a bare ``Command(resume=None)``; the interrupt-id
    -keyed map form is required for a ``None`` reply)."""
    snapshot = compiled.get_state(config)
    interrupts = getattr(snapshot, "interrupts", ()) or ()
    if not (snapshot.next and interrupts):
        return False, None, None
    value = interrupts[0].value
    awaiting = value.get("awaiting") if isinstance(value, dict) else None
    return True, awaiting, getattr(interrupts[0], "id", None)


async def run_instrumented_turn(
    compiled,
    stream_input: Any,
    recorder: Recorder,
    *,
    engagement_id: str | None = None,
    name: str = "engagement",
    metadata: dict | None = None,
    global_nodes: Iterable[str] = (),
    node_tools: Mapping[str, list[str]] | None = None,
    goal_nodes: Iterable[str] = (),
    config: dict | None = None,
) -> TurnResult:
    """Drive one **checkpointed** turn of ``compiled`` and record it.

    Unlike :func:`run_instrumented` (run-to-completion), this advances the graph until it either
    *pauses* at an ``interrupt`` (``ctx.pause``) or *completes*, and reports which via
    :class:`TurnResult`. It supports both starting a fresh engagement (``engagement_id=None``,
    ``stream_input`` = the initial state) and resuming an existing one (``engagement_id`` set,
    ``stream_input`` = a ``langgraph.types.Command(resume=...)``).

    A node that pauses is recorded with :class:`StepStatus.WAITING` (parked for human input); its
    state writes are not yet available, so only its entry is recorded for this turn. On the resume
    turn LangGraph re-invokes that node from the top, and it ends ``COMPLETED`` normally.
    """
    global_set = set(global_nodes)
    goal_set = set(goal_nodes)
    tools_map = dict(node_tools or {})

    if engagement_id is None:
        topology = read_topology(compiled, global_nodes=global_set, node_tools=tools_map)
        engagement_id = recorder.start_engagement(name, metadata=metadata, topology=topology)
    else:
        # Resuming: the engagement is in-flight again until it pauses/finishes.
        recorder.set_engagement_status(engagement_id, EngagementStatus.ACTIVE)

    open_tasks: dict[str, tuple[str, str, float]] = {}
    last_completed: str | None = None
    reached: set[str] = set()
    turn_prompt = turn_completion = turn_total = 0

    try:
        async for chunk in compiled.astream(stream_input, stream_mode="debug", config=config):
            ctype = chunk.get("type")
            payload = chunk.get("payload", {})

            if ctype == "task":
                node = payload["name"]
                is_global = node in global_set
                if is_global:
                    recorder.record_intent_switch(
                        engagement_id,
                        to_step=node,
                        reason="global step entered",
                        from_step=last_completed,
                    )
                step_id = recorder.start_step(
                    engagement_id, node, tools=tools_map.get(node, []), is_global=is_global
                )
                open_tasks[payload["id"]] = (step_id, node, time.perf_counter())

            elif ctype == "task_result":
                opened = open_tasks.pop(payload.get("id"), None)
                if opened is None:
                    continue
                step_id, node, started = opened
                duration_ms = (time.perf_counter() - started) * 1000.0

                # A node that interrupted carries pending interrupts and no usable result yet.
                if payload.get("interrupts"):
                    recorder.end_step(step_id, StepStatus.WAITING, duration_ms=duration_ms)
                    continue

                result = _as_dict(payload.get("result"))
                step_tokens = _record_step_writes(recorder, step_id, result)
                turn_prompt += step_tokens.prompt
                turn_completion += step_tokens.completion
                turn_total += step_tokens.total
                error = payload.get("error")
                recorder.end_step(
                    step_id,
                    StepStatus.FAILED if error else StepStatus.COMPLETED,
                    duration_ms=duration_ms,
                )
                if error:
                    recorder.record_event(step_id, EventKind.ERROR, "node_error", error=str(error))
                last_completed = node
                reached.add(node)
    except Exception as exc:
        recorder.end_engagement(engagement_id, EngagementStatus.FAILED, dropped_at=last_completed)
        raise exc

    turn_tokens = TokenUsage(prompt=turn_prompt, completion=turn_completion, total=turn_total)
    paused, awaiting, interrupt_payload = _read_interrupt(compiled, config)
    if paused:
        # The PAUSED status (not an EngagementEnded) is the marker that this engagement is parked.
        recorder.set_engagement_status(engagement_id, EngagementStatus.PAUSED)
        return TurnResult(engagement_id, "paused", awaiting, interrupt_payload, turn_tokens)

    status = EngagementStatus.COMPLETED
    dropped_at: str | None = None
    if goal_set and not (reached & goal_set):
        status = EngagementStatus.ABANDONED
        dropped_at = last_completed
    recorder.end_engagement(engagement_id, status, dropped_at=dropped_at)
    return TurnResult(engagement_id, "completed", token_usage=turn_tokens)


@dataclass
class StreamEvent:
    """One incremental event yielded by :func:`stream_instrumented_turn`.

    ``kind`` is one of ``"step_started"``, ``"token"``, ``"emit"``, ``"step_finished"``,
    ``"paused"``, ``"completed"``. ``data`` carries the payload (e.g. ``{"node": ...}`` for a
    step boundary, ``{"text": ...}`` for a token, the emitted card for ``"emit"``). The terminal
    event (``"paused"`` / ``"completed"``) additionally carries the resulting :class:`TurnResult`
    on ``turn`` (the orchestration layer maps it to a ``WorkflowTurn``).
    """

    kind: str
    data: dict = field(default_factory=dict)
    turn: Any = None


async def stream_instrumented_turn(
    compiled,
    stream_input: Any,
    recorder: Recorder,
    *,
    engagement_id: str | None = None,
    name: str = "engagement",
    metadata: dict | None = None,
    global_nodes: Iterable[str] = (),
    node_tools: Mapping[str, list[str]] | None = None,
    goal_nodes: Iterable[str] = (),
    config: dict | None = None,
) -> AsyncIterator[StreamEvent]:
    """Drive one checkpointed turn, **yielding events as they occur**, and record it.

    The async-generator counterpart of :func:`run_instrumented_turn`: it consumes LangGraph's
    combined ``["custom", "debug"]`` stream so that node entries/exits (debug ``task`` /
    ``task_result``) interleave with the ``token``/``emit`` payloads a node pushes via the stream
    writer (``ctx.llm(stream=True)`` / ``ctx.emit(...)``). It yields ``step_started`` / ``token`` /
    ``emit`` / ``step_finished`` as they happen and terminates with exactly one ``paused`` or
    ``completed`` event carrying the :class:`TurnResult` — the same boundary as
    :func:`run_instrumented_turn`. Token usage is still recorded into the step trace.
    """
    global_set = set(global_nodes)
    goal_set = set(goal_nodes)
    tools_map = dict(node_tools or {})

    if engagement_id is None:
        topology = read_topology(compiled, global_nodes=global_set, node_tools=tools_map)
        engagement_id = recorder.start_engagement(name, metadata=metadata, topology=topology)
    else:
        recorder.set_engagement_status(engagement_id, EngagementStatus.ACTIVE)

    open_tasks: dict[str, tuple[str, str, float]] = {}
    last_completed: str | None = None
    reached: set[str] = set()
    turn_prompt = turn_completion = turn_total = 0

    try:
        async for mode, chunk in compiled.astream(
            stream_input, stream_mode=["custom", "debug"], config=config
        ):
            if mode == "custom":
                if not isinstance(chunk, dict):
                    continue
                marker = chunk.get("_ft")
                if marker == "token":
                    yield StreamEvent("token", {"text": chunk.get("text", "")})
                elif marker == "emit":
                    yield StreamEvent("emit", chunk.get("payload") or {})
                continue

            # debug mode
            ctype = chunk.get("type")
            payload = chunk.get("payload", {})
            if ctype == "task":
                node = payload["name"]
                is_global = node in global_set
                if is_global:
                    recorder.record_intent_switch(
                        engagement_id,
                        to_step=node,
                        reason="global step entered",
                        from_step=last_completed,
                    )
                step_id = recorder.start_step(
                    engagement_id, node, tools=tools_map.get(node, []), is_global=is_global
                )
                open_tasks[payload["id"]] = (step_id, node, time.perf_counter())
                yield StreamEvent("step_started", {"node": node})

            elif ctype == "task_result":
                opened = open_tasks.pop(payload.get("id"), None)
                if opened is None:
                    continue
                step_id, node, started = opened
                duration_ms = (time.perf_counter() - started) * 1000.0
                if payload.get("interrupts"):
                    recorder.end_step(step_id, StepStatus.WAITING, duration_ms=duration_ms)
                    continue
                result = _as_dict(payload.get("result"))
                step_tokens = _record_step_writes(recorder, step_id, result)
                turn_prompt += step_tokens.prompt
                turn_completion += step_tokens.completion
                turn_total += step_tokens.total
                error = payload.get("error")
                recorder.end_step(
                    step_id,
                    StepStatus.FAILED if error else StepStatus.COMPLETED,
                    duration_ms=duration_ms,
                )
                if error:
                    recorder.record_event(step_id, EventKind.ERROR, "node_error", error=str(error))
                last_completed = node
                reached.add(node)
                yield StreamEvent("step_finished", {"node": node})
    except Exception as exc:
        recorder.end_engagement(engagement_id, EngagementStatus.FAILED, dropped_at=last_completed)
        raise exc

    turn_tokens = TokenUsage(prompt=turn_prompt, completion=turn_completion, total=turn_total)
    paused, awaiting, interrupt_payload = _read_interrupt(compiled, config)
    if paused:
        recorder.set_engagement_status(engagement_id, EngagementStatus.PAUSED)
        result = TurnResult(engagement_id, "paused", awaiting, interrupt_payload, turn_tokens)
        yield StreamEvent("paused", {"awaiting": awaiting}, turn=result)
        return

    status = EngagementStatus.COMPLETED
    dropped_at: str | None = None
    if goal_set and not (reached & goal_set):
        status = EngagementStatus.ABANDONED
        dropped_at = last_completed
    recorder.end_engagement(engagement_id, status, dropped_at=dropped_at)
    yield StreamEvent(
        "completed", {}, turn=TurnResult(engagement_id, "completed", token_usage=turn_tokens)
    )


def _record_step_writes(recorder: Recorder, step_id: str, result: dict) -> TokenUsage:
    """Record the conventional enrichment keys a node returned (tool_calls/llm_calls/events/…).

    Returns the token usage recorded for this step (summed across ``llm_calls`` and any token-
    carrying ``events``), so callers can accumulate per-turn usage.
    """
    prompt = completion = total = 0
    for call in result.get("tool_calls") or []:
        recorder.record_event(
            step_id,
            EventKind.TOOL_CALL,
            call.get("name", "tool"),
            payload=call.get("payload", {}),
        )
    for call in result.get("llm_calls") or []:
        usage = TokenUsage(
            prompt=call.get("prompt_tokens", 0),
            completion=call.get("completion_tokens", 0),
            total=call.get("total_tokens", 0),
        )
        recorder.record_llm_call(
            step_id,
            call.get("name", "llm"),
            prompt=usage.prompt,
            completion=usage.completion,
            total=usage.total,
            duration_ms=call.get("duration_ms"),
            model=call.get("model"),
        )
        prompt += usage.prompt
        completion += usage.completion
        total += usage.total
    for ev in result.get("events") or []:
        tokens = ev.get("tokens")
        usage = TokenUsage(**tokens) if tokens else None
        recorder.record_event(
            step_id,
            EventKind(ev.get("kind", "log")),
            ev.get("name", "event"),
            payload=ev.get("payload", {}),
            duration_ms=ev.get("duration_ms"),
            error=ev.get("error"),
            tokens=usage,
        )
        if usage:
            prompt += usage.prompt
            completion += usage.completion
            total += usage.total
    extraction = result.get("extraction")
    if extraction:
        recorder.record_extraction(step_id, Extraction(**extraction))
    return TokenUsage(prompt=prompt, completion=completion, total=total)
