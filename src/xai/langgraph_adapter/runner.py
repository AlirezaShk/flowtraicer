"""Drive a LangGraph run and record it into the xai trace.

Consumes ``compiled.astream(stream_mode="debug")``, which emits a ``task`` chunk when a
node is entered and a ``task_result`` chunk when it exits. Each node becomes a
:class:`Step`; entering a *global* node records an :class:`IntentSwitch`.

To enrich a step, a node may write these conventional keys to graph state — the runner
records whatever it finds:

* ``tool_calls``: ``list[{"name": str, "payload": dict}]`` -> tool-call events.
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
from collections.abc import Iterable, Mapping
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

                for call in result.get("tool_calls") or []:
                    recorder.record_event(
                        step_id,
                        EventKind.TOOL_CALL,
                        call.get("name", "tool"),
                        payload=call.get("payload", {}),
                    )
                for call in result.get("llm_calls") or []:
                    recorder.record_llm_call(
                        step_id,
                        call.get("name", "llm"),
                        prompt=call.get("prompt_tokens", 0),
                        completion=call.get("completion_tokens", 0),
                        total=call.get("total_tokens", 0),
                        duration_ms=call.get("duration_ms"),
                        model=call.get("model"),
                    )
                for ev in result.get("events") or []:
                    tokens = ev.get("tokens")
                    recorder.record_event(
                        step_id,
                        EventKind(ev.get("kind", "log")),
                        ev.get("name", "event"),
                        payload=ev.get("payload", {}),
                        duration_ms=ev.get("duration_ms"),
                        error=ev.get("error"),
                        tokens=TokenUsage(**tokens) if tokens else None,
                    )
                extraction = result.get("extraction")
                if extraction:
                    recorder.record_extraction(step_id, Extraction(**extraction))

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
