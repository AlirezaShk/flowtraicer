"""A reusable base graph-state carrying the channels the runner drains.

Extend this in your own ``TypedDict`` state so you don't re-declare xai's convention channels
(and their ``Annotated[..., add]`` reducers) in every project::

    from typing import Annotated, TypedDict
    from operator import add
    from xai.langgraph_adapter import TraceState

    class JourneyState(TraceState):
        messages: Annotated[list, add]   # your domain fields
        user_id: str

``messages`` is *not* included here — the runner never reads it; it's a normal LangGraph
field you own. ``total=False`` so you don't have to seed these channels in the initial input
(LangGraph initializes the reduced lists). See :mod:`xai.langgraph_adapter.runner` for what
each channel records.
"""

from __future__ import annotations

from operator import add
from typing import Annotated, TypedDict


class TraceState(TypedDict, total=False):
    """Base state with the xai channels the runner drains into the trace."""

    #: ``[{"name": str, "payload"?: dict}]`` -> tool-call events.
    tool_calls: Annotated[list, add]
    #: ``[{"name": str, "prompt_tokens": int, "completion_tokens": int, ...}]`` -> llm-call events.
    llm_calls: Annotated[list, add]
    #: ``[{"kind": str, "name": str, "payload"?: dict, "tokens"?: {...}}]`` -> typed events.
    events: Annotated[list, add]
    #: ``{"schema_name": str, "values": dict, ...}`` -> the step's extraction.
    extraction: dict
