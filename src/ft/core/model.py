"""The FlowTraicer trace data model.

A user<->agent engagement is modelled as a three-level tree::

    Engagement -> Step (workflow node) -> StepEvent

Steps carry per-step tools and a per-step extraction schema; ``global`` steps can
re-route the whole intent mid-engagement (recorded as an :class:`IntentSwitch`).

Everything here is framework-agnostic — it knows nothing about LangGraph. Adapters
populate these types through the recorder's emit API.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field, computed_field, model_validator


def _new_id() -> str:
    return uuid4().hex


def _now() -> datetime:
    return datetime.now(UTC)


class EngagementStatus(str, Enum):
    """Lifecycle state of an engagement."""

    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    #: The engagement ended without reaching a declared goal node (a drop-off).
    ABANDONED = "abandoned"
    #: Suspended at a human-in-the-loop interrupt, waiting to be ``resume``d. Not terminal —
    #: the engagement has not ended; it continues (same id) on the next ``resume`` (see
    #: ``docs/2026-06-25-checkpoint-resume-design.md``).
    PAUSED = "paused"


class TokenUsage(BaseModel):
    """LLM token usage for a single call.

    ``total`` defaults to ``prompt + completion`` when not given explicitly, so providers
    that report only a grand total (or only the split) both work.
    """

    prompt: int = 0
    completion: int = 0
    total: int = 0

    @model_validator(mode="after")
    def _default_total(self) -> TokenUsage:
        if self.total == 0 and (self.prompt or self.completion):
            self.total = self.prompt + self.completion
        return self


class StepStatus(str, Enum):
    """Lifecycle state of a single step."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    #: Parked at a human-in-the-loop interrupt (``ctx.pause``), awaiting the next ``resume``.
    WAITING = "waiting"


class EventKind(str, Enum):
    """The kind of thing an event records inside a step."""

    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    EXTRACTION = "extraction"
    LOG = "log"
    ERROR = "error"


class StepEvent(BaseModel):
    """An atomic, append-only record of something that happened inside a step."""

    id: str = Field(default_factory=_new_id)
    step_id: str
    kind: EventKind
    name: str
    ts: datetime = Field(default_factory=_now)
    duration_ms: float | None = None
    payload: dict = Field(default_factory=dict)
    error: str | None = None
    #: Token usage, for ``llm_call`` events. None for non-LLM events.
    tokens: TokenUsage | None = None


class Extraction(BaseModel):
    """The per-step structured-data result, validated against a declared schema."""

    schema_name: str
    json_schema: dict = Field(default_factory=dict)
    values: dict = Field(default_factory=dict)
    confidence: float | None = None
    valid: bool = True


class IntentSwitch(BaseModel):
    """A global step re-routing the workflow's intent."""

    id: str = Field(default_factory=_new_id)
    to_step: str
    reason: str
    from_step: str | None = None
    ts: datetime = Field(default_factory=_now)


class Step(BaseModel):
    """One workflow-node execution within an engagement."""

    id: str = Field(default_factory=_new_id)
    engagement_id: str
    name: str
    status: StepStatus = StepStatus.RUNNING
    started_at: datetime = Field(default_factory=_now)
    ended_at: datetime | None = None
    duration_ms: float | None = None
    parent_step_id: str | None = None
    is_global: bool = False
    tools_available: list[str] = Field(default_factory=list)
    extraction: Extraction | None = None
    events: list[StepEvent] = Field(default_factory=list)

    @computed_field
    @property
    def total_tokens(self) -> int:
        """Sum of token usage across this step's events (0 if none)."""
        return sum(e.tokens.total for e in self.events if e.tokens)


class NodeDef(BaseModel):
    """A node in the static workflow topology."""

    name: str
    is_global: bool = False
    tools: list[str] = Field(default_factory=list)


class EdgeDef(BaseModel):
    """An edge in the static workflow topology."""

    source: str
    target: str
    condition: str | None = None


class Topology(BaseModel):
    """The static workflow graph, read from the compiled engine."""

    nodes: list[NodeDef] = Field(default_factory=list)
    edges: list[EdgeDef] = Field(default_factory=list)


class Engagement(BaseModel):
    """A whole user<->agent session."""

    id: str = Field(default_factory=_new_id)
    name: str
    status: EngagementStatus = EngagementStatus.ACTIVE
    started_at: datetime = Field(default_factory=_now)
    ended_at: datetime | None = None
    #: The step the user dropped off at, when ``status`` is ABANDONED.
    dropped_at: str | None = None
    metadata: dict = Field(default_factory=dict)
    topology: Topology = Field(default_factory=Topology)
    steps: list[Step] = Field(default_factory=list)
    intent_switches: list[IntentSwitch] = Field(default_factory=list)

    @computed_field
    @property
    def total_tokens(self) -> int:
        """Sum of token usage across all steps (0 if none)."""
        return sum(step.total_tokens for step in self.steps)
