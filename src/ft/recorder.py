"""The recorder — the emit contract every instrumentation source calls.

It turns high-level calls (``start_step``, ``record_event`` …) into append-only records
on a :class:`Store`. Every method is **fail-open**: a failure to record
is logged and swallowed, never raised into the caller — observability must not crash the
observed agent.
"""

from __future__ import annotations

import logging
from uuid import uuid4

from .core.model import (
    EngagementStatus,
    EventKind,
    Extraction,
    IntentSwitch,
    StepEvent,
    StepStatus,
    TokenUsage,
    Topology,
)
from .store.records import (
    EngagementEnded,
    EngagementStarted,
    EngagementStatusChanged,
    EventRecorded,
    ExtractionRecorded,
    IntentSwitched,
    StepEnded,
    StepStarted,
)

logger = logging.getLogger(__name__)


def _new_id() -> str:
    return uuid4().hex


class Recorder:
    """Translates engagement/step/event calls into append-only store records."""

    def __init__(self, store) -> None:
        self._store = store
        # step_id -> engagement_id, so callers need only pass the step.
        self._step_engagement: dict[str, str] = {}

    def _append(self, record) -> None:
        try:
            self._store.append(record)
        except Exception:
            logger.warning("failed to record %s", type(record).__name__, exc_info=True)

    def start_engagement(
        self,
        name: str,
        *,
        metadata: dict | None = None,
        topology: Topology | None = None,
    ) -> str:
        """Begin an engagement; return its id. Put ``user_id``/``session_id`` in ``metadata``."""
        engagement_id = _new_id()
        self._append(
            EngagementStarted(
                engagement_id=engagement_id,
                name=name,
                metadata=metadata or {},
                topology=topology or Topology(),
            )
        )
        return engagement_id

    def start_step(
        self,
        engagement_id: str,
        name: str,
        *,
        tools: list[str] | None = None,
        is_global: bool = False,
        parent: str | None = None,
    ) -> str:
        """Begin a step within an engagement; return its id."""
        step_id = _new_id()
        self._step_engagement[step_id] = engagement_id
        self._append(
            StepStarted(
                engagement_id=engagement_id,
                step_id=step_id,
                name=name,
                tools=tools or [],
                is_global=is_global,
                parent_step_id=parent,
            )
        )
        return step_id

    def record_event(
        self,
        step_id: str,
        kind: EventKind,
        name: str,
        *,
        payload: dict | None = None,
        duration_ms: float | None = None,
        error: str | None = None,
        tokens: TokenUsage | None = None,
    ) -> None:
        """Record an event of any :class:`EventKind` against ``step_id``."""
        engagement_id = self._step_engagement.get(step_id, "")
        event = StepEvent(
            step_id=step_id,
            kind=kind,
            name=name,
            payload=payload or {},
            duration_ms=duration_ms,
            error=error,
            tokens=tokens,
        )
        self._append(EventRecorded(engagement_id=engagement_id, event=event))

    def record_llm_call(
        self,
        step_id: str,
        name: str,
        *,
        prompt: int = 0,
        completion: int = 0,
        total: int = 0,
        duration_ms: float | None = None,
        model: str | None = None,
    ) -> None:
        """Record an ``llm_call`` event with token usage against ``step_id``."""
        payload = {"model": model} if model else {}
        self.record_event(
            step_id,
            EventKind.LLM_CALL,
            name,
            payload=payload,
            duration_ms=duration_ms,
            tokens=TokenUsage(prompt=prompt, completion=completion, total=total),
        )

    def record_extraction(self, step_id: str, extraction: Extraction) -> None:
        """Record a step's structured extraction (also surfaced as a timeline event)."""
        engagement_id = self._step_engagement.get(step_id, "")
        self._append(
            ExtractionRecorded(engagement_id=engagement_id, step_id=step_id, extraction=extraction)
        )
        # Also surface it on the timeline as an event.
        self.record_event(
            step_id,
            EventKind.EXTRACTION,
            extraction.schema_name,
            payload=extraction.values,
        )

    def record_intent_switch(
        self, engagement_id: str, *, to_step: str, reason: str, from_step: str | None = None
    ) -> None:
        """Record a global-step re-route of the engagement's intent."""
        self._append(
            IntentSwitched(
                engagement_id=engagement_id,
                intent_switch=IntentSwitch(to_step=to_step, reason=reason, from_step=from_step),
            )
        )

    def end_step(
        self,
        step_id: str,
        status: StepStatus = StepStatus.COMPLETED,
        *,
        duration_ms: float | None = None,
    ) -> None:
        """Close a step with a final status and optional measured duration."""
        engagement_id = self._step_engagement.get(step_id, "")
        self._append(
            StepEnded(
                engagement_id=engagement_id,
                step_id=step_id,
                status=status,
                duration_ms=duration_ms,
            )
        )

    def set_engagement_status(self, engagement_id: str, status: EngagementStatus) -> None:
        """Record a **non-terminal** status transition (e.g. ACTIVE<->PAUSED) without ending the
        engagement. Used by human-in-the-loop pause/resume; terminal states use
        :meth:`end_engagement`."""
        self._append(EngagementStatusChanged(engagement_id=engagement_id, status=status))

    def end_engagement(
        self,
        engagement_id: str,
        status: EngagementStatus = EngagementStatus.COMPLETED,
        *,
        dropped_at: str | None = None,
    ) -> None:
        """Close an engagement; for ABANDONED, ``dropped_at`` names the last step reached."""
        self._append(
            EngagementEnded(engagement_id=engagement_id, status=status, dropped_at=dropped_at)
        )
