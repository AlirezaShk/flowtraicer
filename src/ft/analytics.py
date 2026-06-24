"""Cross-engagement analytics — funnels, drop-off, and journey grouping.

These read whole engagements from a :class:`~ft.store.base.Store` and roll them up. The
canonical question for a product built on agent journeys is *"where do users drop off, and
what did each step cost?"* — :func:`funnel` answers exactly that.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from pydantic import BaseModel

from .core.model import Engagement, StepStatus
from .store.base import EngagementSummary


class FunnelStep(BaseModel):
    """One step of a funnel, aggregated across many engagements."""

    name: str
    #: Engagements that reached this step.
    reached: int
    #: Engagements that ended *at* this step (from the recorded ``dropped_at``).
    dropped: int
    #: Fraction that continued past this step, ``(reached - dropped) / reached`` — always in
    #: [0, 1]; ``None`` for the final step. Correct even with optional steps in the order.
    conversion_rate: float | None
    #: Mean wall-clock duration of this step (over engagements that have a measure).
    avg_duration_ms: float | None
    #: Total token usage across this step.
    total_tokens: int


class Funnel(BaseModel):
    """A funnel over an ordered list of step names."""

    order: list[str]
    total: int  # engagements considered
    steps: list[FunnelStep]


def _load(store, where: dict | None) -> list[Engagement]:
    return [store.get_engagement(s.id) for s in store.list_engagements(where)]


def funnel(store, order: list[str], *, where: dict | None = None) -> Funnel:
    """Build a funnel over ``order`` from the engagements in ``store``.

    ``where`` filters engagements by metadata (e.g. ``{"cohort": "june"}``).
    """
    engagements = _load(store, where)

    # Per step name: how many engagements reached it, the durations, token totals, and how
    # many *ended* there. Drop-off is read from the recorded ``dropped_at`` rather than from
    # subtracting reached-counts, so it stays correct when the order contains optional steps
    # (where a later step can have a higher reached-count than an earlier optional one).
    reached: dict[str, int] = defaultdict(int)
    durations: dict[str, list[float]] = defaultdict(list)
    tokens: dict[str, int] = defaultdict(int)
    dropped: dict[str, int] = defaultdict(int)
    for eng in engagements:
        if eng.dropped_at is not None:
            dropped[eng.dropped_at] += 1
        seen: set[str] = set()
        for step in eng.steps:
            if step.name in seen:
                continue
            seen.add(step.name)
            reached[step.name] += 1
            tokens[step.name] += step.total_tokens
            if step.duration_ms is not None:
                durations[step.name].append(step.duration_ms)

    steps: list[FunnelStep] = []
    for i, name in enumerate(order):
        here = reached.get(name, 0)
        dropped_here = dropped.get(name, 0)
        is_last = i == len(order) - 1
        dur = durations.get(name, [])
        # Fraction that continued past this step (didn't abandon here). Always in [0, 1].
        conversion = None if is_last else ((here - dropped_here) / here if here else 0.0)
        steps.append(
            FunnelStep(
                name=name,
                reached=here,
                dropped=dropped_here,
                conversion_rate=conversion,
                avg_duration_ms=(sum(dur) / len(dur) if dur else None),
                total_tokens=tokens.get(name, 0),
            )
        )

    return Funnel(order=order, total=len(engagements), steps=steps)


def journeys(store, **metadata_filter: Any) -> list[EngagementSummary]:
    """Engagement summaries matching the given metadata (e.g. ``journeys(store, user_id="u1")``)."""
    return store.list_engagements(where=metadata_filter or None)


def group_by(store, key: str, *, where: dict | None = None) -> dict[Any, list[EngagementSummary]]:
    """Group engagement summaries by ``metadata[key]``."""
    grouped: dict[Any, list[EngagementSummary]] = defaultdict(list)
    for summary in store.list_engagements(where):
        grouped[summary.metadata.get(key)].append(summary)
    return dict(grouped)


# ``StepStatus`` is re-exported for callers building their own roll-ups.
__all__ = ["Funnel", "FunnelStep", "funnel", "group_by", "journeys", "StepStatus"]
