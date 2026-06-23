"""Cross-engagement analytics — funnels, drop-off, and journey grouping.

These read whole engagements from a :class:`~xai.store.base.Store` and roll them up. The
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
    #: Engagements that reached this step but not the next one in the order.
    dropped: int
    #: Fraction that proceeded to the next step (``None`` for the final step).
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

    # Per step name: how many engagements reached it, the durations, and token totals.
    reached: dict[str, int] = defaultdict(int)
    durations: dict[str, list[float]] = defaultdict(list)
    tokens: dict[str, int] = defaultdict(int)
    for eng in engagements:
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
        is_last = i == len(order) - 1
        nxt = reached.get(order[i + 1], 0) if not is_last else 0
        dur = durations.get(name, [])
        steps.append(
            FunnelStep(
                name=name,
                reached=here,
                dropped=max(0, here - nxt) if not is_last else 0,
                conversion_rate=(None if is_last else (nxt / here if here else 0.0)),
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
