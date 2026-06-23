"""Tests for cross-engagement analytics: summaries, filtering, and the funnel."""

from xai.analytics import funnel, group_by, journeys
from xai.core.model import EngagementStatus, StepStatus
from xai.recorder import Recorder
from xai.store.sqlite import SQLiteStore

ORDER = ["intake", "selection", "comparison", "application", "submitted"]


def _journey(store, user_id, session_id, steps, *, completed=True):
    rec = Recorder(store)
    eid = rec.start_engagement(
        "school_journey", metadata={"user_id": user_id, "session_id": session_id}
    )
    last = None
    for name in steps:
        sid = rec.start_step(eid, name)
        rec.record_llm_call(sid, "gpt", prompt=10, completion=5)
        rec.end_step(sid, StepStatus.COMPLETED, duration_ms=5.0)
        last = name
    if completed:
        rec.end_engagement(eid)
    else:
        rec.end_engagement(eid, EngagementStatus.ABANDONED, dropped_at=last)
    return eid


def _seed(store):
    _journey(store, "u1", "s1", ["intake", "selection", "comparison", "application", "submitted"])
    _journey(store, "u2", "s1", ["intake", "selection", "application", "submitted"])
    _journey(store, "u3", "s1", ["intake", "selection"], completed=False)
    _journey(store, "u1", "s2", ["intake"], completed=False)


def test_summary_includes_metadata_tokens_and_dropoff():
    store = SQLiteStore()
    _journey(store, "u9", "s9", ["intake", "selection"], completed=False)
    summary = store.list_engagements()[0]
    assert summary.metadata["user_id"] == "u9"
    assert summary.total_tokens == 30  # two steps × 15
    assert summary.status is EngagementStatus.ABANDONED
    assert summary.dropped_at == "selection"


def test_list_engagements_filter_by_metadata():
    store = SQLiteStore()
    _seed(store)
    u1 = store.list_engagements(where={"user_id": "u1"})
    assert {s.metadata["session_id"] for s in u1} == {"s1", "s2"}


def test_funnel_reached_dropped_conversion_tokens():
    store = SQLiteStore()
    _seed(store)
    f = funnel(store, ORDER)
    by_name = {s.name: s for s in f.steps}

    assert by_name["intake"].reached == 4
    assert by_name["selection"].reached == 3
    assert by_name["comparison"].reached == 1
    assert by_name["application"].reached == 2
    assert by_name["submitted"].reached == 2

    # intake -> selection: one of four dropped
    assert by_name["intake"].dropped == 1
    assert by_name["intake"].conversion_rate == 0.75

    # tokens accrue per step (15 per reached step)
    assert by_name["intake"].total_tokens == 60
    assert by_name["submitted"].total_tokens == 30
    assert by_name["intake"].avg_duration_ms == 5.0

    # last step has no onward conversion
    assert by_name["submitted"].conversion_rate is None


def test_journeys_and_group_by():
    store = SQLiteStore()
    _seed(store)

    assert len(journeys(store, user_id="u1")) == 2
    grouped = group_by(store, "user_id")
    assert {k: len(v) for k, v in grouped.items()} == {"u1": 2, "u2": 1, "u3": 1}
