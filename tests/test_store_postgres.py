"""Tests for the Postgres store backend (requires a real Postgres; set XAI_TEST_PG_DSN)."""

import asyncio
import os
from uuid import uuid4

import pytest

from xai.core.model import (
    EngagementStatus,
    EventKind,
    Extraction,
    StepEvent,
    StepStatus,
)
from xai.store.records import (
    EngagementEnded,
    EngagementStarted,
    EventRecorded,
    ExtractionRecorded,
    StepEnded,
    StepStarted,
)

PG_DSN = os.environ.get("XAI_TEST_PG_DSN")
pytestmark = pytest.mark.skipif(not PG_DSN, reason="set XAI_TEST_PG_DSN to run postgres tests")

if PG_DSN:
    from xai.store.postgres import PostgresStore


def _pg_store():
    return PostgresStore(PG_DSN, table=f"xai_test_{uuid4().hex[:8]}")


def _stream(engagement_id="e1", step_id="s1", **meta):
    return [
        EngagementStarted(engagement_id=engagement_id, name="house_search", metadata=meta),
        StepStarted(engagement_id=engagement_id, step_id=step_id, name="qualify", tools=["t"]),
        EventRecorded(
            engagement_id=engagement_id,
            event=StepEvent(step_id=step_id, kind=EventKind.TOOL_CALL, name="t"),
        ),
        ExtractionRecorded(
            engagement_id=engagement_id,
            step_id=step_id,
            extraction=Extraction(schema_name="B", values={"budget": 120000}),
        ),
        StepEnded(engagement_id=engagement_id, step_id=step_id, status=StepStatus.COMPLETED),
        EngagementEnded(engagement_id=engagement_id, status=EngagementStatus.COMPLETED),
    ]


def test_pg_append_and_get_roundtrip():
    store = _pg_store()
    for rec in _stream():
        store.append(rec)
    eng = store.get_engagement("e1")
    assert eng.name == "house_search"
    assert eng.steps[0].extraction.values["budget"] == 120000
    assert eng.status is EngagementStatus.COMPLETED


def test_pg_get_unknown_raises():
    store = _pg_store()
    with pytest.raises(KeyError):
        store.get_engagement("nope")


def test_pg_purge_removes_engagement():
    store = _pg_store()
    for rec in _stream("e1"):
        store.append(rec)
    assert store.purge("e1") is True
    with pytest.raises(KeyError):
        store.get_engagement("e1")
    assert store.purge("e1") is False


def test_pg_list_engagements_with_metadata_filter():
    store = _pg_store()
    for rec in _stream("e1", user_id="u1"):
        store.append(rec)
    for rec in _stream("e2", user_id="u2"):
        store.append(rec)

    assert {s.id for s in store.list_engagements()} == {"e1", "e2"}
    u1 = store.list_engagements(where={"user_id": "u1"})
    assert [s.id for s in u1] == ["e1"]
    assert u1[0].metadata["user_id"] == "u1"


@pytest.mark.asyncio
async def test_pg_subscribe_receives_appended_records():
    store = _pg_store()
    received = []

    async def consume():
        async for rec in store.subscribe():
            received.append(rec)
            if len(received) == 2:
                break

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.3)  # let LISTEN register
    store.append(EngagementStarted(engagement_id="live", name="x"))
    store.append(EngagementEnded(engagement_id="live", status=EngagementStatus.COMPLETED))
    await asyncio.wait_for(task, timeout=3.0)

    assert received[0].engagement_id == "live"
    assert isinstance(received[1], EngagementEnded)
