"""Tests for the Redis Streams store backend (offline via fakeredis)."""

import asyncio

import fakeredis
import fakeredis.aioredis
import pytest

from ft.core.model import EngagementStatus, EventKind, Extraction, StepEvent, StepStatus
from ft.store.records import (
    EngagementEnded,
    EngagementStarted,
    EventRecorded,
    ExtractionRecorded,
    StepEnded,
    StepStarted,
)
from ft.store.redis import RedisStore


def _redis_store():
    server = fakeredis.FakeServer()
    sync = fakeredis.FakeStrictRedis(server=server, decode_responses=True)
    aio = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    return RedisStore(client=sync, async_client=aio)


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


def test_redis_append_and_get_roundtrip():
    store = _redis_store()
    for rec in _stream():
        store.append(rec)
    eng = store.get_engagement("e1")
    assert eng.name == "house_search"
    assert eng.steps[0].extraction.values["budget"] == 120000
    assert eng.status is EngagementStatus.COMPLETED


def test_redis_get_unknown_raises():
    store = _redis_store()
    with pytest.raises(KeyError):
        store.get_engagement("nope")


def test_redis_list_engagements_with_metadata_filter():
    store = _redis_store()
    for rec in _stream("e1", user_id="u1"):
        store.append(rec)
    for rec in _stream("e2", user_id="u2"):
        store.append(rec)

    assert {s.id for s in store.list_engagements()} == {"e1", "e2"}
    u1 = store.list_engagements(where={"user_id": "u1"})
    assert [s.id for s in u1] == ["e1"]


@pytest.mark.asyncio
async def test_redis_subscribe_receives_appended_records():
    store = _redis_store()
    received = []

    async def consume():
        async for rec in store.subscribe():
            received.append(rec)
            if len(received) == 2:
                break

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.1)  # let the subscriber start blocking on XREAD
    store.append(EngagementStarted(engagement_id="live", name="x"))
    store.append(EngagementEnded(engagement_id="live", status=EngagementStatus.COMPLETED))
    await asyncio.wait_for(task, timeout=2.0)

    assert received[0].engagement_id == "live"
    assert isinstance(received[1], EngagementEnded)
