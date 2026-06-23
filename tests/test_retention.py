"""Tests for retention: per-backend purge + retention policy."""

from datetime import UTC, datetime, timedelta

import fakeredis
import fakeredis.aioredis
import pytest

from xai.core.model import EngagementStatus
from xai.retention import RetentionPolicy, purge_before
from xai.store.records import EngagementEnded, EngagementStarted
from xai.store.redis import RedisStore
from xai.store.sqlite import SQLiteStore

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _seed(store, engagement_id, ended_at):
    store.append(EngagementStarted(engagement_id=engagement_id, name="e", ts=ended_at))
    store.append(
        EngagementEnded(
            engagement_id=engagement_id, status=EngagementStatus.COMPLETED, ended_at=ended_at
        )
    )


def _redis_store():
    server = fakeredis.FakeServer()
    return RedisStore(
        client=fakeredis.FakeStrictRedis(server=server, decode_responses=True),
        async_client=fakeredis.aioredis.FakeRedis(server=server, decode_responses=True),
    )


def test_sqlite_purge_removes_one_engagement():
    store = SQLiteStore()
    _seed(store, "e1", T0)
    assert store.purge("e1") is True
    with pytest.raises(KeyError):
        store.get_engagement("e1")
    assert store.list_engagements() == []
    assert store.purge("e1") is False  # already gone


def test_redis_purge_removes_one_engagement():
    store = _redis_store()
    _seed(store, "e1", T0)
    assert store.purge("e1") is True
    with pytest.raises(KeyError):
        store.get_engagement("e1")
    assert [s.id for s in store.list_engagements()] == []


def test_purge_before_removes_only_old_engagements():
    store = SQLiteStore()
    _seed(store, "old", T0)
    _seed(store, "new", T0 + timedelta(days=30))

    purged = purge_before(store, T0 + timedelta(days=10))
    assert purged == ["old"]
    assert [s.id for s in store.list_engagements()] == ["new"]


def test_retention_policy_applies_max_age():
    store = SQLiteStore()
    _seed(store, "old", T0)
    _seed(store, "recent", T0 + timedelta(days=20))

    policy = RetentionPolicy(max_age=timedelta(days=14))
    purged = policy.apply(store, now=T0 + timedelta(days=21))
    assert purged == ["old"]
    assert {s.id for s in store.list_engagements()} == {"recent"}
