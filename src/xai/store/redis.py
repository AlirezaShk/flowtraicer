"""Redis Streams store backend — great for cross-process live monitoring.

Each engagement's records go to a per-engagement stream (``<prefix>:eng:<id>``) for
reconstruction, and to a global stream (``<prefix>:all``) that subscribers tail with a
blocking ``XREAD`` — so live monitoring works across processes (the recorder and the viewer
can be different services pointed at the same Redis).

Append is synchronous (``redis.Redis``); :meth:`subscribe` is async (``redis.asyncio``).
``redis`` is imported lazily so this module imports without it installed; tests inject
fakeredis clients.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from ..core.model import Engagement
from .base import EngagementSummary, matches
from .reconstruct import fold
from .records import Record, RecordAdapter


def _data(fields: dict) -> str:
    """Pull the JSON payload out of a stream entry's fields (str or bytes keys)."""
    value = fields.get("data", fields.get(b"data"))
    return value.decode() if isinstance(value, bytes) else value


def _text(value) -> str:
    return value.decode() if isinstance(value, bytes) else value


class RedisStore:
    """Append-only trace store backed by Redis Streams."""

    def __init__(
        self,
        url: str | None = None,
        *,
        client=None,
        async_client=None,
        key_prefix: str = "xai",
    ) -> None:
        self._url = url
        self._client = client
        self._async_client = async_client
        self._prefix = key_prefix
        if client is None and url is not None:
            import redis

            self._client = redis.Redis.from_url(url, decode_responses=True)

    # -- keys -----------------------------------------------------------------

    def _eng_key(self, engagement_id: str) -> str:
        return f"{self._prefix}:eng:{engagement_id}"

    @property
    def _all_key(self) -> str:
        return f"{self._prefix}:all"

    @property
    def _ids_key(self) -> str:
        return f"{self._prefix}:engagement_ids"

    # -- writes ---------------------------------------------------------------

    def append(self, record: Record) -> None:
        """Persist a record to its engagement stream + the global tail."""
        from .records import EngagementStarted

        data = record.model_dump_json()
        self._client.xadd(self._eng_key(record.engagement_id), {"data": data})
        self._client.xadd(self._all_key, {"data": data})
        if isinstance(record, EngagementStarted):
            self._client.rpush(self._ids_key, record.engagement_id)

    # -- reads ----------------------------------------------------------------

    def get_engagement(self, engagement_id: str) -> Engagement:
        """Reconstruct one engagement from its stream. Raises ``KeyError`` if unknown."""
        entries = self._client.xrange(self._eng_key(engagement_id))
        if not entries:
            raise KeyError(engagement_id)
        records = [RecordAdapter.validate_json(_data(fields)) for _id, fields in entries]
        return fold(records)

    def list_engagements(self, where: dict | None = None) -> list[EngagementSummary]:
        """Summaries (oldest first), optionally filtered to those matching ``where``."""
        ids = [_text(i) for i in self._client.lrange(self._ids_key, 0, -1)]
        summaries = (EngagementSummary.from_engagement(self.get_engagement(i)) for i in ids)
        return [s for s in summaries if matches(s.metadata, where)]

    # -- live tail ------------------------------------------------------------

    def _get_async_client(self):
        if self._async_client is None:
            import redis.asyncio

            self._async_client = redis.asyncio.Redis.from_url(self._url, decode_responses=True)
        return self._async_client

    async def subscribe(self) -> AsyncIterator[Record]:
        """Yield records as they are appended, tailing the global stream (cross-process)."""
        client = self._get_async_client()
        last_id = "$"  # only records appended after we start
        while True:
            response = await client.xread({self._all_key: last_id}, block=1000)
            if not response:
                continue
            for _stream_key, entries in response:
                for entry_id, fields in entries:
                    last_id = entry_id
                    yield RecordAdapter.validate_json(_data(fields))
