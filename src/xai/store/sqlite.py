"""Default store backend: SQLite + in-process async pub/sub.

Zero external dependencies — the library runs anywhere. Records are appended to a
single table; ``get_engagement`` folds them back into a tree. Live subscribers receive
each appended record through per-subscriber asyncio queues.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from collections.abc import AsyncIterator

from ..core.model import Engagement
from .base import EngagementSummary
from .reconstruct import fold
from .records import Record, RecordAdapter

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    engagement_id TEXT NOT NULL,
    type TEXT NOT NULL,
    ts TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_records_engagement ON records (engagement_id, seq);
"""


class SQLiteStore:
    """Append-only trace store backed by SQLite."""

    def __init__(self, path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        # Each subscriber's queue paired with the loop it was created on, so we can
        # publish safely from any thread (e.g. a recorder running off the server loop).
        self._subscribers: dict[asyncio.Queue[Record], asyncio.AbstractEventLoop] = {}

    # -- writes ---------------------------------------------------------------

    def append(self, record: Record) -> None:
        cur = self._conn.execute(
            "INSERT INTO records (engagement_id, type, ts, data) VALUES (?, ?, ?, ?)",
            (
                record.engagement_id,
                record.type,
                record.ts.isoformat(),
                record.model_dump_json(),
            ),
        )
        self._conn.commit()
        record.seq = cur.lastrowid
        self._publish(record)

    def _publish(self, record: Record) -> None:
        for queue, loop in list(self._subscribers.items()):
            try:
                loop.call_soon_threadsafe(queue.put_nowait, record)
            except Exception:  # pragma: no cover - best-effort fan-out
                logger.debug("dropped record for a slow subscriber", exc_info=True)

    # -- reads ----------------------------------------------------------------

    def _records_for(self, engagement_id: str) -> list[Record]:
        rows = self._conn.execute(
            "SELECT data FROM records WHERE engagement_id = ? ORDER BY seq",
            (engagement_id,),
        ).fetchall()
        return [RecordAdapter.validate_json(row[0]) for row in rows]

    def get_engagement(self, engagement_id: str) -> Engagement:
        records = self._records_for(engagement_id)
        if not records:
            raise KeyError(engagement_id)
        return fold(records)

    def list_engagements(self) -> list[EngagementSummary]:
        ids = self._conn.execute(
            "SELECT engagement_id FROM records GROUP BY engagement_id ORDER BY MIN(seq)"
        ).fetchall()
        return [EngagementSummary.from_engagement(self.get_engagement(i[0])) for i in ids]

    # -- live tail ------------------------------------------------------------

    async def subscribe(self) -> AsyncIterator[Record]:
        queue: asyncio.Queue[Record] = asyncio.Queue()
        self._subscribers[queue] = asyncio.get_running_loop()
        try:
            while True:
                yield await queue.get()
        finally:
            self._subscribers.pop(queue, None)

    def close(self) -> None:
        self._conn.close()
