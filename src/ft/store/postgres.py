"""Postgres store backend — durable, queryable, with LISTEN/NOTIFY live monitoring.

Records are appended to a JSONB table; engagements are reconstructed by folding the rows
for an id. Live subscribers ``LISTEN`` on a channel that every append signals via
``pg_notify`` (with the new row's ``seq``), then fetch that row — so monitoring works across
processes against a shared database.

Append/query are synchronous (``psycopg``); :meth:`subscribe` opens its own async connection.
``psycopg`` is imported lazily so this module imports without it installed.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator

from ..core.model import Engagement
from .base import EngagementSummary, matches
from .reconstruct import fold
from .records import Record, RecordAdapter

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class PostgresStore:
    """Append-only trace store backed by Postgres (JSONB + LISTEN/NOTIFY)."""

    def __init__(self, dsn: str | None = None, *, conn=None, table: str = "ft_records") -> None:
        if not _IDENT.match(table):
            raise ValueError(f"invalid table name: {table!r}")
        self._dsn = dsn
        self._table = table
        self._channel = table  # one NOTIFY channel per table
        if conn is not None:
            self._conn = conn
        else:
            import psycopg

            self._conn = psycopg.connect(dsn, autocommit=True)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self._conn.execute(
            f"CREATE TABLE IF NOT EXISTS {self._table} ("
            " seq BIGSERIAL PRIMARY KEY,"
            " engagement_id TEXT NOT NULL,"
            " type TEXT NOT NULL,"
            " ts TIMESTAMPTZ NOT NULL,"
            " data JSONB NOT NULL)"
        )
        self._conn.execute(
            f"CREATE INDEX IF NOT EXISTS {self._table}_eng_idx"
            f" ON {self._table} (engagement_id, seq)"
        )

    # -- writes ---------------------------------------------------------------

    def append(self, record: Record) -> None:
        """Persist a record and NOTIFY live subscribers with its new ``seq``."""
        row = self._conn.execute(
            f"INSERT INTO {self._table} (engagement_id, type, ts, data)"
            " VALUES (%s, %s, %s, %s::jsonb) RETURNING seq",
            (record.engagement_id, record.type, record.ts, record.model_dump_json()),
        ).fetchone()
        record.seq = row[0]
        self._conn.execute("SELECT pg_notify(%s, %s)", (self._channel, str(row[0])))

    # -- reads ----------------------------------------------------------------

    def get_engagement(self, engagement_id: str) -> Engagement:
        """Reconstruct one engagement. Raises ``KeyError`` if unknown."""
        rows = self._conn.execute(
            f"SELECT data FROM {self._table} WHERE engagement_id = %s ORDER BY seq",
            (engagement_id,),
        ).fetchall()
        if not rows:
            raise KeyError(engagement_id)
        return fold([RecordAdapter.validate_python(r[0]) for r in rows])

    def list_engagements(self, where: dict | None = None) -> list[EngagementSummary]:
        """Summaries (oldest first), optionally filtered to those matching ``where``."""
        ids = self._conn.execute(
            f"SELECT engagement_id FROM {self._table} GROUP BY engagement_id ORDER BY MIN(seq)"
        ).fetchall()
        summaries = (EngagementSummary.from_engagement(self.get_engagement(i[0])) for i in ids)
        return [s for s in summaries if matches(s.metadata, where)]

    # -- live tail ------------------------------------------------------------

    async def subscribe(self) -> AsyncIterator[Record]:
        """Yield records as they are appended (via LISTEN/NOTIFY; cross-process)."""
        import psycopg

        aconn = await psycopg.AsyncConnection.connect(self._dsn, autocommit=True)
        try:
            await aconn.execute(f"LISTEN {self._channel}")
            async for notify in aconn.notifies():
                # Fetch the row on the sync connection — querying the listen connection
                # mid-iteration would stall its notifies() generator.
                row = self._conn.execute(
                    f"SELECT data FROM {self._table} WHERE seq = %s", (int(notify.payload),)
                ).fetchone()
                if row is not None:
                    yield RecordAdapter.validate_python(row[0])
        finally:
            await aconn.close()

    def purge(self, engagement_id: str) -> bool:
        """Delete an entire engagement (retention). Returns True if it existed."""
        cur = self._conn.execute(
            f"DELETE FROM {self._table} WHERE engagement_id = %s", (engagement_id,)
        )
        return cur.rowcount > 0

    def close(self) -> None:
        """Close the underlying connection."""
        self._conn.close()
