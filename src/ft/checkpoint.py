"""Pluggable checkpointer factory for cross-turn pause/resume.

FlowTraicer's human-in-the-loop pause/resume (see ``docs/2026-06-25-checkpoint-resume-design.md``)
wraps **LangGraph's official savers** rather than re-implementing checkpoint serialization. The
checkpointer holds *resumable execution state* keyed by ``thread_id`` — deliberately separate from
the FlowTraicer trace store, which holds the *audit log* keyed by ``engagement_id``.

``build_checkpointer(backend, ...)`` maps a deployment to a saver:

- ``"memory"`` (default): :class:`langgraph.checkpoint.memory.MemorySaver` — zero-dependency,
  single-process. Resume works only within the same process, so a chat session must be pinned to
  one worker (session affinity). Great for dev, tests, and single-worker deployments.
- ``"sqlite"``: the official ``SqliteSaver`` (``pip install langgraph-checkpoint-sqlite``) — durable
  cross-process resume from a local file.
- ``"postgres"``: the official ``PostgresSaver`` (``pip install langgraph-checkpoint-postgres``) —
  durable cross-process resume in Postgres (pair it with ``PostgresStore`` for the audit log; they
  can share the same database, different tables).

You can also pass any ``BaseCheckpointSaver`` straight to ``Workflow(checkpointer=...)`` /
``Workflow.start(checkpointer=...)``; this factory is convenience, not a requirement.

One-time table provisioning (``setup``)
---------------------------------------

The durable savers (``"sqlite"`` / ``"postgres"``) need their checkpoint tables to exist before
first use. LangGraph's savers provide a ``.setup()`` method (idempotent — safe to call on every
boot) that ``CREATE TABLE IF NOT EXISTS``-es them. **``build_checkpointer`` runs ``setup()`` for you
by default** (``setup=True``), so a freshly-pointed empty database just works:

.. code-block:: python

    ckpt = build_checkpointer("postgres", dsn=DSN)            # tables provisioned on first build

The checkpointer tables live in **the same database as the FlowTraicer trace store** if you point
both at one DSN — they use distinct, non-colliding table names (LangGraph's ``checkpoints`` /
``checkpoint_writes`` / ``checkpoint_blobs`` vs FlowTraicer's own ``ft_*`` audit tables), so they
share a DB cleanly without interfering. Pass ``setup=False`` if you provision the tables out of band
(e.g. a migration) and don't want the factory to issue DDL:

.. code-block:: python

    ckpt = build_checkpointer("postgres", dsn=DSN, setup=False)   # caller provisioned the tables

``"memory"`` needs no setup (``setup`` is ignored for it).
"""

from __future__ import annotations

from typing import Any


def _enter_and_setup(cm: Any, *, setup: bool):
    """LangGraph's ``*Saver.from_conn_string(...)`` returns a context manager that yields the saver.

    We enter it to obtain a long-lived saver (kept open for the process), and — unless
    ``setup=False`` — run the idempotent ``.setup()`` to provision the checkpoint tables. Plain
    saver instances (not context managers) are returned as-is.
    """
    saver = cm.__enter__() if hasattr(cm, "__enter__") else cm
    if setup and hasattr(saver, "setup"):
        saver.setup()
    return saver


def build_checkpointer(backend: str = "memory", *, setup: bool = True, **kwargs: Any):
    """Return a LangGraph checkpointer for ``backend``, with its tables provisioned.

    :param backend: ``"memory"`` (default), ``"sqlite"`` (``path=``), or ``"postgres"`` (``dsn=``).
    :param setup: when ``True`` (default), run the saver's idempotent ``.setup()`` to create the
        checkpoint tables on first build (a no-op if they already exist). Pass ``False`` if you
        provision them out of band (e.g. a migration). Ignored for ``"memory"`` (no tables).
    :raises ValueError: for an unknown backend, or ``"postgres"`` without ``dsn=``.
    :raises ImportError: if the chosen backend's optional package isn't installed.
    """
    backend = (backend or "memory").lower()

    if backend == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()  # in-process, no tables to provision

    if backend == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "the 'sqlite' checkpointer needs `pip install langgraph-checkpoint-sqlite`"
            ) from exc
        path = kwargs.get("path", "ft_checkpoints.db")
        return _enter_and_setup(SqliteSaver.from_conn_string(path), setup=setup)

    if backend == "postgres":
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "the 'postgres' checkpointer needs `pip install langgraph-checkpoint-postgres`"
            ) from exc
        dsn = kwargs.get("dsn")
        if not dsn:
            raise ValueError("the 'postgres' checkpointer requires dsn=")
        return _enter_and_setup(PostgresSaver.from_conn_string(dsn), setup=setup)

    raise ValueError(f"unknown checkpointer backend: {backend!r}")
