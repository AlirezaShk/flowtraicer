"""Checkpointer factory contract, incl. one-time table provisioning (NEEDS.md #E).

``build_checkpointer(backend, *, setup=True, ...)`` returns a ready-to-use LangGraph saver and, for
the durable backends, provisions its checkpoint tables (runs the saver's idempotent ``.setup()``)
on first build. The memory backend is exercised here directly; the durable backends' real
round-trips stay skipped (they need the optional extras + a live DB, like the other postgres/sqlite
store tests), but the factory's contract is asserted via a fake saver.
"""

import pytest

from ft import checkpoint
from ft.checkpoint import build_checkpointer


def test_memory_backend_needs_no_setup():
    from langgraph.checkpoint.memory import MemorySaver

    ckpt = build_checkpointer("memory")
    assert isinstance(ckpt, MemorySaver)
    # setup= is accepted and ignored for memory (no tables).
    assert isinstance(build_checkpointer("memory", setup=False), MemorySaver)


def test_postgres_backend_requires_dsn():
    # Either ImportError (extra not installed) or ValueError (dsn missing) — both are documented;
    # neither silently returns a broken checkpointer.
    with pytest.raises((ValueError, ImportError)):
        build_checkpointer("postgres")


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        build_checkpointer("redis-saver")


class _FakeCM:
    """Stands in for ``*Saver.from_conn_string(...)`` (a context manager yielding a saver)."""

    def __init__(self) -> None:
        self.entered = False
        self.setup_calls = 0

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *exc):
        return False

    def setup(self) -> None:
        self.setup_calls += 1


def test_enter_and_setup_runs_setup_by_default():
    cm = _FakeCM()
    saver = checkpoint._enter_and_setup(cm, setup=True)
    assert saver is cm
    assert cm.entered is True  # the context manager was entered to obtain the saver
    assert cm.setup_calls == 1  # tables provisioned once


def test_enter_and_setup_skips_setup_when_false():
    cm = _FakeCM()
    saver = checkpoint._enter_and_setup(cm, setup=False)
    assert saver is cm
    assert cm.entered is True
    assert cm.setup_calls == 0  # caller provisions tables out of band


def test_enter_and_setup_accepts_plain_saver():
    """A plain saver instance (not a context manager) is returned as-is, with setup run."""

    class _PlainSaver:
        def __init__(self) -> None:
            self.setup_calls = 0

        def setup(self) -> None:
            self.setup_calls += 1

    saver = _PlainSaver()
    out = checkpoint._enter_and_setup(saver, setup=True)
    assert out is saver
    assert saver.setup_calls == 1
