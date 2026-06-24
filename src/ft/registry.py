"""Process-global FlowTraicer settings — currently the default LLM provider.

FlowTraicer is provider-agnostic (see :class:`ft.llm.LLMClient`): the normal path is to pass an
LLM client per run, ``Workflow.run(llm=...)``. For convenience you can also register **one global
default provider** that every workflow falls back to when no per-run / per-workflow client is given.

LiteLLM (:class:`ft.llm.LiteLLMClient`) is the bundled, default provider — but it needs a model/key,
so nothing is auto-instantiated. Register one explicitly, or swap in your own conforming client::

    from ft.llm import LiteLLMClient
    from ft.registry import REGISTER

    REGISTER.set_llm_provider(LiteLLMClient(provider="openai", model="gpt-5-nano", api_key=KEY))
    # ...or bring your own SDK wrapper (must satisfy the LLMClient protocol):
    REGISTER.set_llm_provider(MyGeminiClient(sdk))

``set_llm_provider`` **validates** the client before accepting it: it must expose a callable,
*async* ``acomplete(messages, **overrides)`` method (the one thing ``ctx.llm`` calls), otherwise a
:class:`TypeError` is raised naming what's missing. Resolution order when a node calls ``ctx.llm``::

    run(llm=...)   >   Workflow(llm=...)   >   REGISTER.get_llm_provider()
"""

from __future__ import annotations

import inspect
from typing import Any

from .core.model import TokenUsage
from .llm import LLMClient

#: The recorder methods :meth:`_Registry.record_llm_usage` relies on (duck-typed Recorder).
_RECORDER_METHODS = (
    "start_engagement", "start_step", "record_llm_call", "end_step", "end_engagement",
)


def _validate_provider(client: Any) -> None:
    """Assert ``client`` satisfies :class:`ft.llm.LLMClient`; raise ``TypeError`` if not."""
    acomplete = getattr(client, "acomplete", None)
    if acomplete is None or not callable(acomplete):
        raise TypeError(
            f"{client!r} is not a valid FlowTraicer LLM provider: it must define a callable "
            "`acomplete(messages, **overrides)` method (the ft.llm.LLMClient protocol)."
        )
    if not inspect.iscoroutinefunction(inspect.unwrap(acomplete)):
        raise TypeError(
            f"{client!r}.acomplete must be an async (coroutine) method; wrap a sync SDK with "
            "`anyio.to_thread.run_sync` inside acomplete so ctx.llm can await it."
        )


def _validate_recorder(recorder: Any) -> None:
    """Assert ``recorder`` is a :class:`ft.recorder.Recorder`-like object; raise ``TypeError``."""
    missing = [m for m in _RECORDER_METHODS if not callable(getattr(recorder, m, None))]
    if missing:
        raise TypeError(
            f"{recorder!r} is not a valid FlowTraicer recorder: missing method(s) {missing}. "
            "Pass an ft.recorder.Recorder (over any Store)."
        )


class _Registry:
    """Process-global FlowTraicer settings. Use the module singleton :data:`REGISTER`."""

    def __init__(self) -> None:
        self._llm_provider: LLMClient | None = None
        self._recorder: Any = None

    def set_llm_provider(self, client: LLMClient) -> LLMClient:
        """Register the global default LLM client (after validating it); return it for chaining."""
        _validate_provider(client)
        self._llm_provider = client
        return client

    def get_llm_provider(self) -> LLMClient | None:
        """The registered global default LLM client, or ``None`` if none has been set."""
        return self._llm_provider

    @property
    def llm_provider(self) -> LLMClient | None:
        return self._llm_provider

    # -- global recorder: token usage for LLM calls made outside a workflow ----------------

    def set_recorder(self, recorder: Any) -> Any:
        """Register a global FlowTraicer :class:`~ft.recorder.Recorder` (validated); return it.

        It's the sink for :meth:`record_llm_usage` — token usage from agent/LLM calls that don't
        run inside a :class:`~ft.orchestration.Workflow` (chat, voice, extraction, …).
        """
        _validate_recorder(recorder)
        self._recorder = recorder
        return recorder

    def get_recorder(self) -> Any:
        """The registered global recorder, or ``None`` if none has been set."""
        return self._recorder

    def record_llm_usage(
        self,
        model: str,
        *,
        tokens: TokenUsage,
        caller: str | None = None,
        metadata: dict | None = None,
        duration_ms: float | None = None,
    ) -> str | None:
        """Record one out-of-workflow LLM call's token usage into FlowTraicer.

        Emits a small, self-contained engagement (one step + one ``llm_call`` event carrying
        ``tokens``) so standalone agent calls show up in the viewer and roll up in analytics —
        group by ``metadata['model']`` / ``metadata['caller']``. **Fail-open**: returns ``None``
        and records nothing if no recorder is registered (or on any recorder error). Returns the
        engagement id otherwise.
        """
        recorder = self._recorder
        if recorder is None:
            return None
        try:
            meta = {**(metadata or {}), "model": model, "caller": caller, "standalone": True}
            engagement_id = recorder.start_engagement(caller or "llm", metadata=meta)
            step_id = recorder.start_step(engagement_id, "completion")
            recorder.record_llm_call(
                step_id,
                model,
                prompt=tokens.prompt,
                completion=tokens.completion,
                total=tokens.total,
                duration_ms=duration_ms,
                model=model,
            )
            recorder.end_step(step_id)
            recorder.end_engagement(engagement_id)
            return engagement_id
        except Exception:  # observability must never crash the observed agent
            return None

    def reset(self) -> None:
        """Clear registered globals — mainly for tests/isolation."""
        self._llm_provider = None
        self._recorder = None


#: The process-global FlowTraicer registry singleton.
REGISTER = _Registry()
