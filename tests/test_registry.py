"""Tests for the global LLM-provider registry (ft.registry.REGISTER)."""

from operator import add
from types import SimpleNamespace
from typing import Annotated

import pytest

from ft.core.model import EventKind, TokenUsage
from ft.langgraph_adapter import TraceState
from ft.llm import LiteLLMClient, LLMResult
from ft.orchestration import Workflow
from ft.recorder import Recorder
from ft.registry import REGISTER
from ft.store.sqlite import SQLiteStore


@pytest.fixture(autouse=True)
def _clean_registry():
    REGISTER.reset()
    yield
    REGISTER.reset()


class _GlobalState(TraceState):
    messages: Annotated[list, add]


class _ConformingClient:
    async def acomplete(self, messages, **overrides) -> LLMResult:
        return LLMResult(text="global!", tokens=TokenUsage(prompt=6, completion=3), model="g/m")


def test_default_provider_is_none_until_set():
    assert REGISTER.get_llm_provider() is None


def test_set_and_get_a_conforming_provider():
    client = _ConformingClient()
    returned = REGISTER.set_llm_provider(client)
    assert returned is client
    assert REGISTER.get_llm_provider() is client
    assert REGISTER.llm_provider is client


def test_litellm_client_is_accepted():
    client = LiteLLMClient(model="m")
    REGISTER.set_llm_provider(client)
    assert REGISTER.get_llm_provider() is client


def test_rejects_object_without_acomplete():
    with pytest.raises(TypeError, match="acomplete"):
        REGISTER.set_llm_provider(object())


def test_rejects_sync_acomplete():
    class _SyncClient:
        def acomplete(self, messages, **overrides):  # not async
            return None

    with pytest.raises(TypeError, match="async"):
        REGISTER.set_llm_provider(_SyncClient())


def test_rejects_non_callable_acomplete():
    bad = SimpleNamespace(acomplete="not callable")
    with pytest.raises(TypeError, match="acomplete"):
        REGISTER.set_llm_provider(bad)


def test_reset_clears_provider():
    REGISTER.set_llm_provider(_ConformingClient())
    REGISTER.reset()
    assert REGISTER.get_llm_provider() is None


@pytest.mark.asyncio
async def test_workflow_falls_back_to_registered_global_provider():
    """No per-run / per-workflow llm: the workflow uses the registered global and records tokens."""
    REGISTER.set_llm_provider(_ConformingClient())

    wf = Workflow("uses_global", state_schema=_GlobalState)  # no llm= here

    @wf.step
    async def talk(state, ctx):
        return {"messages": [await ctx.llm("hi")]}  # no llm passed to run() either

    wf.entry("talk")
    wf.finish("talk")

    store = SQLiteStore()
    eid = await wf.run({"messages": []}, Recorder(store))  # <- no llm=

    step = store.get_engagement(eid).steps[0]
    assert step.events[-1].name  # sanity
    llm_events = [e for e in step.events if e.kind is EventKind.LLM_CALL]
    assert len(llm_events) == 1
    assert step.total_tokens == 9  # 6 + 3 from the global provider


@pytest.mark.asyncio
async def test_per_run_llm_overrides_global():
    REGISTER.set_llm_provider(_ConformingClient())  # global = 9 tokens

    class _OtherClient:
        async def acomplete(self, messages, **overrides) -> LLMResult:
            return LLMResult(text="run!", tokens=TokenUsage(prompt=1, completion=1), model="o/m")

    wf = Workflow("override", state_schema=_GlobalState)

    @wf.step
    async def talk(state, ctx):
        return {"messages": [await ctx.llm("hi")]}

    wf.entry("talk")
    wf.finish("talk")

    store = SQLiteStore()
    eid = await wf.run({"messages": []}, Recorder(store), llm=_OtherClient())

    step = store.get_engagement(eid).steps[0]
    assert step.total_tokens == 2  # per-run client wins over the global (not 9)


# --- The global recorder: record token usage for calls made OUTSIDE a workflow ----------


def test_default_recorder_is_none():
    assert REGISTER.get_recorder() is None


def test_set_and_get_recorder():
    rec = Recorder(SQLiteStore())
    assert REGISTER.set_recorder(rec) is rec
    assert REGISTER.get_recorder() is rec


def test_set_recorder_rejects_non_recorder():
    with pytest.raises(TypeError, match="record_llm_call|Recorder"):
        REGISTER.set_recorder(object())


def test_record_llm_usage_is_noop_without_recorder():
    # fail-open: no recorder registered -> returns None, raises nothing
    usage = TokenUsage(prompt=3, completion=2)
    assert REGISTER.record_llm_usage("openai/gpt-5", tokens=usage) is None


def test_record_llm_usage_emits_engagement_with_token_event():
    store = SQLiteStore()
    REGISTER.set_recorder(Recorder(store))

    eid = REGISTER.record_llm_usage(
        "openai/gpt-5-nano",
        tokens=TokenUsage(prompt=10, completion=5),
        caller="instagram_dm.classifier",
        metadata={"session_id": "s1"},
    )
    assert eid is not None

    eng = store.get_engagement(eid)
    assert eng.total_tokens == 15
    assert eng.metadata["model"] == "openai/gpt-5-nano"
    assert eng.metadata["caller"] == "instagram_dm.classifier"
    assert eng.metadata["session_id"] == "s1"
    llm_events = [e for s in eng.steps for e in s.events if e.kind is EventKind.LLM_CALL]
    assert len(llm_events) == 1
    assert llm_events[0].tokens.total == 15
