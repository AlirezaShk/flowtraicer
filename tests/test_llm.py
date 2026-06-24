"""Tests for the LiteLLM-backed provider helper (offline — injected completion fn)."""

from operator import add
from types import SimpleNamespace
from typing import Annotated

import pytest

from ft.core.model import EventKind, TokenUsage
from ft.langgraph_adapter import TraceState
from ft.llm import LiteLLMClient, LLMClient, LLMResult
from ft.orchestration import Workflow
from ft.recorder import Recorder
from ft.store.sqlite import SQLiteStore


def _fake_response():
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hello world"))],
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=5, total_tokens=17),
    )


def _fake_completion(**kwargs):
    _fake_completion.last = kwargs
    return _fake_response()


async def _fake_acompletion(**kwargs):
    _fake_acompletion.last = kwargs
    return _fake_response()


@pytest.mark.asyncio
async def test_acomplete_awaits_and_returns_tokens():
    client = LiteLLMClient(provider="openai", model="gpt-5-nano", _acompletion=_fake_acompletion)
    result = await client.acomplete("hi")
    assert isinstance(result, LLMResult)
    assert result.text == "hello world"
    assert result.tokens.total == 17
    assert _fake_acompletion.last["model"] == "openai/gpt-5-nano"


def test_complete_returns_text_and_token_usage():
    client = LiteLLMClient(
        provider="openai", model="gpt-5-nano", api_key="XXX", _completion=_fake_completion
    )
    result = client.complete("hi")

    assert isinstance(result, LLMResult)
    assert result.text == "hello world"
    assert isinstance(result.tokens, TokenUsage)
    assert result.tokens.total == 17
    assert result.model == "openai/gpt-5-nano"


def test_provider_model_and_key_are_forwarded():
    client = LiteLLMClient(
        provider="openai", model="gpt-5-nano", api_key="XXX", _completion=_fake_completion
    )
    client.complete("hi")
    assert _fake_completion.last["model"] == "openai/gpt-5-nano"
    assert _fake_completion.last["api_key"] == "XXX"
    assert _fake_completion.last["messages"] == [{"role": "user", "content": "hi"}]


def test_as_llm_call_matches_runner_convention():
    client = LiteLLMClient(provider="openai", model="gpt-5-nano", _completion=_fake_completion)
    call = client.complete("hi").as_llm_call()
    assert call == {
        "name": "openai/gpt-5-nano",
        "prompt_tokens": 12,
        "completion_tokens": 5,
        "total_tokens": 17,
    }


def test_from_config_matches_user_config_shape():
    client = LiteLLMClient.from_config(
        {"llm_provider": "openai", "model": "gpt-5-nano", "key": "XXX", "temperature": 0},
        _completion=_fake_completion,
    )
    client.complete("hi")
    assert client.model == "openai/gpt-5-nano"
    assert _fake_completion.last["api_key"] == "XXX"
    assert _fake_completion.last["temperature"] == 0


def test_settings_forwarded_and_overridable_per_call():
    client = LiteLLMClient(
        provider="openai", model="m", temperature=0.2, _completion=_fake_completion
    )
    client.complete("hi", max_tokens=100)
    assert _fake_completion.last["temperature"] == 0.2
    assert _fake_completion.last["max_tokens"] == 100


def test_message_list_passed_through():
    client = LiteLLMClient(provider="openai", model="m", _completion=_fake_completion)
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    client.complete(msgs)
    assert _fake_completion.last["messages"] == msgs


# --- The LLMClient protocol: bring your own provider/SDK (no LiteLLM needed) --------------


class _MyState(TraceState):
    messages: Annotated[list, add]


class CustomProviderClient:
    """A hand-rolled client over some other SDK — implements only ``acomplete``."""

    def __init__(self) -> None:
        self.calls: list = []

    async def acomplete(self, messages, **overrides) -> LLMResult:
        self.calls.append(messages)
        return LLMResult(
            text="from my own sdk",
            tokens=TokenUsage(prompt=7, completion=4),  # total derived = 11
            model="my-provider/my-model",
        )


def test_litellm_client_satisfies_the_protocol():
    assert isinstance(LiteLLMClient(model="m"), LLMClient)


def test_custom_client_satisfies_protocol_and_non_client_does_not():
    assert isinstance(CustomProviderClient(), LLMClient)
    assert not isinstance(object(), LLMClient)  # no acomplete -> not a client


@pytest.mark.asyncio
async def test_byo_client_drives_workflow_and_records_tokens():
    """The contract ctx.llm needs is the protocol, not LiteLLM: any conforming client works."""
    wf = Workflow("byo", state_schema=_MyState)

    @wf.step
    async def talk(state, ctx):
        return {"messages": [await ctx.llm("hi")]}

    wf.entry("talk")
    wf.finish("talk")

    client = CustomProviderClient()
    store = SQLiteStore()
    eid = await wf.run({"messages": []}, Recorder(store), llm=client)

    step = store.get_engagement(eid).steps[0]
    llm_events = [e for e in step.events if e.kind is EventKind.LLM_CALL]
    assert len(llm_events) == 1
    assert step.total_tokens == 11  # tokens from the custom client flowed into the trace
    assert client.calls == ["hi"]
