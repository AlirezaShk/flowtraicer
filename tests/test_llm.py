"""Tests for the LiteLLM-backed provider helper (offline — injected completion fn)."""

from types import SimpleNamespace

import pytest

from ft.core.model import TokenUsage
from ft.llm import LiteLLMClient, LLMResult


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
