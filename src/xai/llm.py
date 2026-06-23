"""Config-driven, multi-provider LLM calls via LiteLLM — with token usage for the trace.

xai's core is provider-agnostic: you give it token counts, it records them. This optional
helper closes the loop for free-form LLM turns using `LiteLLM <https://docs.litellm.ai>`_,
which speaks one OpenAI-style interface to 100+ providers. You configure a provider, model,
and key once::

    from xai.llm import LiteLLMClient

    llm = LiteLLMClient(provider="openai", model="gpt-5-nano", api_key="XXX")
    # or, matching a config blob:
    llm = LiteLLMClient.from_config({"llm_provider": "openai", "model": "gpt-5-nano", "key": "XXX"})

    result = llm.complete("Summarize this for the applicant.")
    result.text          # the completion text
    result.tokens.total  # token usage as an xai TokenUsage

and it drops straight into the LangGraph runner's ``llm_calls`` convention::

    return {"messages": [result.text], "llm_calls": [result.as_llm_call()]}

``litellm`` is imported lazily, so this module imports without it installed (tests inject a
completion function). Install it with the ``litellm`` extra.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .core.model import TokenUsage

Messages = str | list[dict]


@dataclass
class LLMResult:
    """The text + token usage of one completion."""

    text: str
    tokens: TokenUsage
    model: str
    raw: Any = None

    def as_llm_call(self, *, name: str | None = None) -> dict:
        """Shape an ``llm_calls`` state entry the LangGraph runner records."""
        return {
            "name": name or self.model,
            "prompt_tokens": self.tokens.prompt,
            "completion_tokens": self.tokens.completion,
            "total_tokens": self.tokens.total,
        }


def _normalize(messages: Messages) -> list[dict]:
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    return messages


class LiteLLMClient:
    """A thin, config-driven wrapper over ``litellm.completion``.

    :param provider: e.g. ``"openai"``, ``"anthropic"``, ``"gemini"`` (omit if ``model``
        already carries it).
    :param model: e.g. ``"gpt-5-nano"``. Combined with ``provider`` into LiteLLM's
        ``"<provider>/<model>"`` form.
    :param api_key: the provider key (forwarded to LiteLLM per call).
    :param settings: default kwargs forwarded to every call (``temperature``, ``max_tokens``…),
        overridable per :meth:`complete` call.
    """

    def __init__(
        self,
        *,
        provider: str | None = None,
        model: str,
        api_key: str | None = None,
        _completion=None,
        **settings,
    ) -> None:
        self.model = f"{provider}/{model}" if provider else model
        self._api_key = api_key
        self._settings = settings
        self._completion = _completion  # injectable for tests

    @classmethod
    def from_config(cls, config: dict, *, _completion=None) -> LiteLLMClient:
        """Build from a config blob like ``{"llm_provider", "model", "key", ...settings}``."""
        config = dict(config)
        provider = config.pop("llm_provider", None) or config.pop("provider", None)
        model = config.pop("model")
        api_key = config.pop("key", None) or config.pop("api_key", None)
        return cls(
            provider=provider, model=model, api_key=api_key, _completion=_completion, **config
        )

    def _complete_fn(self):
        if self._completion is not None:
            return self._completion
        import litellm

        return litellm.completion

    def complete(self, messages: Messages, **overrides) -> LLMResult:
        """Run a completion and return its text + token usage."""
        kwargs = dict(self._settings)
        kwargs.update(overrides)
        if self._api_key is not None:
            kwargs.setdefault("api_key", self._api_key)
        response = self._complete_fn()(model=self.model, messages=_normalize(messages), **kwargs)

        text = response.choices[0].message.content
        usage = getattr(response, "usage", None)
        tokens = TokenUsage(
            prompt=getattr(usage, "prompt_tokens", 0) or 0,
            completion=getattr(usage, "completion_tokens", 0) or 0,
            total=getattr(usage, "total_tokens", 0) or 0,
        )
        return LLMResult(text=text, tokens=tokens, model=self.model, raw=response)
