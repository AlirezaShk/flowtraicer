"""Instructor-powered, provider-agnostic per-step schema extraction.

Declare a Pydantic schema, hand the model some text, get back a validated instance plus
an :class:`Extraction` ready to drop into the trace.

The provider abstraction is Instructor's own :func:`instructor.from_provider` — the same
code targets OpenAI / Anthropic / Gemini by changing only the model string
(``"openai/gpt-4o-mini"``, ``"anthropic/claude-..."``, ``"google/gemini-..."``).

Two ways to land a result in the trace:

* **Record-via-state (LangGraph happy path):** call :meth:`Extractor.extract`, then write
  ``result.as_record()`` into graph state under ``extraction``; the runner records it.
* **Direct-record (manual / non-LangGraph):** call :meth:`Extractor.extract_and_record`
  with a recorder + step id.

``instructor`` is imported lazily inside :meth:`Extractor.from_provider`, so this module
(and a fake-client based test suite) works without any provider SDK installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel

from .core.model import Extraction
from .recorder import Recorder

T = TypeVar("T", bound=BaseModel)

Messages = str | list[dict]


def _normalize_messages(messages: Messages) -> list[dict]:
    """Accept a plain prompt string or a ready list of chat messages."""
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    return messages


@dataclass
class ExtractionResult:
    """A validated extraction: the Pydantic instance plus a trace-ready record."""

    value: BaseModel
    schema: type[BaseModel]

    @property
    def values(self) -> dict:
        """The extracted values as a plain dict."""
        return self.value.model_dump()

    def as_record(self) -> Extraction:
        """Convert to an :class:`~xai.core.model.Extraction` for recording into the trace."""
        return Extraction(
            schema_name=self.schema.__name__,
            json_schema=self.schema.model_json_schema(),
            values=self.value.model_dump(),
            valid=True,
        )


class Extractor:
    """Wraps an Instructor (sync or async) client to extract Pydantic schemas."""

    def __init__(self, client) -> None:
        self._client = client

    @classmethod
    def from_provider(cls, model: str, *, async_client: bool = False, **kwargs) -> Extractor:
        """Build an extractor for ``model`` (e.g. ``"openai/gpt-4o-mini"``)."""
        import instructor

        return cls(instructor.from_provider(model, async_client=async_client, **kwargs))

    def extract(self, schema: type[T], messages: Messages, *, model: str | None = None, **kwargs):
        """Extract ``schema`` from ``messages`` synchronously."""
        if model is not None:
            kwargs["model"] = model
        value = self._client.create(
            response_model=schema, messages=_normalize_messages(messages), **kwargs
        )
        return ExtractionResult(value=value, schema=schema)

    async def aextract(
        self, schema: type[T], messages: Messages, *, model: str | None = None, **kwargs
    ):
        """Extract ``schema`` from ``messages`` using an async client."""
        if model is not None:
            kwargs["model"] = model
        value = await self._client.create(
            response_model=schema, messages=_normalize_messages(messages), **kwargs
        )
        return ExtractionResult(value=value, schema=schema)

    def extract_and_record(
        self,
        recorder: Recorder,
        step_id: str,
        schema: type[T],
        messages: Messages,
        *,
        model: str | None = None,
        **kwargs,
    ) -> ExtractionResult:
        """Extract and immediately record the result against ``step_id``."""
        result = self.extract(schema, messages, model=model, **kwargs)
        recorder.record_extraction(step_id, result.as_record())
        return result

    async def aextract_and_record(
        self,
        recorder: Recorder,
        step_id: str,
        schema: type[T],
        messages: Messages,
        *,
        model: str | None = None,
        **kwargs,
    ) -> ExtractionResult:
        """Async variant of :meth:`extract_and_record`."""
        result = await self.aextract(schema, messages, model=model, **kwargs)
        recorder.record_extraction(step_id, result.as_record())
        return result
