"""Tests for the Instructor-powered extraction helper (offline — fake clients)."""

from pydantic import BaseModel

from ft.extraction import ExtractionResult, Extractor
from ft.recorder import Recorder
from ft.store.sqlite import SQLiteStore


class BudgetInfo(BaseModel):
    budget: int
    area: str


class _FakeClient:
    """Stands in for an Instructor-patched client; records the last call."""

    def __init__(self, returns):
        self._returns = returns
        self.last = None

    def create(self, *, response_model, messages, **kwargs):
        self.last = {"response_model": response_model, "messages": messages, "kwargs": kwargs}
        return self._returns


class _FakeAsyncClient(_FakeClient):
    async def create(self, *, response_model, messages, **kwargs):
        self.last = {"response_model": response_model, "messages": messages, "kwargs": kwargs}
        return self._returns


def test_extract_returns_validated_model_and_record():
    fake = _FakeClient(BudgetInfo(budget=95000, area="Shibuya"))
    result = Extractor(fake).extract(BudgetInfo, "A place in Shibuya around 95k")

    assert isinstance(result, ExtractionResult)
    assert result.value.budget == 95000
    assert result.values == {"budget": 95000, "area": "Shibuya"}

    record = result.as_record()
    assert record.schema_name == "BudgetInfo"
    assert record.values["area"] == "Shibuya"
    assert "budget" in record.json_schema["properties"]
    assert record.valid is True

    # The schema was forwarded to the client as response_model.
    assert fake.last["response_model"] is BudgetInfo


def test_string_prompt_is_normalized_to_a_user_message():
    fake = _FakeClient(BudgetInfo(budget=1, area="x"))
    Extractor(fake).extract(BudgetInfo, "hello")
    assert fake.last["messages"] == [{"role": "user", "content": "hello"}]


def test_message_list_is_passed_through():
    fake = _FakeClient(BudgetInfo(budget=1, area="x"))
    msgs = [{"role": "system", "content": "extract"}, {"role": "user", "content": "hi"}]
    Extractor(fake).extract(BudgetInfo, msgs)
    assert fake.last["messages"] == msgs


def test_model_override_is_forwarded_as_kwarg():
    fake = _FakeClient(BudgetInfo(budget=1, area="x"))
    Extractor(fake).extract(BudgetInfo, "hi", model="gpt-4o")
    assert fake.last["kwargs"]["model"] == "gpt-4o"


def test_extract_and_record_writes_to_trace():
    store = SQLiteStore()
    rec = Recorder(store)
    eid = rec.start_engagement("e")
    sid = rec.start_step(eid, "qualify")

    fake = _FakeClient(BudgetInfo(budget=95000, area="Shibuya"))
    result = Extractor(fake).extract_and_record(rec, sid, BudgetInfo, "Shibuya, 95k")
    rec.end_step(sid)
    rec.end_engagement(eid)

    assert result.value.area == "Shibuya"
    eng = store.get_engagement(eid)
    step = eng.steps[0]
    assert step.extraction.schema_name == "BudgetInfo"
    assert step.extraction.values["budget"] == 95000


async def test_aextract_awaits_async_client():
    fake = _FakeAsyncClient(BudgetInfo(budget=120000, area="Meguro"))
    result = await Extractor(fake).aextract(BudgetInfo, "Meguro around 120k")
    assert result.value.budget == 120000


def test_from_provider_wires_instructor(monkeypatch):
    captured = {}
    sentinel = object()

    def fake_from_provider(model, *, async_client=False, **kwargs):
        captured["model"] = model
        captured["async_client"] = async_client
        captured["kwargs"] = kwargs
        return sentinel

    import instructor

    monkeypatch.setattr(instructor, "from_provider", fake_from_provider)
    extractor = Extractor.from_provider("openai/gpt-4o-mini", temperature=0)

    assert extractor._client is sentinel
    assert captured["model"] == "openai/gpt-4o-mini"
    assert captured["async_client"] is False
    assert captured["kwargs"] == {"temperature": 0}
