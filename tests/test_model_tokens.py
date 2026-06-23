"""Tests for token usage, computed token totals, and abandonment on the model."""

from xai.core.model import (
    Engagement,
    EngagementStatus,
    EventKind,
    Step,
    StepEvent,
    TokenUsage,
)


def test_token_usage_total_defaults_to_sum():
    assert TokenUsage(prompt=100, completion=20).total == 120


def test_token_usage_explicit_total_is_kept():
    assert TokenUsage(prompt=100, completion=20, total=130).total == 130


def test_token_usage_total_only_provider():
    # Some providers report only a grand total.
    assert TokenUsage(total=77).total == 77


def test_event_carries_tokens():
    ev = StepEvent(
        step_id="s", kind=EventKind.LLM_CALL, name="gpt", tokens=TokenUsage(prompt=10, completion=5)
    )
    assert ev.tokens.total == 15


def test_step_total_tokens_sums_llm_events():
    step = Step(engagement_id="e", name="apply")
    step.events.append(
        StepEvent(
            step_id=step.id,
            kind=EventKind.LLM_CALL,
            name="a",
            tokens=TokenUsage(prompt=10, completion=5),
        )
    )
    step.events.append(
        StepEvent(
            step_id=step.id,
            kind=EventKind.LLM_CALL,
            name="b",
            tokens=TokenUsage(prompt=3, completion=2),
        )
    )
    step.events.append(StepEvent(step_id=step.id, kind=EventKind.TOOL_CALL, name="t"))  # no tokens
    assert step.total_tokens == 20


def test_engagement_total_tokens_sums_steps_and_serializes():
    eng = Engagement(name="e")
    step = Step(engagement_id=eng.id, name="x")
    step.events.append(
        StepEvent(
            step_id=step.id,
            kind=EventKind.LLM_CALL,
            name="a",
            tokens=TokenUsage(prompt=10, completion=5),
        )
    )
    eng.steps.append(step)

    assert eng.total_tokens == 15
    dumped = eng.model_dump(mode="json")
    assert dumped["total_tokens"] == 15
    assert dumped["steps"][0]["total_tokens"] == 15


def test_abandoned_status_and_dropped_at():
    eng = Engagement(name="e", status=EngagementStatus.ABANDONED, dropped_at="comparison")
    assert eng.status is EngagementStatus.ABANDONED
    assert eng.dropped_at == "comparison"
