# FlowTraicer

## From the Human Author

I couldn't find an intuitive package that JUST works when it comes to visualizing, monitoring, auditing, debugging, and orchestrating agentic workflows in a systematic way. So I just decided to build one with Claude. Feel free to use this, and contribute to it if you feel like it!

## Short Introduction

An open source Python library to **map, visualize, monitor, debug, log, and audit** the steps of an engagement between a user and an agentic AI system.

You build your agent as a [LangGraph](https://github.com/langchain-ai/langgraph) graph;
`FlowTraicer` captures each run as a structured, append-only trace and renders it as a linked
**graph + timeline** in a browser.

## The model

An engagement is a three-level tree:

```text
Engagement  →  Step (one workflow node)  →  Events (llm_call / tool_call / extraction / log / error)
```

- **Engagement** — one whole user↔agent session. Carries free-form `metadata` (put your
  `user_id`, `session_id`, etc. here).
- **Step** — one LangGraph node execution. Has a status, timing, the **tools available** to
  it, and an optional **per-step extraction** (a Pydantic schema pulled from the turn).
- **Event** — something that happened inside a step (a tool call, an LLM call, a log line,
  an error, an extraction).
- **Global step** — a node that can fire from anywhere and re-route the workflow's intent
  (e.g. "escalate to a human"). Entering one records an **intent switch**.

## Install

```bash
pip install -e ".[dev]"            # core + test deps
pip install -e ".[extraction,openai]"   # + Instructor extraction with the OpenAI SDK
```

## Getting started: instrument a LangGraph agent

```python
import asyncio
from typing import Annotated, TypedDict
from operator import add

from langgraph.graph import StateGraph, START, END

from ft.store.sqlite import SQLiteStore
from ft.recorder import Recorder
from ft.langgraph_adapter import run_instrumented


# 1. Build your agent as a normal LangGraph graph. Extend TraceState so you don't
#    redeclare FlowTraicer's channels (tool_calls / llm_calls / events / extraction) yourself.
from ft.langgraph_adapter import TraceState

class State(TraceState):
    messages: Annotated[list, add]     # your own domain fields

def greet(state):  return {"messages": ["hi, what are you looking for?"]}
def search(state): return {"messages": ["here are 3 options"],
                           "tool_calls": [{"name": "search_db", "payload": {"hits": 3}}]}

g = StateGraph(State)
g.add_node("greet", greet)
g.add_node("search", search)
g.add_edge(START, "greet")
g.add_edge("greet", "search")
g.add_edge("search", END)
app = g.compile()


# 2. Pick a store. SQLite is the zero-dependency default; pass a path to persist.
store = SQLiteStore("traces.db")
recorder = Recorder(store)

# 3. Run it under instrumentation. Everything the run does is recorded.
engagement_id = asyncio.run(run_instrumented(
    app,
    {"messages": [], "tool_calls": []},
    recorder,
    name="house_search",
    metadata={"user_id": "u-42", "session_id": "s-1"},   # tag the journey
    global_nodes={"escalate"},          # nodes that re-route intent
    node_tools={"search": ["search_db"]},  # tools available per step
))

# 4. Read the trace back.
engagement = store.get_engagement(engagement_id)
for step in engagement.steps:
    print(step.name, step.status, f"{step.duration_ms:.1f}ms",
          [e.name for e in step.events])
```

### Enriching a step from inside a node

`run_instrumented` records node entry/exit, timing, and intent switches automatically. To
capture more, a node may write these **conventional state keys** — the runner records
whatever it finds, so nodes never need a handle to the recorder:

```python
def qualify(state):
    return {
        "messages": ["got it"],
        # tools the node called:
        "tool_calls": [{"name": "lookup_area", "payload": {"area": "Shibuya"}}],
        # LLM calls with token cost (rolls up into step.total_tokens / engagement.total_tokens):
        "llm_calls": [{"name": "gpt-4o-mini", "prompt_tokens": 64, "completion_tokens": 20}],
        # any other typed event (kind = llm_call|tool_call|extraction|log|error):
        "events": [{"kind": "log", "name": "cache_hit", "payload": {"key": "shibuya"}}],
        # a per-step structured extraction:
        "extraction": {"schema_name": "BudgetInfo", "values": {"budget": 95000}},
    }
```

`State` must declare these keys (use `Annotated[list, add]` for the list-valued ones).
Outside a graph you can also call the recorder directly:
`recorder.record_llm_call(step_id, "gpt-4o", prompt=64, completion=20)`.

### Drop-off: goals & abandonment

Pass `goal_nodes` to mark journeys that never reached a goal as **abandoned** (instead of
the indistinguishable `completed`), with `dropped_at` set to the last step reached:

```python
await run_instrumented(app, state, recorder,
                       goal_nodes={"submitted"})   # didn't reach it -> ABANDONED
```

### Free-form LLM calls — bring your own provider

FlowTraicer's core only *records* tokens; **it never imports a specific LLM SDK.** The single
integration point is the `LLMClient` protocol (`ft.llm.LLMClient`): one async method returning the
completion text and its token usage.

```python
from ft.llm import LLMClient   # a runtime_checkable Protocol

class LLMClient(Protocol):
    async def acomplete(self, messages, **overrides) -> LLMResult: ...   # text + token usage
```

When a node calls `await ctx.llm(prompt)`, the workflow calls `acomplete` on whatever you passed as
`llm=` to `Workflow.run(...)`, then records the returned tokens. So you can plug in **any** provider
or SDK — there's no LiteLLM requirement.

**The bundled option** is `LiteLLMClient`, which wraps [LiteLLM](https://docs.litellm.ai) so one
config targets 100+ providers (install the `litellm` extra):

```python
from ft.llm import LiteLLMClient

llm = LiteLLMClient(provider="openai", model="gpt-5-nano", api_key="XXX")
# or from a config blob: LiteLLMClient.from_config({"llm_provider": "openai", "model": "...", "key": "..."})
engagement_id = await workflow.run(state, recorder, llm=llm)
```

#### Adding your own provider / SDK

Implement `acomplete` and return an `LLMResult` (reuse it — `TokenUsage` derives `total` from
`prompt + completion` when you don't set it, and `as_llm_call()` is handled for you):

```python
from ft.core.model import TokenUsage
from ft.llm import LLMResult   # any object with `.text` + `.as_llm_call()` works; LLMResult is the easy path

class MyGeminiClient:
    """Adapt your existing SDK/client to FlowTraicer. `acomplete` is the only method ctx.llm needs."""
    def __init__(self, sdk):
        self._sdk = sdk

    async def acomplete(self, messages, **overrides) -> LLMResult:
        resp = await self._sdk.generate(messages, **overrides)        # however your SDK returns text + usage
        return LLMResult(
            text=resp.text,
            tokens=TokenUsage(prompt=resp.usage.input, completion=resp.usage.output),
            model=resp.model,
        )

# isinstance(MyGeminiClient(sdk), LLMClient) is True (structural check), and:
await workflow.run(state, recorder, llm=MyGeminiClient(sdk))   # tokens flow into every step's trace
```

Notes:
- **Sync SDK?** Wrap the blocking call so you don't stall the event loop:
  `return await anyio.to_thread.run_sync(lambda: self._complete(messages))`.
- **Capture usage.** The whole point of `ctx.llm` is token accounting — read your SDK's usage
  field (e.g. Gemini's `response.usage_metadata`, OpenAI's `response.usage`) into `TokenUsage`.
- **Per-call overrides.** `ctx.llm(prompt, model="…", temperature=0)` forwards `**overrides` to
  `acomplete`; honor what you support and ignore the rest.

#### Setting a global default provider

Passing `llm=` to every `run()` gets repetitive. Register one **global default** instead — every
workflow falls back to it when no per-run / per-workflow client is given:

```python
from ft.llm import LiteLLMClient
from ft.registry import REGISTER

# LiteLLM is the bundled default provider; register a configured instance once at startup:
REGISTER.set_llm_provider(LiteLLMClient(provider="openai", model="gpt-5-nano", api_key=KEY))
# ...or your own client — set_llm_provider VALIDATES it satisfies LLMClient first:
REGISTER.set_llm_provider(MyGeminiClient(sdk))   # TypeError if it has no async acomplete()
```

Resolution order when a node calls `ctx.llm`:

```text
run(llm=…)   >   Workflow(llm=…)   >   REGISTER.get_llm_provider()
```

So the global is the lowest-priority fallback: a per-run `llm=` (e.g. a request-scoped client)
always wins. `REGISTER.set_llm_provider` asserts the client exposes a callable, async
`acomplete(messages, **overrides)` and raises a descriptive `TypeError` otherwise — you can't
register a provider that won't work at run time.

#### Token usage for LLM calls made *outside* a workflow

Not every LLM call runs inside a `Workflow` — chat replies, voice turns, and one-off extractions
often call a model directly. Register a global recorder and push their token usage into FlowTraicer
so it rolls up alongside everything else:

```python
from ft.core.model import TokenUsage
from ft.recorder import Recorder
from ft.registry import REGISTER
from ft.store.postgres import PostgresStore

REGISTER.set_recorder(Recorder(PostgresStore(DSN)))      # validated on registration

# wherever you make a model call (e.g. inside an SDK adapter), after you have the usage:
REGISTER.record_llm_usage(
    "openai/gpt-5-nano",
    tokens=TokenUsage(prompt=resp.usage.prompt_tokens, completion=resp.usage.completion_tokens),
    caller="instagram_dm.classifier",        # shows up in the viewer / group_by
    metadata={"session_id": session_id},
)
```

Each call becomes a small self-contained engagement (one `llm_call` event), so per-model /
per-caller token totals appear in the viewer and `analytics.group_by`. It's **fail-open**: with no
recorder registered it's a no-op, and a recorder error never propagates into the calling agent.

(Structured extraction uses Instructor; see below.)

### Per-step schema extraction (Instructor + Pydantic)

```python
from pydantic import BaseModel
from ft.extraction import Extractor

class BudgetInfo(BaseModel):
    budget: int
    area: str

extractor = Extractor.from_provider("openai/gpt-4o-mini")   # or anthropic/… , google/…
result = extractor.extract(BudgetInfo, "Shibuya, around ¥95,000")

# inside a node — record-via-state (the runner picks it up):
return {"extraction": result.as_record().model_dump()}
# anywhere else — record directly:
extractor.extract_and_record(recorder, step_id, BudgetInfo, "Shibuya, around ¥95,000")
```

## Declarative workflows (the DSL)

`ft.orchestration.Workflow` is sugar over LangGraph: declare steps (with tools), global
steps, goals, and edges once — it compiles the graph and wires per-step tools / global nodes
/ goal nodes into the recorder, so there's no separate bookkeeping to pass to the runner.

```python
from ft.orchestration import Workflow

wf = Workflow("school_journey", state_schema=State, goal_nodes={"submit"})

@wf.step(tools=["search_schools"])
def school_selection(state): ...

@wf.global_step                      # entering it records an intent switch
def escalate(state): ...

wf.entry("intake")
wf.edge("intake", "school_selection")
wf.branch("school_selection", router, {"compare": "comparison", "apply": "consent"})
wf.finish("submit")

engagement_id = await wf.run(initial_state, recorder, metadata={"user_id": "u1"})
```

(The hand-wired `run_instrumented` approach still works; the DSL is optional sugar.)

## Storage backends

The `Store` is pluggable (append-only: write a record, reconstruct an engagement, list
summaries, subscribe to a live tail). Pick by environment — the trace, viewer, and analytics
work identically on all three:

```python
from ft.store.sqlite import SQLiteStore      # default; zero deps, file or :memory:
store = SQLiteStore("traces.db")

from ft.store.redis import RedisStore         # pip install -e ".[redis]"
store = RedisStore("redis://localhost:6379")   # Redis Streams; live tail across processes

from ft.store.postgres import PostgresStore   # pip install -e ".[postgres]"
store = PostgresStore("postgresql://localhost/FlowTraicer")  # durable JSONB + LISTEN/NOTIFY
```

- **SQLite** — local dev, single process, audit-friendly append-only file.
- **Redis** — cross-process live monitoring (recorder and viewer can be separate services).
- **Postgres** — durable + queryable for production, with `LISTEN/NOTIFY` live updates.

## Audit & retention

```python
from datetime import timedelta
from ft.retention import RetentionPolicy, purge_before
from ft.audit import engagement_digest, verify

# retention — drop whole completed engagements past their window (active ones never purged)
purged_ids = RetentionPolicy(max_age=timedelta(days=90)).apply(store, now=...)
# or: purge_before(store, cutoff_datetime)

# tamper-evidence — fingerprint an engagement; later, detect any alteration
digest = engagement_digest(store.get_engagement(eid))   # store this when the engagement ends
verify(store.get_engagement(eid), digest)               # -> False if the trail was altered
```

Anchor the digest outside the trace store (WORM / transparency log / signature) for strong
tamper-evidence — see `FlowTraicer/audit.py` for the threat model.

## Analytics: funnels & journeys

Across many engagements (tag each with `user_id`/`session_id` in `metadata`), answer
*"where do users drop off, and what did each step cost?"*:

```python
from ft.analytics import funnel, journeys, group_by

f = funnel(store, ["intake", "school_selection", "comparison", "application", "submitted"])
for step in f.steps:
    print(step.name, "reached", step.reached, "dropped", step.dropped,
          "conv", step.conversion_rate, "tokens", step.total_tokens,
          "avg_ms", step.avg_duration_ms)

journeys(store, user_id="u-42")     # all engagements for one user (filtered summaries)
group_by(store, "user_id")          # {user_id: [summary, ...]}
store.list_engagements(where={"session_id": "s-1"})   # metadata-filtered index
```

Drop-off is read from each engagement's recorded `dropped_at`, so the funnel stays correct
even when `order` contains an **optional** step (e.g. `comparison`) — `conversion_rate` is
always in `[0, 1]`.

## Viewing traces

The viewer is a FastAPI app + a Cytoscape.js single page (graph on top, timeline below,
linked). To explore the bundled demo:

```bash
python -m ft.server.app        # http://127.0.0.1:8400
```

To view **your own** store, build the app around it:

```python
import uvicorn
from ft.server.app import create_app

uvicorn.run(create_app(SQLiteStore("traces.db")), host="127.0.0.1", port=8400)
```

The API behind the viewer:

| Endpoint | Returns |
|---|---|
| `GET /api/engagements` | one summary row per engagement |
| `GET /api/engagements/{id}` | the full engagement tree (+ topology) |
| `GET /api/engagements/{id}/timeline` | the temporal viewmodel (lanes, marks) |
| `WS  /api/stream` | records pushed live as they're appended |

## Architecture

| Module | Responsibility |
|---|---|
| `ft.core.model` | the Pydantic data model (framework-agnostic) |
| `ft.store` | append-only `Store` protocol + SQLite default backend |
| `ft.recorder` | the fail-open emit contract |
| `ft.langgraph_adapter` | `run_instrumented` + `read_topology` |
| `ft.orchestration` | `Workflow` DSL — declare steps/global-steps/tools/goals over LangGraph |
| `ft.extraction` | Instructor-powered per-step schema extraction |
| `ft.llm` | config-driven multi-provider LLM calls (LiteLLM) |
| `ft.analytics` | cross-engagement funnels, drop-off, journey grouping |
| `ft.retention` / `ft.audit` | purge old engagements; tamper-evident digests |
| `ft.timeline` | temporal viewmodel for the timeline view |
| `ft.server` | FastAPI query + live-stream API and the viewer |

The trace **core knows nothing about LangGraph** — the adapter feeds it through the
recorder's small emit API, so other engines can be added as adapters later.

## Status & roadmap

Done: trace core + SQLite/Redis/Postgres stores, LangGraph auto-instrumentation + the
`Workflow` DSL, Instructor extraction, config-driven LLM (LiteLLM), **per-step token cost**,
**goals/abandonment + drop-off analytics**, retention + tamper-evident audit, and the linked
graph/timeline viewer. Planned: OSS packaging. See [`docs/`](docs/).

## License

Apache-2.0 (intended; final license confirmed at open-source time).
