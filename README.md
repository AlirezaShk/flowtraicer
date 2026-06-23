# xai

> **Working name** — will be renamed before open-sourcing.

An independent Python library to **map, visualize, monitor, debug, log, and audit** the
steps of an engagement between a user and an agentic AI system.

You build your agent as a [LangGraph](https://github.com/langchain-ai/langgraph) graph;
`xai` captures each run as a structured, append-only trace and renders it as a linked
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

(Requires Python ≥ 3.11. The package imports as `xai` regardless of the `src/` layout.)

## Getting started: instrument a LangGraph agent

```python
import asyncio
from typing import Annotated, TypedDict
from operator import add

from langgraph.graph import StateGraph, START, END

from xai.store.sqlite import SQLiteStore
from xai.recorder import Recorder
from xai.langgraph_adapter import run_instrumented


# 1. Build your agent as a normal LangGraph graph.
class State(TypedDict):
    messages: Annotated[list, add]
    tool_calls: Annotated[list, add]   # optional: see "enriching a step" below
    extraction: dict                   # optional

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

### Per-step schema extraction (Instructor + Pydantic)

```python
from pydantic import BaseModel
from xai.extraction import Extractor

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

## Analytics: funnels & journeys

Across many engagements (tag each with `user_id`/`session_id` in `metadata`), answer
*"where do users drop off, and what did each step cost?"*:

```python
from xai.analytics import funnel, journeys, group_by

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
python -m xai.server.app        # http://127.0.0.1:8400
```

To view **your own** store, build the app around it:

```python
import uvicorn
from xai.server.app import create_app

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
| `xai.core.model` | the Pydantic data model (framework-agnostic) |
| `xai.store` | append-only `Store` protocol + SQLite default backend |
| `xai.recorder` | the fail-open emit contract |
| `xai.langgraph_adapter` | `run_instrumented` + `read_topology` |
| `xai.extraction` | Instructor-powered per-step schema extraction |
| `xai.analytics` | cross-engagement funnels, drop-off, journey grouping |
| `xai.timeline` | temporal viewmodel for the timeline view |
| `xai.server` | FastAPI query + live-stream API and the viewer |

The trace **core knows nothing about LangGraph** — the adapter feeds it through the
recorder's small emit API, so other engines can be added as adapters later.

## Status & roadmap

Done: trace core + SQLite store, LangGraph auto-instrumentation, Instructor extraction,
**per-step token cost**, **goals/abandonment + drop-off analytics**, and the linked
graph/timeline viewer. Planned: Postgres/Redis store adapters, audit retention, an
orchestration DSL. See [`docs/`](docs/).

## License

Apache-2.0 (intended; final license confirmed at open-source time).
