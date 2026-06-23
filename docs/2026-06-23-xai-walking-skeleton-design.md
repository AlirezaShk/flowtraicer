# xai — Walking-Skeleton Design

> Working name **`xai`** (will be renamed before open-sourcing). An independent Python
> library for mapping, visualizing, monitoring, debugging, logging, and auditing the
> steps of an engagement between a user and an agentic AI system.

Date: 2026-06-23
Status: approved scope (Approach A — walking skeleton, then increments)

## 1. Purpose & approach

Build the **thinnest end-to-end vertical slice** that proves the whole pipe works on one
real LangGraph run, and validates the riskiest assumption: *that LangGraph's execution
stream + compiled-graph topology expose enough to faithfully reconstruct an
`Engagement → Step → Events` tree, including a global-step intent-switch.*

Once the skeleton renders a real engagement, every later capability is additive (see
§7 Roadmap).

### Decided framing (from brainstorming)

- **Hybrid, instrument-first:** a clean, framework-agnostic trace core fed by an
  orchestration/observability layer. v1 is **LangGraph-first** — auto-instrumentation of
  LangGraph runs is the happy path; the generic emit API is the core contract the adapter
  calls into.
- **3-level data model:** `Engagement → Step (workflow node) → Events`, with per-step
  tools, per-step schema extraction, and **global steps** that re-route intent.
- **Storage:** pluggable, append-only, with a sensible default. Skeleton ships
  **SQLite + in-process async pub/sub** only.
- **Schema extraction:** Pydantic schemas; **Instructor** for extraction and **LangGraph**
  as the LLM engine, with a provider abstraction — *the extraction helper itself is a
  later increment;* the skeleton includes the `Extraction` type and records one manually.
- **Visualization:** two linked views (graph + timeline) eventually; the skeleton ships
  **graph view only** (Cytoscape.js), with the executed path highlighted.

## 2. Data model (`core/model.py`, Pydantic v2)

The model is captured **complete** now (even types the skeleton barely exercises) so we
never reshape it later.

### Enums

- `EngagementStatus`: `active | completed | failed`
- `StepStatus`: `running | completed | failed | skipped`
- `EventKind`: `llm_call | tool_call | extraction | log | error`

### Entities

- **`Event`** — `id, step_id, kind, name, ts, duration_ms?, payload: dict, error?`.
  Append-only; the atomic record of something that happened inside a step (an LLM call, a
  tool call, a log line, an error, or an extraction marker for the timeline).
- **`Extraction`** — `schema_name, json_schema: dict, values: dict, confidence?: float,
  valid: bool`. The per-step structured-data result. Stored on its `Step` and also emitted
  as an `extraction` event so it appears on the timeline.
- **`IntentSwitch`** — `id, from_step?, to_step, reason, ts`. First-class record of a
  **global step** re-routing the workflow. Held on the `Engagement`.
- **`Step`** — `id, engagement_id, name, status, started_at, ended_at?, duration_ms?,
  parent_step_id?, tools_available: list[str], is_global: bool, extraction?: Extraction,
  events: list[Event]`. One workflow node execution.
- **`Topology`** — the static graph: `nodes: list[NodeDef]`, `edges: list[EdgeDef]`.
  `NodeDef(name, is_global, tools)`, `EdgeDef(source, target, condition?)`. Read from the
  compiled LangGraph; the executed path is derived by matching `Step.name` against nodes.
- **`Engagement`** — `id, name, status, started_at, ended_at?, metadata: dict,
  topology: Topology, steps: list[Step], intent_switches: list[IntentSwitch]`. The whole
  user↔agent session.

## 3. Append-only records & the store

The store's source of truth is an **append-only log of delta records** (event-sourcing
"lite"). This gives audit integrity + replay now and projections later.

### Record types (`store/records.py`)

`EngagementStarted, StepStarted, EventRecorded, ExtractionRecorded, IntentSwitched,
StepEnded, EngagementEnded` — each carrying its ids, timestamp, and payload.

### `Store` protocol (`store/base.py`)

- `append(record) -> None` — append-only insert.
- `get_engagement(engagement_id) -> Engagement` — fold the record log into the tree.
- `list_engagements() -> list[EngagementSummary]`
- `subscribe() -> AsyncIterator[Record]` — live tail.

### Default backend (`store/sqlite.py`)

One table `records(seq INTEGER PK AUTOINCREMENT, engagement_id TEXT, type TEXT, ts TEXT,
data JSON)`. `append` = insert + publish to an in-process asyncio broadcaster (a set of
per-subscriber `asyncio.Queue`s). `get_engagement` = select-by-engagement ordered by
`seq`, fold deltas into an `Engagement`. Reconstruction (fold) lives in
`store/reconstruct.py` so every backend reuses it.

## 4. Recorder (`recorder.py`)

The **emit contract** the adapter (and any manual instrumentation) calls:

```
start_engagement(name, metadata, topology) -> engagement_id
start_step(engagement_id, name, tools=None, is_global=False, parent=None) -> step_id
record_event(step_id, kind, name, payload, duration_ms=None, error=None)
record_extraction(step_id, extraction)
record_intent_switch(engagement_id, to_step, reason, from_step=None)
end_step(step_id, status)
end_engagement(engagement_id, status)
```

Each call builds a Record and `append`s it to the store. **Fail-open:** every method wraps
its body in try/except, logs on failure, and never raises into the caller —
*instrumentation must not crash the observed agent.*

## 5. LangGraph adapter (`langgraph_adapter/`)

- **`topology.py`** — `read_topology(compiled_graph, global_nodes=set()) -> Topology`
  using LangGraph's `get_graph()` (nodes + edges). Nodes named in `global_nodes` are
  flagged `is_global`.
- **`runner.py`** — `run_instrumented(graph, input, recorder, *, global_nodes=set(),
  config=None)`. Drives the graph via `graph.astream(..., stream_mode="debug")` and maps
  task enter/exit to `start_step`/`end_step`, tool/LLM activity to `record_event`, and a
  transition **into** a global node to `record_intent_switch`. Using the debug stream
  (rather than callbacks) gives reliable node boundaries + state — this is the assumption
  the skeleton validates.

## 6. Server & viewer

- **`server/app.py`** — FastAPI: `GET /engagements`, `GET /engagements/{id}` (full tree +
  topology), `WS /stream` (tail the store's subscribe). Serves the static viewer at `/`.
- **`viewer/`** — static `index.html` + `app.js` + `style.css`. Cytoscape.js renders the
  topology; the executed path is highlighted, global steps are styled distinctly, and
  clicking a node opens a side panel showing its events + extraction as JSON. A `WS /stream`
  tail appends live records (proves the monitoring path).

## 7. Error handling & testing

- **Fail-open instrumentation** everywhere (recorder, adapter): observability never breaks
  the agent. Store/network failures are logged, not raised.
- **No network in tests:** the demo agent uses a fake/echo chat model so the suite runs
  with no API keys (CI-safe).
- **TDD, unit by unit:** `test_model` (construction + fold correctness), `test_store_sqlite`
  (append/get round-trip, subscribe delivery), `test_recorder` (correct records, fail-open),
  `test_topology` (read from a small compiled graph), `test_runner` (run the demo agent,
  assert expected steps/events/intent-switch), `test_server` (endpoint shapes via TestClient).
- **Integration anchor:** `examples/demo_agent.py` — a tiny LangGraph `StateGraph`
  (`greet → qualify → search`, plus a global `escalate` re-route), one fake tool call in
  `search`, one manual `Extraction` in `qualify` — exercises every model type.

## 8. Build sequence

1. Scaffold: `pyproject.toml`, `README.md`, `.gitignore`, `git init`, venv (Python 3.12).
2. `core/model.py` (+ tests) — the data model.
3. `store/records.py`, `store/reconstruct.py`, `store/base.py`, `store/sqlite.py` (+ tests).
4. `recorder.py` (+ tests).
5. `langgraph_adapter/topology.py` + `runner.py` (+ tests).
6. `examples/demo_agent.py` + integration test.
7. `server/app.py` (+ tests) and `viewer/` static files.

## 9. Roadmap (deferred increments — each its own spec)

- Timeline view + graph/timeline linking.
- Instructor-powered extraction helper + provider abstraction (`init_chat_model` across
  OpenAI / Anthropic / Gemini).
- Postgres + Redis Streams store adapters.
- Audit retention + tamper-evidence.
- Orchestration DSL (declare steps/global-steps/per-step tools as sugar over LangGraph).
- Port a real TeranoAI flow (e.g. `house_search_guide`).
- Open-source packaging: `CONTRIBUTING.md`, CI, license, final name.

## 10. Note on repo nesting

The library is initialized as its **own git repo** at `backend/src/lib/xai/`, nested inside
the backend submodule's working tree. This is intentional for later extraction to a
standalone open-source repo. The backend repo will see `xai/` as an embedded repository;
add it to the backend's ignore rules if that becomes noisy.
