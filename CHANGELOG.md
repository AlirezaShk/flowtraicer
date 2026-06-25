# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) once it reaches a public release.

## [Unreleased]

## [0.11.1] - 2026-06-25

### Fixed
- **Durable checkpointers (`sqlite`/`postgres`) are now actually usable.** `build_checkpointer`
  previously returned the un-entered `*Saver.from_conn_string(...)` **context manager** instead of a
  saver, and never created the checkpoint tables. It now enters the context manager to return a
  ready-to-use, process-lived saver and provisions its tables.

### Added
- **One-time checkpointer table provisioning.** `build_checkpointer(backend, *, setup=True, …)` runs
  the saver's idempotent `.setup()` (`CREATE TABLE IF NOT EXISTS`) on first build by default so a
  freshly-pointed empty DB just works; pass `setup=False` to skip DDL when a migration provisions the
  tables. The checkpoint tables share the FT trace DB cleanly (distinct table names).

### Docs
- **Cardless "await-user" pause guaranteed.** `ctx.pause(awaiting=…, payload=None)` (or `payload`
  omitted) yields `turn.interrupt is None` (and the streaming terminal `paused` event carries no
  card), so a node can pause purely to hand the turn back to the user — making the
  whole-session-as-one-engagement re-entrant loop (`qualify → assist → await_user → assist …`) a
  documented, safe shape. New README bullet + runnable `python -m ft.examples.session_loop`.
 
- **`ctx.emit` from inside an `AgentTool.handler`.** Documented that the `ctx` passed to a tool
  handler is the running step's context, so `ctx.emit(payload)` from a handler during `ctx.run_tools`
  surfaces as an `emit` stream event under `stream`/`stream_resume` (and is a no-op under
  `run`/`start`/`resume`). New README bullet in the agentic-step section.
- **Resume-or-start dispatch idiom endorsed.** README blesses try-`stream_resume`/except-
  `ResumeError(reason="no_resumable_engagement")`→`stream` as the supported "resume the session or
  start it" dispatch for an endpoint that doesn't track session start.

### Tests
- `tests/test_cardless_pause.py` (cardless pause + re-entrant loop + per-turn token scoping),
  `tests/test_emit_from_handler.py` (emit from a tool handler; no-op when non-streaming), and
  `tests/test_checkpointer_factory.py` (factory contract + provisioning).

## [0.11.0] - 2026-06-25

### Added
- **Incremental streaming of a turn.** New `Workflow.stream(input, recorder, *, thread_id, …)` and
  `Workflow.stream_resume(thread_id, recorder, *, input=…, …)` — async-generator counterparts of
  `start`/`resume` that **yield `StreamEvent`s as the turn executes** (`step_started`, `token`,
  `emit`, `step_finished`), terminating on the same paused/completed boundary with one terminal
  event whose `ev.turn` is the resulting `WorkflowTurn`. `ctx.llm(prompt, stream=True)` surfaces
  token chunks through the iterator (consuming the client's `astream(...)`) **and** still accounts
  them into the step trace + per-turn usage; `ctx.emit(payload)` pushes a render-ready payload
  mid-turn without pausing. Splices straight into a FastAPI `StreamingResponse` NDJSON contract.
  (Closes NEEDS.md #2; the mid-turn-emit residual of #3.)
- **Multi-tool agentic step.** New `@wf.agent_step(tools=…, max_iterations=…)` + `ctx.run_tools(messages)`
  run a bounded ReAct propose→execute→feed-back loop where the model chooses among **many** tools,
  recording each tool call + LLM round under the one step. The tool contract `ft.agent.AgentTool`
  (`{name, description, parameters, handler}`, `handler(args, ctx)`) is **app-agnostic** — FT
  imports no application types. Adds `ft.llm.ToolRequest` and `LLMResult.tool_calls` for clients
  that support tool-calling. (Closes NEEDS.md #5.)
- **Per-turn token accounting.** `WorkflowTurn.token_usage` (`TokenUsage`) sums tokens for **only
  the steps advanced during this turn** (not the whole multi-turn engagement) — charge a per-turn
  budget off `turn.token_usage.total`. Also carried on the streaming terminal event's `ev.turn`.
  (Closes NEEDS.md #8.)

### Changed
- **`Workflow.resume` now raises `ResumeError`** (a `RuntimeError` subclass with a `.reason` of
  `"no_resumable_engagement"` / `"not_paused"` / `"awaiting_mismatch"`) instead of a bare `KeyError`,
  and accepts an optional `expect_awaiting="…"` guard that asserts the parked label before
  delivering input. Resuming an **already-completed** thread (double-submit / stale replay) raises
  `reason="no_resumable_engagement"`. (Closes NEEDS.md #12.)

### Docs
- Pinned the resume/turn contract that was previously ambiguous, with
  `tests/test_resume_contract.py`: `turn.interrupt` is guaranteed to be the **unwrapped** `payload`
  dict (`MessageForm.model_validate(turn.interrupt)` is safe); `resume(input=None)` makes
  `ctx.pause(...)` return `None` (not `{}`); a single `resume` may **re-pause** at a different node
  (chained pauses — drive an N-card flow as `while turn.is_paused: resume(...)`). README "The resume
  contract" + streaming/agentic-step/per-turn-token sections; design-doc "Resume contract" section.
- New runnable examples `ft.examples.streaming_turn` (streaming + chained pauses + per-turn tokens)
  and `ft.examples.agentic_step` (multi-tool loop).

## [0.10.0] - 2026-06-25

### Added
- **Cross-turn checkpoint / resume (human-in-the-loop).** New `Workflow.start(input, recorder, *,
  thread_id, …)` and `Workflow.resume(thread_id, recorder, *, input=…)` advance a workflow one turn
  at a time, pausing at `ctx.pause(awaiting=…, payload=…)` (which wraps LangGraph's `interrupt()`)
  and resuming the **same** engagement on a later call — even in a fresh process — keyed by a stable
  `thread_id` (your chat session id). Both return a `WorkflowTurn{engagement_id, thread_id, status,
  awaiting, interrupt}` (`status` is `"paused"` or `"completed"`). The engagement spans the pause as
  one continuous journey. `Workflow.run()` is unchanged (run-to-completion, returns `str`).
- `ft.checkpoint.build_checkpointer("memory"|"sqlite"|"postgres", …)` — a pluggable factory that
  **wraps LangGraph's official savers** (the checkpointer holds resumable execution state, separate
  from the FlowTraicer trace/audit store). `Workflow(..., checkpointer=…)` / `start(checkpointer=…)`
  accept any `BaseCheckpointSaver`; the default is an in-process `MemorySaver`.
- `EngagementStatus.PAUSED` and `StepStatus.WAITING` for the parked-for-human-input states, plus the
  `EngagementStatusChanged` record + `Recorder.set_engagement_status(...)` (a non-terminal status
  transition that does not end the engagement).
- Example `ft.examples.hitl_resume` (`python -m ft.examples.hitl_resume`) and design doc
  `docs/2026-06-25-checkpoint-resume-design.md`.

### Docs
- New **"Operational model"** section in the README documenting Recorder/Store lifecycle (build one
  app-scoped singleton), concurrency safety (single-loop safe; multi-thread needs per-thread stores
  or Postgres/Redis per process), the `Recorder` step-map memory note, long-lived `ACTIVE`/`PAUSED`
  engagements never being purged, and **embedding FlowTraicer behind a feature flag** (its store is
  observability-only; the app stays the system of record; align FT threads to sessions via
  `thread_id`/`metadata`). (Closes NEEDS.md #7, #10.)
- New README guidance on **per-step schema extraction through the injected `ctx.llm`** (no second
  provider) — `Extractor.from_provider` is optional sugar. (Closes NEEDS.md #9.)
- CONTRIBUTING corrected: the **import name is `ft`** (was wrongly stated as `FlowTraicer`).

## [0.9.2] - 2026-06-24

### Added
- `ft.server.serve(store, host=…, port=…)` — run the trace viewer in one line; FlowTraicer owns
  the uvicorn server, so you no longer build the app and call `uvicorn.run` yourself. `create_app`
  is still exported for when you want the ASGI app (custom server / auth).

## [0.9.1] - 2026-06-24

### Changed
- `instructor` and `litellm` are now **core dependencies** — `pip install flowtraicer` is
  batteries-included (schema extraction via `ft.extraction` and `ft.llm.LiteLLMClient` work out of
  the box). The `extraction` and `litellm` optional extras are removed. The `openai` / `anthropic`
  / `google` / `providers` extras remain for picking a specific Instructor `from_provider` SDK, and
  `redis` / `postgres` for the store backends.
- README install instructions now use `pip install flowtraicer` (with `[redis]` / `[postgres]` /
  `[openai]` extras), instead of an editable `pip install -e .`.

## [0.9.0] - 2026-06-24

First public release. FlowTraicer maps, visualizes, monitors, debugs, logs, and audits the
steps of an engagement between a user and an agentic AI system.

### Added

**Trace core**
- `Engagement → Step → Event` model with per-step tools, per-step extraction, global-step
  intent switches, and per-step / per-engagement token totals.
- `TraceState` base graph-state declaring the conventional channels (`tool_calls`, `llm_calls`,
  `events`, `extraction`) so nodes never redeclare them.

**Storage**
- Pluggable append-only `Store` with SQLite (default, zero-dependency), Redis Streams, and
  Postgres (JSONB + `LISTEN/NOTIFY`) backends, plus live `subscribe` on all three.

**LangGraph integration**
- Auto-instrumentation: `run_instrumented` records node entry/exit, timing, tools, LLM calls,
  extractions, and intent switches; `read_topology` reflects the compiled graph.
- Drop-off tracking: `goal_nodes` mark journeys that never reached a goal as `ABANDONED` with
  `dropped_at` set to the last step reached.

**Workflow orchestration**
- `Workflow` DSL over LangGraph (declare steps, tools, global steps, goals, edges) with an
  injected per-step context (`ctx.llm`) that records token usage automatically.
- **Build once, run many:** a `Workflow` compiles its graph once (cached); per-run dependencies
  pass through `run(..., llm=…, deps=…)` and surface to nodes via `ctx.llm` / `ctx.deps`
  (LangGraph `configurable` DI). Nodes no longer close over request state, so one workflow
  instance serves every request without rebuilding/recompiling the graph.

**LLM integration (provider-agnostic)**
- `ft.llm.LLMClient` protocol (one async `acomplete` method) is the only contract `ctx.llm`
  requires, so any provider/SDK can be plugged in. `LiteLLMClient` is the bundled, config-driven
  implementation (one config → 100+ providers). See the README's "bring your own provider".
- Instructor-powered per-step schema extraction (`ft.extraction`).
- Global LLM-provider registry: `ft.registry.REGISTER.set_llm_provider(client)` sets one default
  provider every workflow falls back to (resolution order: per-run `llm=` > `Workflow(llm=)` >
  registry). Validates the client satisfies `LLMClient` and raises a descriptive `TypeError`
  otherwise.
- Global recorder for out-of-workflow LLM calls: `REGISTER.set_recorder(recorder)` +
  `REGISTER.record_llm_usage(model, tokens=…, caller=…, metadata=…)` record token usage from
  agent/LLM calls made outside a `Workflow` (chat, voice, extraction). Each becomes a small
  self-contained engagement so it rolls up by model/caller. Fail-open; validated on registration.

**Analytics, retention & audit**
- Cross-engagement analytics: `funnel`, `journeys`, `group_by`.
- Retention: `purge` / `RetentionPolicy` (completed engagements past their window; active ones
  never purged).
- Tamper-evident audit digests (`ft.audit`): fingerprint an engagement and later detect any
  alteration.

**Viewer**
- FastAPI query + live-stream server and a linked Cytoscape.js **graph + timeline** viewer.

> Pre-1.0: the public API may still change between minor versions.
