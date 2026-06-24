# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) once it reaches a public release.

## [Unreleased]

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
