# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) once it reaches a public release.

## [Unreleased]

### Added
- Trace core: `Engagement → Step → Event` model with per-step tools, per-step extraction,
  global-step intent switches, and per-step/engagement token totals.
- Pluggable append-only `Store`: SQLite (default), Redis Streams, and Postgres
  (JSONB + `LISTEN/NOTIFY`) backends; live `subscribe` on all three.
- LangGraph auto-instrumentation (`run_instrumented`, `read_topology`) and a `TraceState`
  base graph-state.
- `Workflow` orchestration DSL with an injected per-step LLM context (`ctx.llm`) that records
  token usage automatically.
- **Reusable workflows (build once, run many):** a `Workflow` compiles its graph once (cached);
  per-run dependencies pass through `run(..., llm=…, deps=…)` and surface to nodes via `ctx.llm`
  / `ctx.deps` (LangGraph `configurable` DI). Nodes no longer close over request state, so a
  single workflow instance serves every request without rebuilding/recompiling the graph.
- Instructor-powered per-step schema extraction and a provider-agnostic LLM integration: the
  `ft.llm.LLMClient` protocol (one `acomplete` method) is the only contract `ctx.llm` requires, so
  any provider/SDK can be plugged in; `LiteLLMClient` is the bundled config-driven implementation
  (one config → 100+ providers). See the README's "bring your own provider" section.
- Global LLM-provider registry: `ft.registry.REGISTER.set_llm_provider(client)` sets one default
  provider every workflow falls back to (resolution order: per-run `llm=` > `Workflow(llm=)` >
  registry). It **validates** the client satisfies `LLMClient` (callable, async `acomplete`) and
  raises a descriptive `TypeError` otherwise.
- Global recorder for out-of-workflow LLM calls: `REGISTER.set_recorder(recorder)` +
  `REGISTER.record_llm_usage(model, tokens=…, caller=…, metadata=…)` record token usage from
  agent/LLM calls that don't run inside a `Workflow` (chat, voice, extraction). Each call becomes a
  small self-contained engagement (one `llm_call` event) so it rolls up in the viewer/analytics by
  model/caller. Fail-open (no recorder set → no-op); the recorder is validated on registration.
- Cross-engagement analytics (`funnel`, `journeys`, `group_by`), retention (`purge`,
  `RetentionPolicy`), and tamper-evident audit digests (`ft.audit`).
- FastAPI query + live-stream server and a linked Cytoscape.js **graph + timeline** viewer.

> Pre-1.0 and under active development; the public name is provisional.
