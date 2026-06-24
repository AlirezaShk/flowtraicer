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
- Instructor-powered per-step schema extraction and a LiteLLM-based, config-driven
  multi-provider LLM client (`xai.llm`).
- Cross-engagement analytics (`funnel`, `journeys`, `group_by`), retention (`purge`,
  `RetentionPolicy`), and tamper-evident audit digests (`xai.audit`).
- FastAPI query + live-stream server and a linked Cytoscape.js **graph + timeline** viewer.

> Pre-1.0 and under active development; the public name is provisional.
