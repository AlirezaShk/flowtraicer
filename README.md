# xai

> **Working name** — will be renamed before open-sourcing.

An independent Python library to **map, visualize, monitor, debug, log, and audit** the
steps of an engagement between a user and an agentic AI system.

`xai` models an engagement as a three-level tree:

```
Engagement  →  Step (workflow node)  →  Events (llm_call / tool_call / extraction / log / error)
```

Steps carry **per-step tools** and a **per-step extraction schema**, and **global steps**
can re-route the whole intent mid-engagement. The library captures all of this from a
[LangGraph](https://github.com/langchain-ai/langgraph) run, stores it in an append-only
trace, and renders it as a linked **graph + timeline** in a browser viewer.

## Status: walking skeleton

This is the first vertical slice (see [`docs/`](docs/)). It proves the full pipe end to
end on one LangGraph run:

- `core/model` — the complete Pydantic data model.
- `store` — append-only trace store (default: **SQLite + in-process pub/sub**).
- `recorder` — the fail-open emit contract.
- `langgraph_adapter` — auto-instruments a LangGraph run into the trace.
- `server` — FastAPI query + live-stream API.
- `viewer` — Cytoscape.js **graph view** with the executed path highlighted.

See the design doc for the deferred roadmap (timeline view, Instructor-powered extraction,
Postgres/Redis adapters, audit retention, orchestration DSL).

## Why it's framework-agnostic at the core

The trace core knows nothing about LangGraph — the adapter feeds it through a small emit
API. LangGraph is the v1 happy path; other engines can be added as adapters.

## Quickstart (dev)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,extraction]"
pytest
python -m xai.server.app          # serve the viewer + API at http://localhost:8000
```

## Design

- [`docs/2026-06-23-xai-walking-skeleton-design.md`](docs/2026-06-23-xai-walking-skeleton-design.md)

## License

Apache-2.0 (intended; final license confirmed at open-source time).
