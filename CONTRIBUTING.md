# Contributing to FlowTraicer

Thanks for your interest! `FlowTraicer` is a small, focused library — contributions that keep it
that way (clear boundaries, tested, documented) are very welcome.

> **Note:** `FlowTraicer` is a working name and will be renamed before the first public release. The
> package imports as `FlowTraicer` today; if you reference it, expect a rename.

## Development setup

Requires Python ≥ 3.11.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                 # core + test/lint deps (includes fakeredis)
# Instructor extraction and the LiteLLM client ship as core deps — nothing extra needed.
# optional, depending on what you touch:
pip install -e ".[postgres]"            # the Postgres store backend
pip install -e ".[redis]"               # the Redis store backend
```

## Tests

```bash
pytest                                   # the full suite
```

- The **SQLite** and **Redis** store tests run offline (Redis via `fakeredis`).
- The **Postgres** store tests are skipped unless `FT_TEST_PG_DSN` is set:

  ```bash
  docker run -d --name FlowTraicer-pg -e POSTGRES_PASSWORD=FlowTraicer -e POSTGRES_DB=FlowTraicer \
    -p 5432:5432 postgres:16-alpine
  FT_TEST_PG_DSN="postgresql://postgres:FlowTraicer@127.0.0.1:5432/FlowTraicer" pytest
  ```

- No test needs a network or an API key — LLM/provider calls are injected/stubbed.

We use **TDD**: write a failing test first, then the minimal code to pass it. Every new
function/method gets a test; behaviour changes get a test that fails before your change.

## Lint & format

```bash
ruff check .
ruff format .
```

CI runs `ruff check`, `ruff format --check`, and `pytest` (with a Postgres service) on every
push and PR. Keep the build green.

## Design principles

- **The trace core is framework-agnostic.** Only `ft.langgraph_adapter` knows about
  LangGraph; other engines should be added as adapters that call the recorder's emit API.
- **Instrumentation is fail-open** — recording must never crash the observed agent.
- **Small, well-bounded units** with clear interfaces. If a file is doing too much, split it.
- Prefer relative imports inside the package (the name is provisional).

See the [README](README.md) architecture table and `docs/` for the bigger picture.

## Commits & PRs

- Conventional-style messages help (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`).
- One logical change per PR; include tests and doc updates with the code.
- By contributing, you agree your contributions are licensed under the project's
  [Apache-2.0 License](LICENSE).

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). Be kind.
