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
pip install flowtraicer
```

Batteries included: schema extraction (Instructor) and config-driven multi-provider LLM calls
(LiteLLM) ship by default. Optional extras for store backends and specific provider SDKs:

```bash
pip install "flowtraicer[redis]"      # Redis Streams store backend
pip install "flowtraicer[postgres]"   # Postgres store backend
pip install "flowtraicer[openai]"     # an explicit provider SDK for Instructor.from_provider
```

(Requires Python ≥ 3.11. The package imports as `ft`.) Developing FlowTraicer itself? Clone the
repo and `pip install -e ".[dev]"`.

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
config targets 100+ providers (LiteLLM ships with FlowTraicer):

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

#### Extraction through your injected `ctx.llm` (no second provider)

`Extractor.from_provider(...)` stands up its **own** Instructor client (its own SDK, key, and
token budget), separate from the `llm=` client you injected for the run. That is convenient for a
one-off script, but inside a `Workflow` you usually want extraction to go through the **same**
injected `ctx.llm` so it shares the request's model, key, and token accounting.

The supported pattern is: **do the structured call yourself via `ctx.llm`, then record it via the
`extraction` state key** — the `from_provider` extractor is optional sugar, not the only path.
`ctx.llm` records the token usage automatically (it's a normal `ctx.llm` call), and the returned
`extraction` dict is folded onto the step:

```python
import json
from pydantic import BaseModel

class BudgetInfo(BaseModel):
    budget: int
    area: str

@wf.step
async def criteria(state, ctx):
    user_text = state["messages"][-1]
    # One LLM turn through the INJECTED client — tokens roll into this step automatically.
    raw = await ctx.llm(
        "Extract the budget (JPY int) and area as JSON matching "
        f'{json.dumps(BudgetInfo.model_json_schema())}. Text: "{user_text}"'
    )
    values = BudgetInfo.model_validate_json(raw).model_dump()
    return {
        # the extraction is recorded on this step, using the run's model + budget:
        "extraction": {"schema_name": "BudgetInfo", "values": values},
    }
```

Use `Extractor.from_provider(...)` when you *want* a dedicated extraction provider (a cheaper or
more reliable model than the chat model, billed separately); use the `ctx.llm` + `extraction`-key
pattern above when extraction should share the run's injected client, model, and token budget. Both
end up recorded identically as the step's `Extraction` — readers can't tell which path produced it.

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

### Pausing for human input and resuming across turns (human-in-the-loop)

`run()` executes the graph to completion in one call and returns the engagement id. A multi-turn
chat instead advances **one human exchange per HTTP request**: a node emits something (a card),
**stops to wait** for the user's reply, and **resumes** on a *later request* (possibly a different
process) — as the **same** engagement. Use `start` / `resume`, keyed by a stable `thread_id` (your
chat session id), with a node that calls `ctx.pause(...)`:

```python
from ft.orchestration import Workflow

wf = Workflow("qualification_chat", state_schema=State)   # optional: checkpointer=build_checkpointer("postgres", dsn=DSN)

@wf.step
async def qualify(state, ctx):
    card = build_qualification_card(state)               # your render-ready payload
    reply = await ctx.pause(awaiting="qualification_confirm", payload=card)  # ── stops here ──
    # On resume, `reply` is whatever resume(input=...) supplied. Normalise the two channels:
    confirmed = bool(reply.get("confirm")) or reply.get("text", "").lower() in {"yes", "y"}
    return {"messages": [f"qualified={confirmed}"]}

@wf.step
async def answer(state, ctx): ...

wf.entry("qualify"); wf.edge("qualify", "answer"); wf.finish("answer")

# turn 1 (HTTP request 1): runs until the card pauses
turn = await wf.start(initial_state, recorder, thread_id=session_id, llm=llm, deps=deps)
# turn.status == "paused"; turn.awaiting == "qualification_confirm"; turn.interrupt == card
#   -> render turn.interrupt to the user; return the HTTP response.

# turn N (a later HTTP request): the user clicked confirm (or typed "yes")
turn = await wf.resume(thread_id=session_id, recorder=recorder, input={"confirm": True},
                      llm=llm, deps=deps)
# turn.status == "completed"; turn.engagement_id is the SAME engagement as turn 1.
```

Both return a `WorkflowTurn`:

| field | meaning |
|---|---|
| `status` | `"paused"` (render `interrupt`, collect input for the next `resume`) or `"completed"` |
| `engagement_id` | the **one** engagement spanning all turns of this session |
| `thread_id` | the session id you keyed on |
| `awaiting` | the label the paused node is waiting on (`ctx.pause(awaiting=…)`) |
| `interrupt` | the payload the paused node emitted — the **unwrapped** `payload` dict (the card) |
| `token_usage` | `TokenUsage` for **only the steps advanced this turn** (charge your per-turn budget off `turn.token_usage.total`) |

#### The resume contract (double-submit, wrong turn, empty input, chained pauses)

`/api/chat` is concurrent and clients double-submit and replay stale turns, so the edge cases are
nailed down:

- **`turn.interrupt` is the *unwrapped* `payload`.** It is exactly the dict you passed to
  `ctx.pause(payload=…)` — `MessageForm.model_validate(turn.interrupt)` is safe with no unwrapping.
  The `awaiting` label is on `turn.awaiting`, never folded into `interrupt`.
- **`resume()` raises `ResumeError` (a `RuntimeError` subclass) with a `.reason`** when the thread
  can't be resumed as asked, so a router can branch on it:

  ```python
  from ft.orchestration import ResumeError
  try:
      turn = await wf.resume(thread_id=session_id, recorder=recorder, input=reply,
                            expect_awaiting="qualification_confirm")
  except ResumeError as e:
      if e.reason == "no_resumable_engagement":   # already completed / never started (double-submit)
          ...   # re-render the latest persisted message; do NOT re-run the flow
      elif e.reason == "awaiting_mismatch":        # stale client delivering wrong-turn input
          ...   # re-render the card the graph is actually parked on
      # e.reason is also "not_paused" (the engagement exists but isn't parked at a pause)
  ```

  - `reason="no_resumable_engagement"` — no in-flight engagement for this `thread_id` (never
    started, **or already completed/abandoned/failed** — the double-submit / stale-replay case).
  - `reason="not_paused"` — the engagement exists but isn't parked at a pause right now.
  - `reason="awaiting_mismatch"` — the optional `expect_awaiting="…"` guard didn't match the label
    the graph is actually parked on. The check runs *before* any input is delivered, so a stale
    client can't push wrong-turn input through.
- **`resume(input=None)`** (a bare resume) makes `ctx.pause(...)` return **`None`** (not `{}`) —
  guard with `reply or {}` if your node assumes a dict.
- **Chained pauses.** A single `resume` can itself return `status="paused"` again at a *different*
  node. Drive an N-card flow as a loop:

  ```python
  turn = await wf.start(initial_state, recorder, thread_id=session_id, llm=llm)
  while turn.is_paused:                 # each card is one HTTP turn in production
      render(turn.interrupt)            # show the card; collect the reply (next request)
      turn = await wf.resume(thread_id=session_id, recorder=recorder, input=collect_reply())
  # turn.status == "completed"
  ```
- **Cardless pauses (hand the turn back with no card).** `ctx.pause(awaiting=…, payload=None)` (or
  `payload` omitted) is a valid pause that carries **no** card: `turn.interrupt is None` (and the
  streaming terminal `paused` event likewise carries no card). Use it for a node that pauses purely
  to **await the next user message** — emit only your streamed answer + `usage`, no `message_form`:

  ```python
  @wf.step
  async def await_user(state, ctx):
      await ctx.pause(awaiting="user_turn")    # no payload → turn.interrupt is None
      return {}
  ```

  This lets the **whole chat session map to one FT engagement**: after each agentic answer, pause
  to await input and resume back into the agentic step (a re-entrant loop), so the engagement stays
  `PAUSED` between turns and never finishes. (`resume(input=None)` then makes the next
  `ctx.pause(...)` return `None`.) Per-turn `token_usage` stays scoped to each turn's steps. Put the
  streamed answer in the node *after* the cardless pause (so it streams/charges once on each
  re-entry — the replay rule). Runnable: `python -m ft.examples.session_loop`.

Notes:

- **Checkpointer = resumable execution state; trace store = audit log.** They are separate
  backends. `ctx.pause` wraps LangGraph's `interrupt()`; the resume state lives in a checkpointer
  keyed by `thread_id`, while the trace store records the journey keyed by `engagement_id` (joined
  to the thread via `metadata["ft_thread_id"]`). Pick a checkpointer with
  `ft.checkpoint.build_checkpointer("memory"|"sqlite"|"postgres", …)` or pass any LangGraph
  `BaseCheckpointSaver` as `checkpointer=`. The **default is an in-process `MemorySaver`** — fine
  for dev / a single worker (the session must stick to one process); use `"sqlite"`/`"postgres"`
  for durable cross-process resume. `build_checkpointer` **provisions the saver's tables on first
  build** (runs its idempotent `.setup()`; pass `setup=False` to skip if you migrate them yourself);
  the checkpoint tables can share the FT trace DB cleanly (distinct table names) — see
  [`src/ft/checkpoint.py`](src/ft/checkpoint.py).
- **The paused engagement is `PAUSED`, not ended** — it is never purged by retention, and the
  resumed steps record under the same `engagement_id`. The reconstructed trace shows one journey:
  the paused node `WAITING`, then re-run and `COMPLETED`, then the downstream nodes.
- **`ctx.pause` re-runs the node from its top on resume.** LangGraph replays the interrupted node,
  and `interrupt()` returns the resume value the second time — so keep work *before* `ctx.pause`
  idempotent (side effects there will run again on resume).
- A runnable example is `python -m ft.examples.hitl_resume` (single pause) and
  `python -m ft.examples.streaming_turn` (chained pauses + streaming). Full design:
  [`docs/2026-06-25-checkpoint-resume-design.md`](docs/2026-06-25-checkpoint-resume-design.md).

### Streaming a turn incrementally (NDJSON / `StreamingResponse`)

`start`/`resume` return one `WorkflowTurn` at the *end* of the turn. For a chat endpoint that
streams tokens and events to the client as work happens, use the async-generator counterparts
`stream` / `stream_resume` — same arguments, same paused/completed boundary, but they **yield
`StreamEvent`s as the turn executes**:

```python
from ft.orchestration import Workflow

wf = Workflow("streaming_chat", state_schema=State)

@wf.step
async def answer(state, ctx):
    text = await ctx.llm("Summarise the plan.", stream=True)   # tokens stream out as they arrive
    return {"messages": [text]}

# Inside a FastAPI generator → produce the existing NDJSON contract:
async def event_stream():
    async for ev in wf.stream_resume(thread_id=session_id, recorder=recorder, input=reply, llm=llm):
        if ev.kind == "step_started":
            yield ndjson({"type": "status", "label": ev.data["node"]})
        elif ev.kind == "token":
            yield ndjson({"type": "text_chunk", "text": ev.data["text"]})
        elif ev.kind == "emit":                      # a card pushed mid-turn via ctx.emit(...)
            yield ndjson({"type": "message_form", "form": ev.data})
        elif ev.kind in ("paused", "completed"):     # the TERMINAL event carries the WorkflowTurn
            turn = ev.turn
            if turn.is_paused:
                yield ndjson({"type": "message_form", "form": turn.interrupt})   # the card
            yield ndjson({"type": "usage", "total": turn.token_usage.total})     # per-turn tokens

return StreamingResponse(event_stream(), media_type="application/x-ndjson")
```

`StreamEvent.kind` is `step_started` / `token` / `emit` / `step_finished`, terminated by exactly one
`paused` or `completed` event whose `ev.turn` is the resulting `WorkflowTurn`. Details:

| `ev.kind` | `ev.data` | when |
|---|---|---|
| `step_started` / `step_finished` | `{"node": …}` | a node entered / exited |
| `token` | `{"text": …}` | a streamed chunk from `ctx.llm(prompt, stream=True)` |
| `emit` | the payload | a mid-turn `ctx.emit(payload)` (a card that does **not** pause) |
| `paused` / `completed` | `{}` / `{"awaiting": …}` | terminal; `ev.turn` is the `WorkflowTurn` |

- **`ctx.llm(prompt, stream=True)`** consumes the client's `astream(...)` (it must yield text chunks
  then a final `LLMResult` for usage), surfaces each chunk as a `token` event, and **still records
  the tokens** into the step trace and `turn.token_usage` — streaming and non-streaming account
  tokens identically. (A client with no `astream` falls back to a normal `acomplete`.)
- **`ctx.emit(payload)`** pushes a render-ready payload to the iterator *without* pausing (use it
  for a card mid-turn; a card that coincides with a wait should use `ctx.pause(payload=…)`, which
  surfaces on the terminal `turn.interrupt`). Outside a streaming run, `ctx.emit` is a no-op.
- **Replay rule (streaming under pause/resume).** A node re-runs from its top on resume, so its
  `ctx.llm(stream=True)` chunks before a `ctx.pause` will re-stream on the resuming turn. Put the
  streamed answer *after* the last `ctx.pause` in the node (or in a downstream node) so tokens
  stream exactly once. Per-turn token accounting follows the same rule — pre-pause `ctx.llm` work is
  discarded by LangGraph on the pausing turn and recorded only when the node replays to completion,
  so tokens are counted once, on the turn the node actually finishes.

#### Resume-or-start dispatch (when the endpoint doesn't track session start)

An HTTP endpoint often doesn't independently know whether a session already has a paused engagement.
The blessed idiom is **try `stream_resume`, and on
`ResumeError(reason="no_resumable_engagement")` fall back to `stream`** (a fresh start) — no
out-of-band "has this thread started?" lookup against the store:

```python
from ft.orchestration import ResumeError

async def event_stream():
    try:
        agen = wf.stream_resume(thread_id=session_id, recorder=recorder, input=reply, llm=llm)
        async for ev in agen:
            yield to_ndjson(ev)
    except ResumeError as e:
        if e.reason != "no_resumable_engagement":
            raise                              # not_paused / awaiting_mismatch → handle as a stale turn
        # No resumable engagement for this thread → it's a new (or restarted) session: start it.
        async for ev in wf.stream(initial_state, recorder, thread_id=session_id, llm=llm):
            yield to_ndjson(ev)
```

The same try-`resume`/except-`no_resumable_engagement`→`start` shape applies to the non-streaming
`resume`/`start`. (`reason="no_resumable_engagement"` also covers an *already-completed* thread —
the double-submit / stale-replay case — so distinguish a genuine "start over" from a duplicate of a
finished turn with your own per-turn idempotency key if that matters to you.)

### A multi-tool agentic step (the model chooses among many tools in a loop)

For an open-ended phase where the model picks among **many** tools turn after turn (a ReAct loop),
declare an `@wf.agent_step` and bind **executable** tools to it. FT runs the propose→execute→feed
-back loop and records each tool call + LLM round under the one step — no hand-coded fixed service
call per node.

The tool contract (`ft.agent.AgentTool`) is **app-agnostic** — FT imports no app types:

```python
from ft.agent import AgentTool
from ft.orchestration import Workflow

async def search_schools(args, ctx):           # handler(args: dict, ctx) -> JSON-serializable
    svc = ctx.deps["school_service"]            # reach request-scoped services via ctx.deps
    return await svc.search(area=args["area"])

TOOLS = [
    AgentTool(name="search_schools", description="Search language schools by area.",
              parameters={"type": "object", "properties": {"area": {"type": "string"}}},
              handler=search_schools),
    AgentTool(name="compare_schools", description="Compare two schools.",
              parameters={"type": "object", "properties": {"a": {"type": "string"},
                                                            "b": {"type": "string"}}},
              handler=compare_schools),
    # … as many as you like
]

wf = Workflow("school_qa", state_schema=State)

@wf.agent_step(tools=TOOLS, max_iterations=8)
async def qa(state, ctx):
    answer = await ctx.run_tools(state["messages"])   # loops; records each tool_call + llm_call
    return {"messages": [answer]}
```

- **`ctx.run_tools(messages)`** runs the bounded ReAct loop with the run's `ctx.llm` client and the
  step's tools, returning the model's final text. The client must accept a `tools=` kwarg on
  `acomplete` and may return `LLMResult.tool_calls` (a list of `ft.llm.ToolRequest`); FT executes
  each requested tool's `handler(args, ctx)`, records it as a `tool_call` and each model round as an
  `llm_call` **under this step**, feeds the result back, and repeats until the model returns a final
  answer or `max_iterations` rounds elapse. Its tokens roll into `step.total_tokens` and
  `turn.token_usage`.
- **The tool *names*** are also registered as the step's available-tools (for the trace/topology),
  exactly like `@wf.step(tools=[…])`.
- **`ctx.emit(...)` works from inside a tool handler.** The `ctx` passed to `handler(args, ctx)` is
  the **running step's context** — the same one the node body has — so a tool that produces a
  render-ready card can push it to the caller mid-loop via `ctx.emit(payload)`. It surfaces as an
  `emit` `StreamEvent` during `stream` / `stream_resume` (interleaved with the loop's tool calls),
  and is a harmless no-op under non-streaming `run` / `start` / `resume`:

  ```python
  async def compare_schools(args, ctx):
      result = await ctx.deps["school_service"].compare(args["a"], args["b"])
      ctx.emit({"type": "school_comparison", "schools": result})   # mid-loop render card → emit event
      return {"winner": result["winner"]}                          # what the model reads
  ```

- **Replay rule.** Under pause/resume a node re-runs from its top, so a tool with side effects will
  re-execute. Keep tool side effects idempotent, or place the agentic step *after* any `ctx.pause`
  so its loop runs exactly once. A runnable example is `python -m ft.examples.agentic_step`.

## Storage backends

The `Store` is pluggable (append-only: write a record, reconstruct an engagement, list
summaries, subscribe to a live tail). Pick by environment — the trace, viewer, and analytics
work identically on all three:

```python
from ft.store.sqlite import SQLiteStore      # default; zero deps, file or :memory:
store = SQLiteStore("traces.db")

from ft.store.redis import RedisStore         # pip install "flowtraicer[redis]"
store = RedisStore("redis://localhost:6379")   # Redis Streams; live tail across processes

from ft.store.postgres import PostgresStore   # pip install "flowtraicer[postgres]"
store = PostgresStore("postgresql://localhost/FlowTraicer")  # durable JSONB + LISTEN/NOTIFY
```

- **SQLite** — local dev, single process, audit-friendly append-only file.
- **Redis** — cross-process live monitoring (recorder and viewer can be separate services).
- **Postgres** — durable + queryable for production, with `LISTEN/NOTIFY` live updates.

## Operational model (lifecycle, concurrency, embedding)

This section is the intended *operational contract* for running FlowTraicer inside a long-lived,
concurrent service (e.g. a FastAPI app) rather than a one-off script.

### Recorder / Store lifecycle

- **Build one `Recorder(Store)` per process and share it.** The `Workflow`, the `Recorder`, and the
  `Store` are all **build-once, app-scoped singletons** — construct them at startup and reuse them
  across every request. There is no per-request construction; per-request data flows through
  `run(...)` / `start(...)` / `resume(...)` (`input`, `llm`, `deps`, `thread_id`, `metadata`).
- **The store owns a single backing connection.** `SQLiteStore` and `PostgresStore` each hold one
  synchronous connection (`SQLiteStore` opens it with `check_same_thread=False`); `RedisStore`
  holds one client. `append`/`get_engagement`/`list_engagements` are **synchronous and
  non-blocking-ish** (a single local insert/select); they do not `await` mid-operation.
- **Closing.** Call `store.close()` on shutdown. The `Recorder` itself holds no OS resources.

### Concurrency safety under many concurrent `run()`s

- **Single event loop (the normal FastAPI case): safe.** Each `append` is a synchronous
  insert+commit with no `await` inside it, so concurrent `run()`/`start()`/`resume()` coroutines on
  one loop interleave only *between* appends — the store is never re-entered mid-write. The
  `Recorder` keys every event by an explicit `engagement_id`/`step_id`, so concurrent engagements
  never alias each other's records.
- **Multiple OS threads sharing one store: not guaranteed.** The single DB-API connection is not
  meant to be written from multiple threads simultaneously (Python's `sqlite3` is `threadsafety=1`;
  `psycopg` connections are likewise single-owner). If you run several event loops in separate
  threads, give each its own `Store`/`Recorder` (they can point at the same SQLite file / Postgres
  DB), or front the store with your own lock. For production scale-out, use `PostgresStore` (or
  `RedisStore`) so multiple *processes* each hold their own connection to the shared database.
- **Recorder memory note.** The `Recorder` keeps a small in-memory `step_id -> engagement_id` map
  so callers can record events by step alone. It grows with the number of steps started in the
  process lifetime; it is not currently evicted. For a hot, long-lived endpoint this is bounded by
  your traffic — if it matters, run the recorder in a worker you recycle, or use
  `recorder.record_*` overloads that you can scope. (A bounded/evicting map is a tracked follow-up.)

### Long-lived `active` / `paused` engagements

A multi-turn chat engagement stays open across many HTTP requests (see the pause/resume section
below). That is expected and supported:

- An engagement that has not ended stays `ACTIVE` (or `PAUSED` while waiting for human input).
  **Retention never purges a non-completed engagement** (`RetentionPolicy` / `purge_before` only
  drop engagements past their window that have *ended*), so an engagement open for hours/days and
  resumed N times is safe from purge.
- The **resumable execution state** (LangGraph checkpoints) lives in the *checkpointer*, which is
  **separate** from the trace store (see "Pausing and resuming" below). The trace store is the
  audit log; the checkpointer is the resume buffer.

### Embedding FlowTraicer behind a feature flag, alongside an existing loop

FlowTraicer is happy being **one of two engines** behind a flag for the same surface — you can run
the FT `Workflow` for some turns/personas and your legacy loop for others, both writing to the same
chat session.

- **FT's store is observability/audit, not your system of record.** The trace store records *what
  happened* (steps, tokens, extractions, drop-off) for monitoring, debugging, and analytics. Your
  application remains the **system of record** for chat history (`ChatSession` / `ChatMessage`,
  message `form`s, etc.). Do not read chat history *back* out of the FT store to render to users —
  persist your messages in your own tables as you do today, and let FT trace alongside.
- **Align FT threads to your sessions via `thread_id` + `metadata`.** Start/resume an engagement
  keyed by your app's `session_id` (pass it as `thread_id=` to `start`/`resume`, and put
  `session_id`/`user_id` in `metadata`). That is the supported way to make "one app session = one FT
  engagement" so the trace and your session line up 1:1, while the two stores stay independent.
- **Flagging one persona.** Per request, branch on your flag: FT engine vs. legacy loop. Because the
  FT store is independent and fail-open, turning FT on/off for a persona never touches your chat
  persistence or the other personas' behaviour.

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

To view **your own** store, just hand it to `serve` — FlowTraicer runs the server for you:

```python
from ft.server import serve
from ft.store.sqlite import SQLiteStore   # or PostgresStore / RedisStore

serve(SQLiteStore("traces.db"), host="127.0.0.1", port=8400)
```

(Need the app object instead — e.g. to mount behind your own ASGI server or add auth?
`from ft.server import create_app; app = create_app(store)`.)

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
