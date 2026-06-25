# FlowTraicer — Capability & Documentation Needs Ledger

This file is the boundary artifact for the two-agent clean-room dogfooding model:

- **Agent A (Application Developer)** consumes FlowTraicer as a **black box** — only its
  `README.md`, `docs/`, `CHANGELOG.md`, `CONTRIBUTING.md`, and the usage samples under
  `real_estate/workflows/`. When the documented surface is missing, unclear, or doesn't
  support a real need, A appends an entry below instead of reading `src/ft/**`.
- **Agent B (FT Library Maintainer)** works each open entry — implementing the capability
  and/or fixing the documentation, then filling in the **Resolution**.

Every entry is a real need surfaced by building a real use-case (the language-school chat
flow). This ledger is the durable record of gaps found and closed.

## How to file an entry

Append under "Open needs" using this template:

```
### N. <short title>
- **Filed by:** A — <round/date>
- **Kind:** capability gap | doc gap | unclear contract | bug
- **Context:** what I was trying to build and why.
- **What the docs say (or don't):** quote/point to README/docs/sample, or note the absence.
- **Need / proposed API:** the smallest thing that would unblock me (signature sketch OK).
- **Acceptance:** how I'll know it's resolved (a usage snippet I want to be able to write).
---
- **Resolution (B):** what changed — code + docs/sample pointers + CHANGELOG entry.
- **Status:** open | in-progress | resolved
```

## Triage (Round 2, B)

Categorization of A's Round-1 needs, mapped to the roadmap sub-projects (SP1 checkpoint/resume,
SP2 HITL emit/ingest, SP3 streaming + tool-loop nodes, SP4 app cutover, or pure-doc).

| # | Title (short) | Kind | Category | Sub-project | Round-2 disposition |
|---|---|---|---|---|---|
| 1 | Pause/resume across HTTP turns | capability gap | **capability build** | **SP1** (keystone) | **Built (R2)**; **A-verified PASS (R3)** from black box — minor doc gaps spun out as #11–#13. |
| 2 | Stream a run's output incrementally | capability gap | capability build | SP3 | **Built (R4)** — `Workflow.stream`/`stream_resume` async-gens yielding `StreamEvent` + `ctx.llm(stream=True)`/`ctx.emit`. |
| 3 | Emit a render-ready payload (`emit`) to caller | capability gap | capability build | SP2/SP3 | **Resolved** — pausing card via `turn.interrupt` (R2/#1); mid-turn `ctx.emit` lands with SP3 streaming (R4). |
| 4 | Ingest the human's reply on resume | unclear contract | capability build | SP2 | Partially unblocked by SP1: `resume(input=…)` is the carrier (see #1 resolution); the `ctx.pause()`→reply ergonomics + dual-channel normalisation land in SP2. |
| 5 | Free-form multi-tool agentic step | capability gap | capability build | SP3 | **Built (R4)** — `@wf.agent_step(tools=AgentTool[...])` + `ctx.run_tools(...)`; app-agnostic tool contract. |
| 6 | Intent-trigger into a global step + deterministic guard | capability gap | capability build (+doc) | SP2/SP4 | Deferred; partly a doc clarification of `global_step`/`branch` semantics, to be resolved with the SP4 flow. |
| 7 | Recorder/Store lifecycle + concurrency safety | unclear contract | **doc-fix-only** | pure-doc | **Resolved this round** (README "Operational model"). |
| 8 | Per-turn token accounting on the handle | unclear contract | capability build | SP1/SP3 | **Built (R4)** — `WorkflowTurn.token_usage` scoped to the steps advanced this turn (also on streaming terminal `ev.turn`). |
| 9 | Extraction through the injected `ctx.llm` | unclear contract | **doc-fix-only** | pure-doc | **Resolved this round** (README "Per-step schema extraction"). |
| 10 | Feature-flag FT vs. legacy loop; store = observability | doc gap | **doc-fix-only** | pure-doc | **Resolved this round** (README "Embedding FlowTraicer behind a flag"). |

Plus a cross-cutting **doc bug** (not an A need): CONTRIBUTING said the package "imports as
`FlowTraicer`" while the README/samples import `ft`. Fixed CONTRIBUTING — `ft` is authoritative.

## Open needs

### 1. Pause a run for human input and resume it on a later HTTP request
- **Filed by:** A — Round 1 / 2026-06-25
- **Kind:** capability gap
- **Context:** The whole feature is a multi-turn, human-in-the-loop chat. Each user message
  is a *separate HTTP request*. The flow must: run the `qualification` step, *emit a card*,
  then **stop and wait** for the user's next message (which arrives minutes later in a new
  request to `/api/chat`), then **resume from where it paused** with that input folded into
  state. The documented `Workflow.run(state, recorder, ...)` runs the graph **to completion**
  in one call and returns a finished `engagement_id` (str). Every sample
  (`school_workflow.py`, `apartment_workflow.py`) runs intake→…→submit in a single `run()`;
  none pauses for a real user reply. There is no documented `interrupt`, `resume`,
  `checkpoint`, or `thread_id` concept, even though LangGraph (which FT wraps) has
  `interrupt()` + a checkpointer. Without this I cannot map "one HTTP turn = advance the flow
  by one human exchange"; I'd have to run a fresh whole workflow every turn (losing the FT
  engagement/step continuity that is the entire point) or reach into LangGraph directly
  (breaking the black box).
- **What the docs say (or don't):** README "Declarative workflows" + CHANGELOG 0.9.0
  ("Build once, run many: per-run dependencies pass through `run(...)`") describe only
  start→finish runs. No mention of suspending/resuming, durable thread state, or how a second
  `run()` could *continue* a prior engagement rather than start a new one.
- **Need / proposed API:** a first-class pause/resume on `Workflow`, e.g. a step can signal a
  pause and the runner returns a suspended handle keyed by a caller-supplied thread id; a later
  call resumes with the human's input:
  ```python
  # turn 1 — runs until a step pauses for input
  turn = await wf.start(initial_state, recorder, llm=llm, deps=deps,
                        thread_id=session_id, metadata={...})
  # turn.status in {"paused", "completed"}; turn.awaiting -> what input the paused step wants
  # turn N — resume the SAME engagement with the user's reply
  turn = await wf.resume(thread_id=session_id, recorder, llm=llm, deps=deps,
                        input={"confirm": True})   # or input={"text": "yes"}
  ```
  A node pauses via something like `await ctx.pause(awaiting="qualification_confirm", payload=form_dict)`
  (analogous to LangGraph `interrupt`), and FT records the pause as part of the same engagement.
- **Acceptance:** I can run a step, get a `paused` turn back with the data it emitted,
  return that to the HTTP client, and on the next HTTP request call `wf.resume(thread_id=…,
  input=…)` so the SAME FT engagement continues from the paused node — its steps/events/timeline
  show one continuous journey across N requests, not N disconnected engagements.
---
- **Resolution (B):** **Built (sub-project 1).** Added a first-class checkpointed pause/resume,
  keeping `run()` exactly as-is for backward compat.
  - **API (new):**
    - `Workflow.start(input, recorder, *, thread_id, name=None, metadata=None, config=None,
      llm=None, deps=None, checkpointer=None) -> WorkflowTurn`
    - `Workflow.resume(thread_id, recorder, *, input=None, config=None, llm=None, deps=None,
      checkpointer=None) -> WorkflowTurn` (raises `KeyError` if no resumable engagement for the
      thread)
    - `StepContext.pause(*, awaiting: str, payload: dict | None = None) -> Any` — emits `payload`,
      suspends; returns the `resume(input=…)` value on the next turn.
    - `WorkflowTurn{engagement_id, thread_id, status ("paused"|"completed"), awaiting, interrupt}`
      with `.is_paused`.
    - `Workflow(..., checkpointer=…)` ctor arg; `ft.checkpoint.build_checkpointer("memory"|"sqlite"
      |"postgres", …)` factory wrapping LangGraph's official savers (default in-process
      `MemorySaver`).
    - New `EngagementStatus.PAUSED`, `StepStatus.WAITING`, `EngagementStatusChanged` record, and
      `Recorder.set_engagement_status(...)`.
  - **Behaviour:** `start` opens **one** engagement tagged `metadata["ft_thread_id"]=thread_id` and
    runs until a node pauses or completes; `resume` finds that same engagement by thread and records
    the resumed steps under it (no second `EngagementStarted`). The trace shows one journey
    (paused node `WAITING`, then re-run `COMPLETED`, then downstream). Checkpointer (resumable
    execution state, keyed by `thread_id`) is deliberately separate from the trace store (audit log,
    keyed by `engagement_id`). Verified across a **fresh `Workflow` instance** sharing only the
    checkpointer + store (the separate-process case).
  - **Acceptance met:** `start` returns a `paused` turn carrying the emitted card + `awaiting`;
    `resume(thread_id=…, input=…)` continues the SAME engagement to `completed`.
  - **Code:** `src/ft/checkpoint.py` (new), `src/ft/orchestration.py` (`start`/`resume`/`ctx.pause`/
    `WorkflowTurn`/dual compile), `src/ft/langgraph_adapter/runner.py` (`run_instrumented_turn` +
    `TurnResult`, shared `_record_step_writes`), `src/ft/core/model.py` (`PAUSED`/`WAITING`),
    `src/ft/store/records.py` + `reconstruct.py` (`EngagementStatusChanged`), `src/ft/recorder.py`
    (`set_engagement_status`).
  - **Tests:** `tests/test_checkpoint_resume.py` (5 tests, TDD red→green): pause-returns-turn,
    resume-completes-one-engagement, resume-delivers-input, fresh-process round-trip, run()-still-
    works. Full suite green (`105 passed, 5 skipped`).
  - **Docs/sample:** README "Pausing for human input and resuming across turns" + the `WorkflowTurn`
    table; CHANGELOG `0.10.0`; design doc `docs/2026-06-25-checkpoint-resume-design.md`; runnable
    sample `python -m ft.examples.hitl_resume`.
- **Verification (A — Round 3 / 2026-06-25): PASS — RESOLVED from the black-box standpoint.**
  Working only from the README "Pausing for human input…" section, the `WorkflowTurn` table, the
  design doc, and `ft.examples.hitl_resume` (no `src/ft/**` read), I can build the
  qualification-card turn. The three things I most needed are all documented:
  - **How `resume(input=…)` surfaces in the node:** it is the return value of `ctx.pause(...)`
    (README + design §"How it integrates with LangGraph" step 5 + example lines 40–42). Clear.
  - **Reading the emitted card off the turn:** `turn.interrupt` is the `payload` dict, `turn.awaiting`
    is the label (`WorkflowTurn` table + example line 62). Clear.
  - **Node-replay-on-resume idempotency:** documented ("`ctx.pause` re-runs the node from its top on
    resume … keep work *before* `ctx.pause` idempotent"). Safe for my turn because the card is a pure
    function of state, so replay is benign.
  Residual gaps found while building (filed as #11–#13; none block #1's keystone claim, each just
  forced a small guess): #11 `turn.interrupt` wrapped-vs-unwrapped ambiguity; #12 `resume()` contract
  on already-completed / stale double-submit threads; #13 whether `resume` can itself return
  `status="paused"` again at a different node.
- **Status:** resolved

### 2. Stream a run's output incrementally as it executes (NDJSON turn)
- **Filed by:** A — Round 1 / 2026-06-25
- **Kind:** capability gap
- **Context:** `/api/chat` returns `StreamingResponse(..., media_type="application/x-ndjson")`:
  the current loop streams `status`, `text_chunk`, tool `event`, `message_form`, `usage`, and
  `session_title` lines as work happens (see `chat_turn.py::stream_chat_turn`). To replace that
  loop with FT, I need FT to surface a step's output *as it is produced* — token chunks from
  `ctx.llm`, the qualification card the step emits, and per-tool events — not only after the
  whole `run()` resolves to an `engagement_id`. `Workflow.run` is documented as returning a
  single `str` at the end; there is no documented streaming iterator over a run.
- **What the docs say (or don't):** README shows `engagement_id = await wf.run(...)` (awaits the
  whole run). The viewer has a `WS /api/stream` that tails *the store's records* live, but that
  is for an out-of-band dashboard — it is not a per-request token/event stream I can splice into
  an HTTP `StreamingResponse` for the end user, and it carries trace records, not my NDJSON
  contract (`text_chunk`, `message_form`, …).
- **Need / proposed API:** an async-generator run mode that yields the step outputs as they
  occur, so the router can translate them to NDJSON lines:
  ```python
  async for ev in wf.stream(state, recorder, llm=llm, deps=deps, thread_id=session_id):
      # ev.kind in {"token", "step_started", "step_finished", "emit", "paused", "completed"}
      # ev.data carries text chunks / the emitted card / tool payloads
      yield to_ndjson(ev)
  ```
  Ideally `ctx.llm(prompt, stream=True)` yields chunks that surface through this iterator while
  still being token-accounted into the step trace.
- **Acceptance:** I can write `async for ev in wf.stream(...)` inside a FastAPI generator and
  produce the exact NDJSON the frontend already consumes (`text_chunk`, `message_form`, `status`,
  `usage`), with tokens still rolled into `step.total_tokens`.
---
- **Refinement (A — Round 3 / 2026-06-25): STILL OPEN. SP3. Sharpened against the new turn shape.**
  `start`/`resume` return one `WorkflowTurn` at the *end* of a turn. I need an **incremental**
  counterpart that yields *as the turn executes* and **terminates on the same `paused`/`completed`
  boundary** (B's note: "streaming should extend the checkpointed path"). Proposed async-gen API
  mirroring `start`/`resume`:
  ```python
  async for ev in wf.stream_start(input, recorder, *, thread_id, llm=…, deps=…, metadata=…):
      ...   # ev.kind in {"token","step_started","step_finished","emit","paused","completed"}
  async for ev in wf.stream_resume(thread_id, recorder, *, input=…, llm=…, deps=…):
      ...
  # the FINAL ev is terminal: ev.kind=="paused" (ev.turn carries .awaiting/.interrupt) or
  # ev.kind=="completed" (ev.turn carries .token_usage, see #8).
  ```
  Token chunks must come from `ctx.llm(prompt, stream=True)` and still roll into `step.total_tokens`.
  - **Acceptance (now exact):** inside a FastAPI generator,
    `async for ev in wf.stream_resume(thread_id=session_id, input=reply, …): yield to_ndjson(ev)`
    produces the existing contract — `status` (on `step_started`), `text_chunk` (on `token`),
    `message_form` (on `emit`/terminal `paused` carrying the card), `usage` (from `ev.turn.token_usage`)
    — without awaiting the whole turn before the first byte.
---
- **Resolution (B):** **Built (SP3).** Added async-generator `Workflow.stream(input, recorder, *,
  thread_id, …)` and `Workflow.stream_resume(thread_id, recorder, *, input=…, expect_awaiting=…, …)`
  that mirror `start`/`resume` exactly but **yield `ft.orchestration.StreamEvent`s as the turn
  executes**: `kind in {step_started, token, emit, step_finished, paused, completed}`, terminating on
  the **same paused/completed boundary** with one terminal event whose `ev.turn` is the resulting
  `WorkflowTurn` (`.awaiting`/`.interrupt` on pause; `.token_usage` for usage). Implementation
  consumes LangGraph's combined `astream(stream_mode=["custom","debug"])`: the *debug* stream gives
  node entry/exit + pause detection (re-using the `_read_interrupt` boundary logic), the *custom*
  stream carries `ctx.llm(prompt, stream=True)` token chunks and `ctx.emit(payload)` cards via
  LangGraph's `get_stream_writer()`. `ctx.llm(stream=True)` consumes the client's `astream(...)`
  (yield text chunks, then a final `LLMResult` for usage), surfaces each chunk as a `token` event,
  **and still records the tokens** into the step trace + `turn.token_usage` (streaming and
  non-streaming account identically). **Acceptance met** — the README "Streaming a turn" section shows
  the exact NDJSON mapping inside a FastAPI generator; verified by `tests/test_streaming.py` (tokens
  reconstruct the answer, terminal `completed`/`paused` carries the turn, `stream_resume` continues
  the same engagement, mid-turn `ctx.emit`). **Replay rule documented:** put streamed `ctx.llm` work
  *after* the last `ctx.pause` (or downstream) so tokens stream/charge exactly once.
  - **Code:** `src/ft/langgraph_adapter/runner.py` (`StreamEvent`, `stream_instrumented_turn`),
    `src/ft/orchestration.py` (`Workflow.stream`/`stream_resume`/`StreamEvent`, `ctx.llm(stream=)`,
    `ctx.emit`, `_resume_command`/`_stream_event` shared helpers).
  - **Example:** `python -m ft.examples.streaming_turn`. CHANGELOG `0.11.0`.
- **Status:** resolved

### 3. Emit a structured, render-ready payload (the qualification card / MessageForm) from a step to the caller
- **Filed by:** A — Round 1 / 2026-06-25
- **Kind:** capability gap
- **Context:** The first step must emit a **financial-qualification card**: a
  `MessageForm(kind="qualification", body=<estimate>, fields=[confirm: boolean])` that the
  frontend renders and persists on the chat message. The chat tools today return this as a
  stream `event` (`{"type": "message_form", "form": form.model_dump()}`) consumed by the
  router. In FT, a node returns conventional state keys — `messages`, `tool_calls`,
  `events`, `extraction`. `events`/`tool_calls`/`extraction` are explicitly described as **trace
  records** (they "roll up into step.total_tokens", appear "on the timeline", etc.) — i.e.
  observability data, not a channel for handing a render-payload back to the HTTP caller. There
  is no documented way for a step to say "this dict is the user-facing card for this turn; give
  it to the caller." I can stuff the form into `messages` or `events`, but the docs don't promise
  those are delivered to the application as turn output (vs. only persisted to the trace store).
- **What the docs say (or don't):** README "Enriching a step from inside a node" documents
  `messages` as a domain field and `events`/`tool_calls`/`extraction` as recorder inputs. Nothing
  states how a step's *user-facing* output (a card to render this turn) is returned to the caller
  distinctly from trace events.
- **Need / proposed API:** a documented "turn output" channel distinct from trace events — e.g.
  a reserved state key `emit` (list) or `ctx.emit(payload)` whose items are delivered to the
  caller via the stream/resume handle (need #2/#1) AND optionally mirrored into the trace:
  ```python
  return {"messages": [text], "emit": [{"type": "message_form", "form": form.model_dump()}]}
  # surfaces as an `emit` stream event AND is visible on turn.emitted
  ```
- **Acceptance:** a step returns the qualification-card dict and I can read it off the turn
  handle (`turn.emitted`) / stream (`ev.kind == "emit"`) to write the `message_form` NDJSON line,
  without abusing the trace `events` channel or parsing it back out of the store.
---
- **Refinement (A — Round 3 / 2026-06-25): RESOLVED-BY-#1 for the pausing turn; small residual for
  mid-turn emits.** The `ctx.pause(payload=…)` → `turn.interrupt` channel IS the documented
  "turn output distinct from trace events" I asked for. The card I emit is exactly `turn.interrupt`,
  and I persist it on my own `ChatMessage.form` (FT store stays observability-only per README
  "Operational model"). Proof snippet (black-box, builds today):
  ```python
  turn = await wf.start(initial_state, recorder, thread_id=session_id, llm=llm, deps=deps)
  if turn.is_paused and turn.awaiting == "qualification_confirm":
      form = MessageForm.model_validate(turn.interrupt)          # the card, verbatim
      chat_controller.add_message(session_id, role="assistant", content=…, form=form.model_dump())
      yield ndjson({"type": "message_form", "form": form.model_dump()})
  ```
  - **Downgrade:** the dedicated `emit` channel I originally asked for is **unnecessary for a card
    that coincides with a pause** — `pause(payload=…)` already carries it. Resolved by #1.
  - **Residual (still open, rolls into #2 streaming):** an emit that does NOT pause (e.g. stream a
    card mid-turn and keep going) has no documented carrier today — `turn.interrupt` only exists when
    the turn pauses. For my SP2 flow every card coincides with a wait, so this is **not blocking**;
    file it as the `ctx.emit(...)`/`ev.kind=="emit"` follow-up that lands with #2's `stream_*`.
---
- **Resolution (B):** The pausing-card channel is `ctx.pause(payload=…)`→`turn.interrupt`
  (resolved-by-#1). The residual **emit-without-pause** is now also closed by SP3: `ctx.emit(payload)`
  pushes a render-ready payload to the streaming iterator as an `emit` event mid-turn (no pause), and
  is a no-op outside a streaming run. See README "Streaming a turn" (the `emit` row) and
  `tests/test_streaming.py::test_stream_event_carries_emit_when_node_emits_midturn`. CHANGELOG `0.11.0`.
- **Status:** resolved

### 4. A step that ingests the human's reply for this turn (typed input AND/OR a clicked form field)
- **Filed by:** A — Round 1 / 2026-06-25
- **Kind:** unclear contract
- **Context:** After the card, the confirmation arrives on either channel and must be handled
  identically: (a) the user clicks the boolean `confirm` field (carried on the chat message as
  `MessageSchema.form`), or (b) the user types "yes"/"no" as ordinary chat. The resuming step
  needs *this turn's human input* (the clicked field value or the raw text) to decide yes/no. The
  samples seed everything up front in `initial_state` (`criteria`, `applicant_fields`, …) and
  never read a per-turn user message inside a node — `messages` is only ever appended to, never
  consumed as "the latest user turn." It's unclear how, in an FT run, a node reads "the input the
  human just supplied on this resume."
- **What the docs say (or don't):** `TraceState` declares `messages: Annotated[list, add]`
  (append-only). No doc shows a node reading the newest inbound user message or a resume input.
  With pause/resume (#1) the resume input would be the natural carrier, but its shape and how a
  node reads it is undocumented.
- **Need / proposed API:** define that `wf.resume(input=…)` makes the input available to the
  resumed node (e.g. `ctx.input` or as the return value of the `ctx.pause(...)` call), with a
  documented shape, so one branch handles both `{"confirm": true}` (form click) and
  `{"text": "yes"}` (typed) by normalising to a bool:
  ```python
  reply = await ctx.pause(awaiting="qualification_confirm", payload=form)
  confirmed = reply.get("confirm") if "confirm" in reply else parse_yes_no(reply.get("text",""))
  ```
- **Acceptance:** one resumed node reads the human's reply uniformly whether it came as a clicked
  boolean field or as typed "yes"/"no", and routes to record-qualification vs keep-clarifying.
---
- **Refinement (A — Round 3 / 2026-06-25): RESOLVED-BY-#1.** `resume(input=…)` is delivered as the
  **return value of `ctx.pause(...)`** (documented contract + shown in the example). The dual-channel
  normalization is **the app's responsibility and lives in the router**, not in FT — which is the
  right boundary: the router maps both inbound shapes to one dict before calling `resume`, and the
  node normalizes to a bool. `hitl_resume.py` even demonstrates the exact normalization I need:
  ```python
  # ROUTER (before resume): collapse the two inbound channels into one resume input
  if request.form and "confirm" in request.form:        # clicked boolean MessageFormField
      reply_input = {"confirm": bool(request.form["confirm"])}
  else:                                                  # typed "yes"/"no" chat message
      reply_input = {"text": request.message}
  turn = await wf.resume(thread_id=session_id, recorder=recorder, input=reply_input, llm=llm, deps=deps)

  # NODE (resumed): one branch handles both (verbatim from ft.examples.hitl_resume)
  reply = await ctx.pause(awaiting="qualification_confirm", payload=card)
  confirmed = bool(reply.get("confirm")) or reply.get("text", "").lower() in {"yes", "y"}
  ```
  - **Downgrade:** no new FT capability is needed — the resume-input carrier (#1) plus app-side
    normalization fully satisfies this. Resolved by #1.
  - **Doc note (minor):** the README normalization one-liner assumes `reply` is always a dict; one
    sentence that `resume(input=None)` (a bare resume with no payload) surfaces as `None` (not `{}`)
    would save a `None`-guard guess. Folded into doc-gap #12.
---
- **Resolution (B):**
- **Status:** resolved-by-#1

### 5. Free-form, model-chooses-among-many-tools step (the post-qualification Q&A phase)
- **Filed by:** A — Round 1 / 2026-06-25
- **Kind:** capability gap
- **Context:** After qualifying, the flow is open-ended: "budget → search schools → discuss
  area/commute → time-to-N2 → answer questions," where the model must pick among many tools
  (`search_schools`, `compare_schools`, `get_school_document_requirements`,
  location/property/commute tools, `explain_school_costs`, …) turn after turn — exactly today's
  agentic tool loop (`agent.generate_with_tools_stream`). FT's `@wf.step(tools=[...])` only
  *declares which tool names are available to a step for the trace*; the node body still calls
  services by hand and `ctx.llm(prompt)` returns plain text. There is no documented
  "let the LLM choose & invoke tools this step, loop until it produces a final answer, and record
  each tool_call automatically." The samples hard-code exactly one service call per node.
- **What the docs say (or don't):** README: `node_tools={"search": ["search_db"]}` and
  `@wf.step(tools=["search_schools"])` declare *available* tools; `ctx.llm` is a single
  text-completion call. Nothing offers a tool-calling agent loop or a way to bind executable
  tool implementations (vs. just names) to a step.
- **Need / proposed API:** either (a) a documented agent-step that runs a bounded tool-calling
  loop and auto-records each `tool_call`/`llm_call`:
  ```python
  @wf.agent_step(tools=TOOL_IMPLS, max_iterations=8)
  async def qa(state, ctx):
      return await ctx.run_tools(state["messages"])   # loops, records tool_calls, returns final text
  ```
  or (b) explicit guidance that I keep my existing tool loop and only push token/tool usage into
  FT via `REGISTER.record_llm_usage` / `recorder.record_event` — with a documented pattern for
  doing that *inside* an FT step so it nests under the right `step_id`.
- **Acceptance:** I can express the open-ended Q&A phase as one FT step where the model chooses
  among N tools across several internal iterations, and the trace shows each tool_call + llm_call
  under that step — without hand-coding a single fixed service call per node.
---
- **Refinement (A — Round 3 / 2026-06-25): STILL OPEN. SP3.** Restated as a precise capability with a
  signature, now that the turn shape is settled. The post-qualification Q&A is exactly today's
  `agent.generate_with_tools_stream` loop: the model picks among ~12 tools (`search_schools`,
  `compare_schools`, location/property/commute tools, …) and loops until it emits a final answer.
  Two acceptable shapes (either unblocks me):
  ```python
  # (a) a first-class agent step that binds EXECUTABLE tools and runs a bounded loop:
  @wf.agent_step(tools=TOOL_IMPLS, max_iterations=8)   # TOOL_IMPLS: name -> async callable
  async def qa(state, ctx):
      return await ctx.run_tools(state["messages"])    # loops; auto-records each tool_call+llm_call
  # (b) OR documented guidance to keep my existing loop and nest its events under the step:
  @wf.step(tools=[...names...])
  async def qa(state, ctx):
      # call my agent loop; for each tool/LLM call, record under THIS step:
      ctx.record_tool_call(name, payload); ctx.record_llm_usage(model, tokens)   # ← need ctx handles
      return {"messages": [final_text]}
  ```
  Today `@wf.step(tools=[...])` only declares *available names* for the trace and `ctx.llm` is a
  single text completion — there is no documented way to (i) bind tool *implementations* to a step
  and loop, or (ii) record an arbitrary tool_call/llm_call **nested under the current step from
  inside the node via `ctx`** (the README only shows recording via returned state keys or the
  out-of-workflow `REGISTER.record_llm_usage`, which is a *separate* engagement — wrong nesting).
  - **Acceptance (now exact):** one FT step expresses the open-ended Q&A; the reconstructed trace
    shows N `tool_call` + `llm_call` events nested under that single step's `step_id`, across the
    loop's internal iterations, without one fixed service call per node.
  - **Smallest unblock if (a) is too big:** just document/expose `ctx.record_tool_call(...)` /
    `ctx.record_llm_usage(...)` that nest under the running step — then I keep my loop and (a) is sugar.
---
- **Resolution (B):** **Built (SP3) — shape (a).** Added `@wf.agent_step(tools=IMPLS,
  max_iterations=8)` + `ctx.run_tools(messages)` which run the bounded ReAct loop and **auto-record
  each `tool_call` + `llm_call` nested under the one running step**. The tool contract is the new
  **app-agnostic** `ft.agent.AgentTool` — `{name, description, parameters (json schema),
  handler}` where `handler(args: dict, ctx) -> JSON-serializable` (sync or async); handlers reach
  request-scoped services via `ctx.deps`, so FT imports **no** app types. Tool-calling rides on the
  existing `LLMClient`: `acomplete` gains a `tools=` kwarg and `LLMResult` gains
  `tool_calls: list[ToolRequest]` (`ft.llm.ToolRequest{name,args,id}`); a client that supports
  function-calling parses its provider response into that, and `run_tools` does propose→execute→feed
  -back until a final text answer or `max_iterations`. The tool *names* also register as the step's
  available-tools for the trace/topology. **Acceptance met** — `tests/test_agentic_step.py` shows N
  `tool_call` + `llm_call` events nested under one `qa` step across loop iterations, `max_iterations`
  bounding, and the loop's tokens rolling into `turn.token_usage`.
  - **Code:** `src/ft/agent.py` (new — `AgentTool`), `src/ft/llm.py` (`ToolRequest`,
    `LLMResult.tool_calls`), `src/ft/orchestration.py` (`Workflow.agent_step`, `ctx.run_tools`,
    `ctx._drain_tools`). **Example:** `python -m ft.examples.agentic_step`. CHANGELOG `0.11.0`.
  - **SP4 note (tool-registry adaptation):** the app's existing `lib/agents/llm/tools/base.py`
    `Tool`/`ToolRegistry` (with `args_schema`, `requires_auth`, `dispatch`) maps onto `AgentTool` by
    wrapping each app `Tool` as `AgentTool(name=t.name, description=t.description,
    parameters=t.json_schema(), handler=lambda args, ctx: registry.dispatch(t.name, args,
    build_tool_context(ctx.deps)))`. The app keeps its auth-gating/audit (`ToolRegistry.dispatch`)
    inside the handler; FT only orchestrates + traces. (Details in the SP4 findings.)
- **Status:** resolved

### 6. Intent detection as the entry trigger (route INTO the flow only on school intent) + a deterministic search gate
- **Filed by:** A — Round 1 / 2026-06-25
- **Kind:** capability gap
- **Context:** Two control requirements: (1) "the MOMENT the user shows intent to find/study at
  a Japanese language school, the FIRST step is the qualification card" — i.e. the flow should
  *enter* the qualification path only when intent is detected, otherwise behave as normal Q&A;
  and (2) a deterministic backstop: an authenticated, not-yet-qualified user must be **blocked**
  from school search until they qualify. FT has `@wf.global_step` ("a node that can fire from
  anywhere and re-route the workflow's intent") which sounds right for (1), but the docs never
  show *how* a global step is triggered — entering one "records an intent switch," yet there's no
  documented predicate/condition that *causes* entry from an arbitrary step. For (2) I need a
  guard that can hard-block the search step (return a deterministic refusal) based on
  `deps`/state, independent of the LLM.
- **What the docs say (or don't):** README/CHANGELOG describe `global_step` only as "re-route
  intent / records an intent switch" and `escalate` as the lone example; `wf.branch(node,
  router, {...})` does conditional routing but only *out of a named source node*. No documented
  trigger that evaluates on every turn to jump INTO a global step, and no documented "guard"
  pattern.
- **Need / proposed API:** a documented trigger for global steps (a predicate evaluated each
  turn) and/or confirmation that a router branch can target a global step:
  ```python
  @wf.global_step(trigger=lambda state, ctx: detect_school_intent(state["messages"]))
  async def qualification(state, ctx): ...
  ```
  plus a documented place to run a deterministic guard before a step’s body (raise/short-circuit
  to a refusal when `not ctx.deps["service"].is_qualified(user)`).
- **Acceptance:** with no school intent the run stays in free Q&A; the first turn that shows
  school intent routes into `qualification` (recording an intent switch); and `search` is
  deterministically refused for an authenticated, unqualified user regardless of what the model
  tries.
---
- **Resolution (B):** **Deferred — not required for SP4.** Product chose **persona-as-intent**: the
  upstream persona resolver (selecting `visa_school_guide`) is the intent gate, so the flow's first
  step is the qualification card with no FT global-step trigger needed; the deterministic search
  backstop lives in the app's `SearchSchoolsTool` (audited layer), not in FT. The `global_step`
  trigger-predicate / guard ergonomics remain a tracked FT enhancement for a future flow that wants
  finer in-persona intent gating. Left open/deferred.
- **Status:** deferred (not required for SP4)

### 7. One reusable Recorder/Store per process — lifecycle, and is `run()` safe under concurrent requests?
- **Filed by:** A — Round 1 / 2026-06-25
- **Kind:** unclear contract
- **Context:** `/api/chat` is a hot, concurrent FastAPI endpoint. The samples build
  `recorder = Recorder(PostgresStore(DSN))` in a one-off script and the README says the
  `Workflow` is a build-once singleton reused across requests. But it's unclear (a) whether one
  `Recorder`/`Store` instance is meant to be a long-lived app singleton shared across all
  requests (and is it concurrency-safe / connection-pooled for Postgres?), and (b) whether the
  single `WORKFLOW` singleton's `run()`/`stream()` is safe to call concurrently for many users at
  once given pause/resume state keyed by `thread_id` (#1). If each engagement spans many HTTP
  turns, the store also holds long-lived `active` engagements — I need to know that's expected and
  won't be purged (retention says active ones aren't) or leaked.
- **What the docs say (or don't):** README shows single-script construction; "Build once, run
  many" addresses graph compilation but not Recorder/Store lifetime or concurrency, and not
  long-lived active engagements across requests.
- **Need / proposed API:** documentation (not necessarily code) stating: the intended
  process-lifetime ownership of `Recorder`/`Store`, their thread/async-safety under concurrent
  `run()`/`stream()`, connection pooling for `PostgresStore`, and the expected lifecycle of an
  engagement that stays `active` across many requests (open for hours, resumed N times).
- **Acceptance:** I can confidently construct one app-scoped `Recorder(PostgresStore(DSN))` at
  startup, share it across all concurrent chat requests, and know whether `thread_id`-keyed
  pause/resume is safe under concurrency.
---
- **Resolution (B):** Doc-only (the code already provides what's documented; no contract bug found).
  Added a new **"Operational model (lifecycle, concurrency, embedding)"** section to `README.md`
  (after "Storage backends") documenting: (a) build one app-scoped `Recorder(Store)`/`Workflow`
  singleton and share it; (b) the store owns a single synchronous connection, so concurrent
  `run()/start()/resume()` on **one event loop are safe** (appends don't `await` mid-write and every
  record is keyed by explicit ids), while **multiple OS threads sharing one connection are not**
  guaranteed (use Postgres/Redis per-process, or a per-thread store) — Python `sqlite3` is
  `threadsafety=1`, `psycopg` connections are single-owner; (c) the `Recorder`'s in-memory
  `step_id→engagement_id` map grows over process lifetime (a bounded/evicting variant is a tracked
  follow-up); (d) long-lived `ACTIVE`/`PAUSED` engagements are expected and **never purged by
  retention**, and resumable execution state lives in the *checkpointer*, separate from the trace
  store. CHANGELOG `[Unreleased]` notes the docs add.
- **Status:** resolved

### 8. Token-accounting parity: ctx.llm vs. the app's GeminiTextAgent, and feeding usage into the budget meter
- **Filed by:** A — Round 1 / 2026-06-25
- **Kind:** unclear contract
- **Context:** Today the chat persona runs on `GeminiTextAgent` (a Gemini client), and the turn
  charges a token budget (`record_token_usage_sync(usage_plan, turn_tokens)`) streamed back as a
  `usage` NDJSON line. If FT drives the turn via `ctx.llm`, I need (a) to plug Gemini in as the
  `LLMClient` (the README's "bring your own provider" — fine), and (b) to read *this turn's*
  token total back out of FT so I can charge the budget and emit the `usage` line. The docs show
  tokens rolling up into `step.total_tokens` / `engagement.total_tokens`, but those are read from
  the *store* after the fact (`store.get_engagement(eid)`), which is awkward mid-stream and
  per-engagement (engagement spans many turns) rather than per-turn.
- **What the docs say (or don't):** README documents per-step/per-engagement totals via the
  recorder and a separate out-of-workflow `REGISTER.record_llm_usage`. No documented "tokens used
  by *this run/turn*" returned from `run()`/`stream()` for live budgeting.
- **Need / proposed API:** expose this-turn token usage on the turn/stream handle, e.g.
  `turn.token_usage` (prompt/completion/total for just the steps advanced this resume), so I can
  charge the budget and emit `usage` without re-reading the whole engagement from the store.
- **Acceptance:** after advancing one turn I can read the tokens that turn consumed off the
  returned handle and pass them straight to `record_token_usage_sync`.
---
- **Refinement (A — Round 3 / 2026-06-25): STILL OPEN. SP3 — but now there IS a carrier.** B's #1
  resolution explicitly notes `WorkflowTurn` is the natural place for per-turn usage and that
  `turn.token_usage` lands alongside SP3 streaming. The need is now a single precise field, scoped
  to *this turn only* (the steps advanced by this `start`/`resume` call), NOT the whole engagement:
  ```python
  class WorkflowTurn(BaseModel):
      ...
      token_usage: TokenUsage   # prompt/completion/total for the steps advanced THIS turn only
  ```
  - **Acceptance (now exact):**
    ```python
    turn = await wf.resume(thread_id=session_id, input=reply, llm=llm, deps=deps)
    usage = record_token_usage_sync(usage_plan, turn.token_usage.total)
    yield ndjson({"type": "usage", **usage})
    ```
    i.e. I read this-turn tokens straight off the returned handle (and off the streaming terminal
    `ev.turn`, #2), without re-reading the whole engagement from the store and without summing across
    the many turns the engagement spans. Per-engagement totals stay in the store as today.
  - **Note:** must be per-*turn*, since one engagement = one whole multi-turn chat session;
    `engagement.total_tokens` (store) is the wrong granularity for charging a single exchange.
---
- **Resolution (B):** **Built (SP3).** Added `WorkflowTurn.token_usage: TokenUsage`
  (prompt/completion/total) scoped to **only the steps advanced during this `start`/`resume`/`stream`
  turn**. The runner already records per-step tokens; `_record_step_writes` now *returns* the tokens
  it recorded (from `llm_calls` + token-carrying `events`) and the turn loop sums them onto
  `TurnResult.token_usage`, which `_turn` surfaces on `WorkflowTurn`. Also carried on the streaming
  terminal `ev.turn`. **Acceptance met** —
  `usage = record_token_usage_sync(plan, turn.token_usage.total)` reads this-turn tokens straight off
  the handle; per-engagement totals stay in the store (`engagement.total_tokens`). Verified by
  `tests/test_turn_tokens.py`: per-turn sum is scoped to this turn (engagement total spans all turns
  at a different granularity), and is 0 for a tokenless turn.
  - **Replay subtlety (documented):** a node's `ctx.llm` calls *before* a `ctx.pause` are discarded
    by LangGraph on the pausing turn (the partial node result is dropped), so they are accounted
    **once**, on the resuming turn when the node replays to completion — never double-charged.
  - **Code:** `src/ft/langgraph_adapter/runner.py` (`_record_step_writes` returns `TokenUsage`,
    `TurnResult.token_usage`, per-turn accumulation in `run_instrumented_turn` +
    `stream_instrumented_turn`), `src/ft/orchestration.py` (`WorkflowTurn.token_usage`). CHANGELOG `0.11.0`.
- **Status:** resolved

### 9. Per-step extraction with a provided model/parser (not the default Instructor provider)
- **Filed by:** A — Round 1 / 2026-06-25
- **Kind:** unclear contract
- **Context:** The qualification estimate is *deterministic* (built by
  `school_qualification_service.build_cost_estimate`), and the sample records it as
  `{"extraction": {"schema_name": "FinancialQualification", "values": estimate.model_dump()}}`.
  That's clear for a hand-built dict. But the search/criteria steps want to *extract* structured
  criteria from free user text (budget/area), and `Extractor.from_provider("openai/gpt-4o-mini")`
  hardcodes a provider + spins its own client — separate from my injected `ctx.llm` Gemini client
  and outside the request's token budget. It's unclear how to do schema extraction *through the
  same injected LLM client* so it's counted and uses the configured model, rather than a second
  provider.
- **What the docs say (or don't):** README "Per-step schema extraction" only shows
  `Extractor.from_provider(...)` (its own SDK) or recording a pre-built dict via state. No
  documented path to extract using the run's `ctx.llm` / injected client.
- **Need / proposed API:** either confirm "for non-deterministic extraction, call your own
  extractor and record via the `extraction` state key" is the intended pattern (and the
  `from_provider` extractor is optional sugar), or provide `ctx.extract(Schema, text)` that uses
  the injected client and records automatically.
- **Acceptance:** I can extract `RoomFilterSchema`/`SchoolSearch` from user text using the same
  injected (Gemini) client/budget and have it recorded as the step's extraction, without standing
  up a second Instructor provider.
---
- **Resolution (B):** Doc-only — the supported path already exists; it just wasn't written down.
  Added **"Extraction through your injected `ctx.llm` (no second provider)"** under README's
  "Per-step schema extraction": do the structured call yourself via `await ctx.llm(...)` (which
  records token usage automatically on that step, using the run's injected client/model/budget),
  validate into your Pydantic schema, and record it via the `{"extraction": {"schema_name", "values"}}`
  state key. `Extractor.from_provider(...)` is documented as *optional sugar* for when you
  deliberately want a dedicated extraction provider billed separately; both paths record an
  identical `Extraction`. CHANGELOG `[Unreleased]` notes the docs add.
- **Status:** resolved

### 10. Feature-flag two engines side by side for one persona (FT-driven visa_school_guide vs. the prompt loop)
- **Filed by:** A — Round 1 / 2026-06-25
- **Kind:** doc gap
- **Context:** The feature must ship behind a flag: only `visa_school_guide` runs on FT; every
  other persona (and `visa_school_guide` with the flag off) keeps the existing
  `stream_chat_turn` loop. So per request I branch: FT engine vs. legacy loop, both writing to
  the same chat session/messages and emitting the *same* NDJSON contract. Nothing in the docs
  speaks to running an FT `Workflow` for *some* turns of a session and the legacy loop for
  others, or to reconciling FT's engagement/thread identity with the app's existing
  `ChatSession`/`ChatMessage` persistence (the app already persists messages + `form`; FT also
  persists a trace). I need to know FT is happy being *one of two* engines behind a flag and that
  its trace store is purely observational (not the source of truth for chat history).
- **What the docs say (or don't):** README positions FT as *the* workflow engine; no guidance on
  coexisting with a non-FT loop for the same surface, or on the boundary between FT's trace store
  and the app's own conversation persistence.
- **Need / proposed API:** a short "embedding FlowTraicer behind a flag / alongside an existing
  loop" note clarifying that (a) FT's store is observability-only and the app remains the system
  of record for chat history, and (b) starting/resuming an engagement keyed by the app's
  `session_id` is the supported way to align FT threads with app sessions.
- **Acceptance:** documented confirmation that I can run FT for one persona behind a flag, keyed
  by my `session_id`, with the app still owning chat persistence — and a pointer to the
  recommended split of responsibilities.
---
- **Resolution (B):** Doc-only. Added **"Embedding FlowTraicer behind a feature flag, alongside an
  existing loop"** to the new README "Operational model" section, confirming: (a) FT's trace store
  is **observability/audit only** — the application stays the **system of record** for chat history
  (`ChatSession`/`ChatMessage`, message `form`s); don't read chat history back out of FT to render;
  (b) align FT threads to app sessions by passing your `session_id` as `thread_id=` to
  `start`/`resume` and putting `session_id`/`user_id` in `metadata`, giving 1 app session = 1 FT
  engagement while the two stores stay independent; (c) per-request flag branching (FT vs. legacy
  loop) is supported and, because the FT store is independent + fail-open, never touches your chat
  persistence or other personas. CHANGELOG `[Unreleased]` notes the docs add.
- **Status:** resolved

### 11. Pin the exact shape of `WorkflowTurn.interrupt` (wrapped vs. unwrapped payload)
- **Filed by:** A — Round 3 / 2026-06-25
- **Kind:** doc gap / unclear contract
- **Context:** I read the qualification card off `turn.interrupt` and want
  `MessageForm.model_validate(turn.interrupt)` to work without defensive unwrapping.
- **What the docs say (or don't):** README `WorkflowTurn` table says `interrupt` = "the payload the
  paused node emitted (the card)" — implying it is exactly my `payload`. But the design doc says
  `ctx.pause` calls `interrupt({"awaiting": awaiting, "payload": payload})` and the turn takes "the
  first interrupt's `value` (`{"awaiting","payload"}`)". So it is ambiguous whether `turn.interrupt`
  is my bare `payload` dict or the `{"awaiting","payload"}` wrapper. `hitl_resume.py` prints it as
  the bare card, suggesting unwrapped — but it is not stated normatively.
- **Need / proposed API:** one normative sentence in the README: "`turn.interrupt` IS the `payload`
  dict you passed to `ctx.pause(payload=…)`, verbatim and unwrapped; the `awaiting` label is on
  `turn.awaiting`." (No code change expected if that's already true — just guarantee it.)
- **Acceptance:** I can write `MessageForm.model_validate(turn.interrupt)` directly, relying on a
  documented guarantee that `turn.interrupt` is the unwrapped payload.
---
- **Resolution (B):** **Pinned (no behaviour change — it was already unwrapped; now guaranteed +
  tested).** `ctx.pause` internally calls `interrupt({"awaiting", "payload"})`, but the runner splits
  that: `turn.awaiting` ← the label, `turn.interrupt` ← **the bare `payload` dict, verbatim**.
  Documented normatively in README's `WorkflowTurn` table ("the **unwrapped** `payload` dict") and in
  the `WorkflowTurn.interrupt` docstring ("`MessageForm.model_validate(turn.interrupt)` works
  directly"); design-doc "Resume contract" §#11. Test
  `tests/test_resume_contract.py::test_interrupt_is_unwrapped_payload` asserts `turn.interrupt == card`
  and that no `awaiting`/`payload` wrapper key leaks in. CHANGELOG `0.11.0` Docs.
- **Status:** resolved

### 12. `resume()` contract on a completed / stale thread (double-submit, wrong `awaiting`, empty input)
- **Filed by:** A — Round 3 / 2026-06-25
- **Kind:** unclear contract
- **Context:** `/api/chat` is concurrent and clients double-submit / replay stale turns. I need to
  know what `resume(thread_id=…)` does when the thread's engagement has already `completed`, when the
  graph isn't actually parked on the `awaiting` label my input is for, and what shape a bare
  `resume(input=None)` delivers into the node.
- **What the docs say (or don't):** docs say `resume` "raises `KeyError` if no resumable engagement
  for the thread" — but say nothing about an *already-completed* thread (KeyError? idempotent
  re-deliver? error?), no way to assert the resume input matches the parked `awaiting` label, and
  whether `resume(input=None)` surfaces as `None` vs `{}` inside `ctx.pause(...)` (matters for the
  `reply.get(...)` normalization in #4).
- **Need / proposed API:** document (a) the behaviour/exception when resuming a `completed` thread;
  (b) optionally let `resume(..., expect_awaiting="qualification_confirm")` raise on mismatch so a
  stale client can't deliver the wrong-turn input; (c) the exact value `ctx.pause` returns for
  `input=None`.
- **Acceptance:** I can write a router branch that safely handles a double-submitted / stale resume
  (catch the documented exception → re-render the current card) without reading `src/ft` to learn
  the behaviour.
---
- **Resolution (B):** **Defined + built.** `resume`/`stream_resume` now raise
  `ft.orchestration.ResumeError` (a `RuntimeError` subclass) with a `.reason` you can branch on:
  (a) resuming an **already-completed** (or never-started / abandoned / failed) thread →
  `reason="no_resumable_engagement"` (the double-submit / stale-replay case); (b) an engagement that
  exists but isn't parked → `reason="not_paused"`; (c) a new optional `expect_awaiting="…"` guard
  that, on mismatch with the parked label, raises `reason="awaiting_mismatch"` **before** delivering
  any input (so a stale client can't push wrong-turn input through). `resume(input=None)` makes
  `ctx.pause(...)` return **`None`** (not `{}`) — delivered via LangGraph's interrupt-id-keyed
  resume-map form, since a bare `Command(resume=None)` hits a LangGraph `UnboundLocalError`.
  Documented in README "The resume contract" (with the try/except branch snippet) + design-doc
  "Resume contract" §#12; tested in `tests/test_resume_contract.py`
  (`test_resume_on_completed_thread_raises_resume_error`, `test_resume_expect_awaiting_mismatch_raises`,
  `test_resume_with_none_input_surfaces_as_none`, `test_resume_unknown_thread_still_raises_resume_error`).
  CHANGELOG `0.11.0` (Changed: `ResumeError` replaces the bare `KeyError`).
- **Status:** resolved

### 13. Can a single `resume` itself return `status="paused"` again (chained pauses across nodes)?
- **Filed by:** A — Round 3 / 2026-06-25
- **Kind:** doc gap
- **Context:** A realistic school flow can pause twice: qualify-card (pause) → on confirm, run a
  clarifying step that emits ANOTHER card and pauses again — all driven by repeated HTTP turns.
- **What the docs say (or don't):** the design doc says `start`/`resume` "runs until the next pause or
  completion", which *implies* a `resume` can return `status="paused"` again with a new
  `awaiting`/`interrupt`. But every README/example/test shows exactly one pause then completion;
  the multi-pause case is never demonstrated.
- **Need / proposed API:** one example (or a sentence + the existing example extended) showing a
  `resume` that returns `paused` again at a *different* node, so I can drive an N-card flow as a loop:
  ```python
  turn = await wf.start(...) ; while turn.is_paused: reply = collect(turn); turn = await wf.resume(input=reply)
  ```
- **Acceptance:** documented confirmation (with a snippet) that `resume` can re-pause at a new node,
  so a multi-card flow is a simple `while turn.is_paused: resume(...)` loop.
---
- **Resolution (B):** **Confirmed + demonstrated.** Because `start`/`resume` each run "until the
  *next* pause or completion", a `resume` can return `status="paused"` again at a **different** node.
  Documented with the loop snippet in README "The resume contract" (chained-pauses bullet) +
  design-doc §#13; the runnable `ft.examples.streaming_turn` is a two-card flow driven by
  `while turn.is_paused: stream_resume(...)`. Test
  `tests/test_resume_contract.py::test_resume_can_re_pause_at_a_different_node` asserts a 3-node graph
  pauses at `card_one`, the resume **re-pauses** at `card_two`, and the next resume completes — the
  `seen == ["card_one", "card_two"]` sequence. CHANGELOG `0.11.0` Docs.
- **Status:** resolved

### 14. A cardless pause that purely hands the turn back (await the next user message)
- **Filed by:** A — Round 5 / 2026-06-25 (SP4 planning)
- **Kind:** doc gap / unclear contract
- **Context:** The SP4 school flow wants "one app chat session = one FT engagement" across the whole
  conversation, not just the qualify gate. After the agentic `assist` step answers, the flow should
  **pause to await the next user message** and resume into `assist` again (chained pauses, #13) — so
  the engagement stays one continuous `PAUSED` journey rather than completing each turn. That await
  step has **no card** to show; it pauses solely to yield the turn back to the endpoint.
- **What the docs say (or don't):** every pause example carries a `payload` (a card) surfaced as
  `turn.interrupt`. Nothing states what `ctx.pause(awaiting="user_turn", payload=None)` yields:
  is `turn.interrupt` then `None`? Does the streaming terminal `paused` event carry no card so I emit
  no `message_form` line? (`resume(input=None)` → `ctx.pause` returns `None` is documented (#12), but
  that's the *resume* side, not the *pause-with-no-payload* side.)
- **Need / proposed API:** one normative sentence: `ctx.pause(awaiting=…, payload=None)` (or omitted)
  yields `turn.interrupt is None` (and the streaming terminal `paused` event likewise carries no
  card), so a node can pause purely to await input. A tiny example extension would help.
- **Acceptance:** I can write an `await_user` node `await ctx.pause(awaiting="user_turn")`, read
  `turn.is_paused and turn.interrupt is None`, emit only `text_chunk`+`usage` (no `message_form`), and
  resume into the next agentic turn — driving the whole session as one engagement.
---
- **Resolution (B):** **Guaranteed + tested + exampled (no behaviour change — it already worked; now
  it's a contract).** `ctx.pause(awaiting=…, payload=None)` (or `payload` omitted) is a first-class
  cardless pause: the runner splits the interrupt wrapper so `turn.awaiting` is the label and
  `turn.interrupt` is exactly the `payload` you passed — which is `None` when omitted. The streaming
  terminal `paused` event likewise carries no card (`ev.turn.interrupt is None`), so the app emits no
  `message_form` line. This makes **shape A** (whole chat session = one FT engagement) safe: a thin
  `await_user` node pauses cardlessly after each agentic `assist` answer and resumes back into
  `assist` (a re-entrant loop), so the engagement stays `PAUSED` and never finishes. Per-turn
  `token_usage` is scoped to each turn's steps only (verified across a 6-turn loop). The streamed
  answer must live in the node **after** the cardless pause so it streams/charges once (replay rule).
  - **Docs:** README "Pausing for human input…" → new **"Cardless pauses (hand the turn back with no
    card)"** bullet, with the `await_user` snippet and the shape-A explanation.
  - **Example:** `python -m ft.examples.session_loop` — `qualify → assist → await_user → assist …`
    over 5 turns, one continuous `PAUSED` engagement.
  - **Tests:** `tests/test_cardless_pause.py` (4 tests): `payload=None` → `turn.interrupt is None`;
    streaming terminal `paused` carries no card; re-entrant loop pauses at the same node many times
    under one engagement (stays `PAUSED`); per-turn token accounting scoped to that turn.
  - **CHANGELOG:** `0.11.1` (Docs/Tests — guarantee for the cardless-pause + re-entrant-loop shape).
- **Status:** resolved

### 15. Can `ctx.emit` be called from inside an `AgentTool.handler` (not just a node body)?
- **Filed by:** A — Round 5 / 2026-06-25 (SP4 planning)
- **Kind:** unclear contract
- **Context:** SP4 wraps the app's existing `ToolRegistry` tools as `ft.agent.AgentTool`s for an
  `@wf.agent_step`. Several app tools return **render-ready events** (`school_comparison`,
  `embassy_visa_info`, and the per-school consent `message_form`) that the frontend must render
  mid-turn. The natural carrier is `ctx.emit(payload)` — but the tool runs inside `ctx.run_tools`'s
  ReAct loop via `handler(args, ctx)`, and the README only shows `ctx.emit` from a **node body**.
- **What the docs say (or don't):** README "Streaming a turn" shows `ctx.emit(payload)` mid-node and
  says it's a no-op outside a streaming run; the agentic-step section shows `handler(args, ctx)` and
  `ctx.deps` but does **not** state whether the handler's `ctx` exposes `emit`, nor whether emits from
  within `run_tools` surface as `emit` stream events on the enclosing step's turn.
- **Need / proposed API:** confirm (and document) that the `ctx` passed to an `AgentTool` handler is
  the running step's context — i.e. `ctx.emit(...)` from a handler surfaces as an `emit` stream event
  during the agentic loop (and is a no-op under `run`/non-streaming). If not, document the supported
  way for an agentic tool to push a render payload to the client mid-loop.
- **Acceptance:** an `AgentTool.handler` calling `ctx.emit({"type":"school_comparison", ...})` during
  `ctx.run_tools` produces an `emit` `StreamEvent` I can map to the existing NDJSON event line.
---
- **Resolution (B):** **Confirmed + tested + documented (it already worked — now it's a contract).**
  `ctx.run_tools` invokes each tool's `handler(args, ctx)` passing the **running step's own
  `StepContext`** (`tool.invoke(req.args, self)`), so the handler's `ctx` is identical to the node
  body's — it exposes `ctx.deps`, `ctx.emit`, `ctx.llm`, etc. A handler calling `ctx.emit(payload)`
  during the agentic loop surfaces as an `emit` `StreamEvent` under `stream`/`stream_resume`
  (interleaved with the loop's tool calls), and is a harmless **no-op** under non-streaming
  `run`/`start`/`resume` (the stream writer is `None`, exactly as for `ctx.emit` from a node body).
  - **Docs:** README "A multi-tool agentic step…" → new **"`ctx.emit(...)` works from inside a tool
    handler"** bullet, with a `compare_schools` handler emitting a `school_comparison` card.
  - **Tests:** `tests/test_emit_from_handler.py` (3 tests): handler `ctx.emit` surfaces as an `emit`
    StreamEvent under `stream`; no-op (no crash, tool still runs) under `run`; no-op under `start`.
  - **CHANGELOG:** `0.11.1` (Docs/Tests).
- **Status:** resolved

### 16. Endorse the resume-else-start idiom for an endpoint that doesn't track session start
- **Filed by:** A — Round 5 / 2026-06-25 (SP4 planning)
- **Kind:** doc gap (minor)
- **Context:** `/api/chat` doesn't independently know whether a session already has a paused FT
  engagement; the cleanest dispatch is "try `stream_resume`; if it raises
  `ResumeError(reason="no_resumable_engagement")`, fall back to `stream` (a fresh start)". I want
  confirmation this is the intended idiom rather than requiring an out-of-band "has this thread
  started?" check against the store.
- **What the docs say (or don't):** the resume contract documents `ResumeError.reason` values (#12)
  but frames `no_resumable_engagement` as the *double-submit/stale* case; it doesn't explicitly bless
  using that branch as the **new-session** signal in a try/except dispatch.
- **Need / proposed API:** one README sentence endorsing try-`stream_resume`/except-
  `no_resumable_engagement`→`stream` as the supported "resume-or-start" dispatch (or, if not endorsed,
  the recommended alternative — e.g. an `is_thread_active(thread_id)` helper).
- **Acceptance:** a documented, blessed one-branch dispatch for "resume the session or start it",
  without a separate store lookup.
---
- **Resolution (B):** **Endorsed (doc-only; no behaviour change).** README "Streaming a turn…" → new
  **"Resume-or-start dispatch (when the endpoint doesn't track session start)"** subsection blesses
  the idiom: try `stream_resume(thread_id=…)`, and on
  `ResumeError(reason="no_resumable_engagement")` fall back to `stream(...)` (a fresh start) — no
  out-of-band "has this thread started?" store lookup. Includes the try/except snippet and notes the
  same shape applies to non-streaming `resume`/`start`, plus the caveat that
  `no_resumable_engagement` also covers an already-completed thread (so use your own per-turn
  idempotency key to tell a genuine "start over" from a stale duplicate of a finished turn).
  - **CHANGELOG:** `0.11.1` (Docs).
- **Status:** resolved

### 17. Does `build_checkpointer("postgres"/"sqlite", …)` provision its own tables? (one-time setup)
- **Filed by:** A — Round 5 / 2026-06-25 (SP4 planning, risk §16 #E)
- **Kind:** doc gap / unclear contract (ops)
- **Context:** SP4 prod uses `build_checkpointer("postgres", dsn=settings.flowtraicer_dsn)` for
  durable cross-worker resume, sharing the FT trace DB. It was unclear whether the factory runs
  `PostgresSaver.setup()` (creating its checkpoint tables) on first use or the app must call setup
  once out of band — and whether those tables coexist cleanly with the FT trace tables in one DB.
- **What the docs say (or don't):** `ft/checkpoint.py` and the design doc described the backends but
  not table provisioning; the factory also returned `*Saver.from_conn_string(...)` (a **context
  manager**, not a ready saver) and never ran `setup()`.
- **Need / proposed API:** pick + document a clear behaviour (e.g. `setup=True` default) and confirm
  the checkpoint tables can share the FT trace DB.
---
- **Resolution (B):** **Made ergonomic + documented.** `build_checkpointer(backend, *, setup=True,
  **kwargs)` now: (a) **enters** the saver's `from_conn_string(...)` context manager to return a
  ready-to-use, process-lived saver (it previously returned the un-entered context manager — a latent
  bug for the durable backends); and (b) **runs the saver's idempotent `.setup()` by default**
  (`CREATE TABLE IF NOT EXISTS`), so a freshly-pointed empty DB just works on first build. Pass
  `setup=False` to skip DDL when you provision the tables via a migration. `"memory"` ignores
  `setup` (no tables). Documented that the checkpoint tables (`checkpoints`/`checkpoint_writes`/
  `checkpoint_blobs`) share the FT trace DB cleanly — distinct, non-colliding names vs the `ft_*`
  audit tables — so one DSN can back both the audit log and resumable execution state.
  - **Code:** `src/ft/checkpoint.py` (`build_checkpointer(setup=…)` + `_enter_and_setup`).
  - **Docs:** module docstring "One-time table provisioning (`setup`)" + README checkpointer note.
  - **Tests:** `tests/test_checkpointer_factory.py` (6 tests): memory needs no setup; postgres w/o
    dsn raises; unknown backend raises; `_enter_and_setup` enters the CM + runs setup by default,
    skips it when `setup=False`, and accepts a plain saver. (Real postgres/sqlite round-trips stay
    skipped like the other DB-backed tests — they need the optional extras + a live DB.)
  - **CHANGELOG:** `0.11.1` (Fixed: durable checkpointers now usable; Added: `setup=` + provisioning).
- **Status:** resolved

## Resolved
