# FlowTraicer — Cross-turn checkpoint / resume (Sub-project 1)

Date: 2026-06-25
Status: implemented (this round)
Closes: NEEDS.md #1 (keystone). Foundation for #2, #3, #4, #8.

## Problem

A multi-turn, human-in-the-loop chat advances **one human exchange per HTTP request**. A node must
be able to run partway, **emit something and stop to wait for the user's reply**, and then — on a
*separate* HTTP request that may hit a *different process* minutes later — **resume from exactly
where it paused**, with the human's input folded in, as the **same** FlowTraicer engagement.

Before this change, `Workflow.run()` ran the graph to completion in one call and returned a finished
`engagement_id: str`. There was no pause, no resume, no durable execution state, and no stable key
that let a later call *continue* a prior engagement instead of starting a new one.

## What we add (public API)

Two new methods on `Workflow`, a per-step `ctx.pause(...)`, a richer return type, and a new
engagement status. **`Workflow.run()` is unchanged** (still runs to completion, still returns
`str`) — the existing analytics samples (`real_estate/workflows/*.py`) keep working verbatim.

```python
# ── types ────────────────────────────────────────────────────────────────────
class WorkflowTurn(BaseModel):
    engagement_id: str
    thread_id: str
    status: Literal["paused", "completed"]   # mirrors EngagementStatus paused/completed
    awaiting: str | None = None              # the label the paused node is waiting on
    interrupt: dict | None = None            # the payload the paused node emitted (the card)

    @property
    def is_paused(self) -> bool: ...

# ── pausing from inside a node ─────────────────────────────────────────────────
@wf.step
async def qualify(state, ctx):
    card = build_card(state)
    reply = await ctx.pause(awaiting="qualification_confirm", payload=card)  # stops here turn 1
    # turn 2 resumes HERE with reply == the value passed to resume(input=...)
    return {"messages": [f"got {reply}"]}

# ── turn 1: run until a node pauses (or to completion) ──────────────────────────
turn = await wf.start(initial_state, recorder, thread_id=session_id,
                      llm=llm, deps=deps, metadata={"session_id": session_id})
# turn.status == "paused"; turn.awaiting == "qualification_confirm"; turn.interrupt == card

# ── turn N (a later HTTP request, possibly a fresh process): resume same engagement ─
turn = await wf.resume(thread_id=session_id, recorder, input={"confirm": True},
                      llm=llm, deps=deps)
# turn.status == "completed"
```

- **`start(input, recorder, *, thread_id, name=None, metadata=None, config=None, llm=None,
  deps=None) -> WorkflowTurn`** — begins a checkpointed run keyed by `thread_id`. Runs until the
  graph either pauses (a node called `ctx.pause`) or completes. Returns a `WorkflowTurn`.
- **`resume(thread_id, recorder, *, input=None, expect_awaiting=None, llm=None, deps=None,
  config=None) -> WorkflowTurn`** — continues the checkpointed graph for `thread_id`, delivering
  `input` as the return value of the paused node's `ctx.pause(...)`. Runs until the next pause or
  completion. Reattaches to the **same** engagement (no new `EngagementStarted`). A single `resume`
  may itself return `status="paused"` again at a different node (**chained pauses** — see "Resume
  contract" below). Raises **`ResumeError`** (a `RuntimeError` subclass with a `.reason`) if the
  thread can't be resumed as asked.

### Resume contract (the edge cases — NEEDS.md #11, #12, #13)

These were ambiguous in the first cut and are now pinned (with tests in `tests/test_resume_contract.py`):

- **`turn.interrupt` is the unwrapped `payload`.** Although `ctx.pause` internally calls
  `interrupt({"awaiting": awaiting, "payload": payload})`, the runner *splits* that wrapper: the
  label goes to `turn.awaiting` and **`turn.interrupt` is exactly the `payload` dict you passed**,
  verbatim — so `MessageForm.model_validate(turn.interrupt)` is safe with no unwrapping. (#11)
- **`resume()` failure modes raise `ResumeError(reason=…)`** instead of a bare exception, so a
  router can branch:
  - `reason="no_resumable_engagement"` — no in-flight engagement bound to `thread_id` (never
    started, **or already completed/abandoned/failed** — the double-submit / stale-replay case).
  - `reason="not_paused"` — the engagement exists but isn't parked at a pause.
  - `reason="awaiting_mismatch"` — `expect_awaiting="…"` was given and doesn't match the parked
    label (a stale client delivering wrong-turn input). The check runs *before* any input is
    delivered. (#12)
- **`resume(input=None)`** (a bare resume) makes `ctx.pause(...)` return **`None`** (not `{}`).
  Guard with `reply or {}` if a node assumes a dict. (LangGraph rejects a bare
  `Command(resume=None)`; we deliver `None` via the interrupt-id-keyed resume-map form internally.)
  (#12)
- **Chained pauses.** Because `start`/`resume` each run "until the *next* pause or completion", a
  single `resume` can return `status="paused"` again at a different node. Drive an N-card flow as a
  loop: `turn = await wf.start(...); while turn.is_paused: turn = await wf.resume(input=collect(turn))`.
  (#13)
- **`ctx.pause(*, awaiting: str, payload: dict | None = None) -> Any`** — inside a node, emit
  `payload` to the caller and suspend; on resume, returns the value passed to `resume(input=...)`.

`run()` remains: run-to-completion, **no checkpointer**, returns `engagement_id: str`. It does not
support `ctx.pause` (a node that pauses under `run()` will raise, by design — pausing requires a
checkpointer, which only `start`/`resume` configure).

## How it integrates with LangGraph

We **wrap LangGraph's official savers** rather than hand-roll checkpoint serialization (decision
below). The mechanics, verified against `langgraph==1.2.6` / `langgraph-checkpoint==4.1.1`:

1. **Compile with a checkpointer.** `start`/`resume` compile the graph *with* a checkpointer
   (`graph.compile(checkpointer=…)`); `run()` compiles *without* one. We therefore keep **two
   cached compiled graphs** on the `Workflow`: `_compiled` (no checkpointer, for `run`) and
   `_compiled_ckpt` (with checkpointer, for `start`/`resume`). Both are built once and reused.
2. **`thread_id` keys execution state.** Every `start`/`resume` passes
   `config={"configurable": {"thread_id": thread_id}}`. LangGraph stores the post-node checkpoints
   under that thread; a later `resume` with the same `thread_id` picks up exactly where it left off
   — *even in a fresh process*, as long as the checkpointer backend is durable (SQLite/Postgres).
3. **`ctx.pause` == `interrupt()`.** `ctx.pause(awaiting=…, payload=…)` calls LangGraph's
   `interrupt({"awaiting": awaiting, "payload": payload})`. The node suspends; the debug stream's
   `task_result` for that node carries a non-empty `interrupts: [...]` and an empty `result`.
4. **Detecting the pause.** After `astream(...)` drains, we read `compiled.get_state(config)`. If
   `snapshot.next` is non-empty **and** `snapshot.interrupts` is non-empty, the run paused: we take
   the first interrupt's `value` (`{"awaiting", "payload"}`) as the turn's `awaiting`/`interrupt`.
   Otherwise the run completed.
5. **Resume.** `resume` calls `astream(Command(resume=input), config={…thread_id…})`. LangGraph
   **re-invokes the interrupted node from its top**, and `interrupt()` now *returns* `input` instead
   of suspending. (Implication, documented for callers: code in a node **before** `ctx.pause` runs
   again on resume — keep it idempotent / side-effect-light, or guard it.)

### One engagement spans the pause (trace ↔ checkpointer separation)

The **trace store is the audit log; the checkpointer is the resumable execution state.** They are
deliberately separate concerns with separate backends:

- On `start`, we create **one** engagement (`recorder.start_engagement(...)`) and stamp the binding
  `metadata["ft_thread_id"] = thread_id` (alongside the caller's metadata). We do **not** call
  `end_engagement` when the run pauses — instead we record a paused marker (a `log` event
  `engagement_paused` plus leaving the engagement non-ended) and the turn reports `status="paused"`.
  We expose the engagement's effective status as `PAUSED` via a lightweight marker rather than the
  terminal `EngagementEnded` (which would close it).
- On `resume`, we **find the existing engagement** for the thread via
  `store.list_engagements(where={"ft_thread_id": thread_id})` (most-recent, not-yet-completed), and
  record the resumed steps **under that same `engagement_id`**. No second `EngagementStarted`.
- When a resume runs the graph to completion, we `end_engagement(COMPLETED)` (or `ABANDONED` if goal
  nodes were declared and unmet). The reconstructed engagement then shows **one** id with the paused
  node recorded as `WAITING` on the pausing turn and `COMPLETED` on the resuming turn, followed by
  the downstream nodes.

So: the **trace store** answers "what happened, for audit/analytics" and is keyed by
`engagement_id` (joined to a thread via `metadata.ft_thread_id`); the **checkpointer** answers "how
do I continue execution" and is keyed by `thread_id`. Either can be purged/retained on its own
schedule. The store is never the resume buffer, and the checkpointer never holds trace records.

### Step status on a pause

A node that pauses is recorded with new step status **`StepStatus.WAITING`** for the pausing turn
(it has not failed or completed — it's parked for input). On resume, the node re-runs and ends
`COMPLETED`. We add **`EngagementStatus.PAUSED`** for the engagement-level "waiting on a human" state
(surfaced on the `WorkflowTurn`, and as a recorded marker; the engagement's `EngagementEnded` is not
written until it truly completes/abandons/fails).

## Checkpointer backend — decision

**Decision: wrap LangGraph's official savers behind a tiny `ft.checkpoint.build_checkpointer(...)`
factory; default to `MemorySaver`.** Reasons:

- **Don't hand-roll serialization.** LangGraph's checkpoint format (channel versions, pending
  writes, interrupt bookkeeping) is intricate and evolves with the engine. Re-implementing it over
  the FT trace store would couple us to LangGraph internals and be a perennial source of resume
  bugs. The savers are the supported, tested path.
- **Pluggable, matched to the store backend.** The factory maps the deployment to a saver:
  `MemorySaver` (default; dev / single-process), and — when the user installs the extra — the
  official `SqliteSaver` / `PostgresSaver` for durable cross-process resume. `start`/`resume` accept
  an explicit `checkpointer=` too, so callers can pass any `BaseCheckpointSaver`.
- **Separation of concerns preserved.** The checkpointer is *only* execution state. The audit trail
  stays in the FT `Store`. A user can run `PostgresStore` for the audit log and `PostgresSaver`
  (same DB, different tables) for resume — or `MemorySaver` for resume if they only need
  within-process multi-turn (acceptable when a session is pinned to one worker), keeping the durable
  audit in Postgres regardless.

`MemorySaver` is the zero-dependency default so the feature works out of the box and in tests. For
true cross-process durability (a session that may land on a different worker), pass a durable saver:

```python
from ft.checkpoint import build_checkpointer
ckpt = build_checkpointer("memory")           # default
# ckpt = build_checkpointer("sqlite", path="ckpt.db")   # needs langgraph-checkpoint-sqlite
# ckpt = build_checkpointer("postgres", dsn=DSN)        # needs langgraph-checkpoint-postgres
wf = Workflow("school", state_schema=State, checkpointer=ckpt)
```

If no checkpointer is configured on the `Workflow` and none is passed to `start`/`resume`, a process
-wide `MemorySaver` is created once and reused (so multiple threads in one process share resumable
state). The README documents that within-process `MemorySaver` resume requires session affinity;
use a durable saver to lift that constraint.

## Backward compatibility

- `run()` signature, behaviour, and `-> str` return type are unchanged. `real_estate/workflows/
  school_workflow.py` and `apartment_workflow.py` (and their `run_*_workflow` wrappers, which return
  `str`) are untouched and keep passing.
- New optional `Workflow(..., checkpointer=…)` ctor arg defaults to `None` (lazy `MemorySaver` on
  first `start`).
- `EngagementStatus.PAUSED` / `StepStatus.WAITING` are additive enum members; `fold`/analytics treat
  any non-`COMPLETED`/`ABANDONED` engagement as in-flight as before.

## Tests (TDD — written red first)

`tests/test_checkpoint_resume.py`:

1. **pause returns a paused turn** — 2-node graph (`a` calls `ctx.pause`, then `b`); `start` returns
   `status="paused"`, the interrupt payload, the `awaiting` label, and an `engagement_id`.
2. **resume completes** — `resume(input=…)` finishes `a` and runs `b`; `status="completed"`.
3. **one engagement across the pause** — the reconstructed engagement has **one** id; node `a`
   appears once as the journey, `b` completed; the resume did not start a second engagement.
4. **fresh-process round-trip** — a brand-new `Workflow` instance (simulating a separate HTTP
   request/worker) sharing the *same* checkpointer + store, same `thread_id`, resumes and completes.
5. **`ctx.pause` delivers the resume input** — the value passed to `resume(input=…)` is what
   `ctx.pause(...)` returns inside the node.
6. **`run()` still works** — existing run-to-completion path unaffected (covered by the existing
   suite; one direct assertion here too).
