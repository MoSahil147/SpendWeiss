# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

SpendWeiss is an agentic AI card-and-spend optimiser for the Indian market. It's a learning project, built phase by phase, each phase re-implementing (or extending) the same agent with a different technique: starting from a hand-rolled ReAct loop and progressing to a deployed multi-agent system. Every phase lives in its own package under `backend/` and is never deleted or rewritten once later phases move on — they're kept side by side as a record of the progression.

## Commands

All commands run from `backend/`, using `uv`:

```bash
uv sync                                        # install/update dependencies
uv run pytest                                  # run the full test suite
uv run pytest tests/test_phase6_multiagent.py  # run one phase's tests
uv run python -m phase6_multiagent.agent       # run a phase's interactive agent (module form required, see below)
```

Each phase's live agent needs `GROQ_API_KEY` set (via `backend/.env`, loaded with `python-dotenv`). The test suite must pass with no `GROQ_API_KEY` present — see Lazy model initialisation below.

CI (`.github/workflows/ci.yml`) runs `uv sync` + `uv run pytest` on every PR into `main`, then auto-merges (squash) on green and comments on failure.

## Architecture

### Phase packages

Each `backend/phaseN_*` package is self-contained but freely imports from earlier phases rather than duplicating code (e.g. Phase 6's supervisor imports Phase 5's compiled graph, Phase 5 builds on Phase 4's nodes, Phase 3's tools re-export Phase 2's unchanged). Read the phase's entry in `JOURNAL.md` before modifying it — it records *why* the code looks the way it does, not just what it does.

Run a phase's agent as a module (`python -m phaseN_x.agent`), not as a script — Phase 2 onward import across phase package boundaries, which only works when `backend/` is on `sys.path`, and a plain script invocation only puts its own directory there.

Progression so far:
- **Phase 1** (`phase1_raw_react`) — hand-rolled ReAct loop against the raw Groq API, manual JSON action protocol, hard iteration cap.
- **Phase 2** (`phase2_langchain`) — same agent rebuilt on `langchain.agents.create_agent` + `ChatGroq`; tools wrap Phase 1's functions rather than reimplementing them.
- **Phase 3** (`phase3_memory`) — adds memory: a local Chroma `PersistentClient` collection over `transactions.json` for long-term retrieval, plus a session-long `messages` list for short-term memory.
- **Phase 4** (`phase4_langgraph`) — same agent as an explicit LangGraph `StateGraph` (`retrieve_memory -> reason -> [tools loop] -> respond`) instead of `create_agent`'s implicit internal graph. `retrieve_memory` is a mandatory node, not a model-discretion tool call.
- **Phase 5** (`phase5_critic`) — adds a `critic` node after `respond` that reviews the recommendation against tool results already in the conversation (no re-fetching) and can route back to `reason` for exactly one revision, driven by an explicit `critic_verdict` state field (not by inspecting message shape/content).
- **Phase 6** (`phase6_multiagent`) — splits into a multi-agent setup: `CardOptimizerAgent` (Phase 5's graph), a new `SubscriptionHunterAgent` (single-tool `create_agent`, no memory, no critic — deliberately simpler, matched to the task), and a `Supervisor` that classifies each query and dispatches to one or both specialists in sequence.

Check `JOURNAL.md`'s final entry for the current "next up" phase before starting new work.

### Mock data

`backend/data/{cards.json,offers.json,transactions.json}` — real Indian banks, cards, and merchants (not invented), used as the ground truth every phase's tools read from. `transactions.json` is the corpus for both Phase 3's vector memory and Phase 6's subscription detection; changes to it affect both.

### Key cross-cutting patterns

- **Lazy model initialisation.** Never construct a `ChatGroq` (or other client needing `GROQ_API_KEY`) at module import time — build it lazily inside a `_get_model()`-style function on first real use. Module-scope construction breaks `pytest` collection in CI, which has no API key and shouldn't need one just to test graph/function shape.
- **Tool return values must be JSON strings, never raw Python objects** — `json.dumps(...)` even (especially) for empty results. Groq rejects a `role: tool` message whose content is an empty array; an empty list must serialise to the non-empty string `"[]"`.
- **Routing/control-flow decisions are driven by explicit state fields, not by inspecting message content or shape.** Phase 5's `critic_verdict` field exists because the earlier approach (re-parsing an assistant message's text to decide what to do next) broke the moment two assistant-role messages appeared back to back.
- **Recurring-charge detection groups by exact `(merchant, amount)`**, not merchant alone — real variable-amount merchants (groceries, ride-hailing) should *not* be flagged as subscriptions, only genuinely fixed-amount recurring charges should.
- **Comments are welcome and expected** (a repo-specific reversal of the usual "no comments" default) — this is a learning project, so explanatory comments about *why* code is structured a certain way are part of the point, not clutter.

## Planning workflow

Design and planning docs live in `docs/superpowers/specs/` and `docs/superpowers/plans/`, one pair per phase, created via the `brainstorming` and `writing-plans` skills before implementation starts. Follow this same spec → plan → implement flow for new phases rather than jumping straight to code.

## Git workflow

- Each phase is developed on its own branch, checked out from `main`.
- Never `git push`, `git add`, stage, commit, or open a PR — the user does this themselves.
- Never delete branches (locally or in CI) — GitHub's post-merge auto-delete was deliberately disabled; branches from finished phases are kept.
- Open a GitHub issue before starting work on a phase; work against it and close it when done.

## Journal

`JOURNAL.md` is a running, per-phase log (what was built, key decisions, bugs hit, what was learned, what's next). Add an entry when a phase's work is complete — this is where the *why* behind decisions belongs, not in code comments duplicating it.
