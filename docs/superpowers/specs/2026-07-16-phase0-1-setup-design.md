# SpendWeiss: Phase 0 and Phase 1 design

Date: 2026-07-16
Status: approved, awaiting spec review

## Purpose

SpendWeiss is a multi phase learning project that builds up an agentic AI system for card and spend optimisation, from a hand rolled ReAct loop through to a deployed multi agent product. This document scopes only the first two phases: project setup and the raw ReAct loop. Later phases (LangChain, memory, LangGraph, critic node, multi agent supervisor, human in the loop, deployment) are recorded as a roadmap in the top level project brief and will each get their own spec when work reaches them.

## Goals

- Stand up the project skeleton with `uv` for environment and dependency management.
- Produce a hand rolled ReAct loop in plain Python, calling the Groq API directly, with no agent framework involved. This is deliberate: the point of Phase 1 is to see the raw mechanism before any framework hides it.
- Establish the working conventions for the rest of the project: branch per phase, a journal, British English throughout, no comments in code, no em dashes anywhere.

## Non goals

- No LangChain, LangGraph or native Groq tool calling (`tools=` parameter) in Phase 1. That arrives in Phase 2.
- No FastAPI backend or frontend yet. That is Phase 8.
- No automated test suite for Phase 1. Verification is manual, by running sample queries and inspecting the trace.

## Tooling and workflow

- `uv init` to create the backend project, `uv add <package>` for dependencies, `uv run agent.py` to execute.
- One git branch per phase (for example `phase-0-setup`, `phase-1-raw-react`), merged into `main` once that phase is built and manually verified. No `git push` or PR creation without an explicit request each time, per user instruction.
- All prose in this repository (docs, journal, commit messages, code comments if any are ever unavoidable) is written in British English, with no em dashes.
- No comments in code. Identifiers should be descriptive enough to make comments unnecessary.

## Repository layout

```
Spendweiss/
  .gitignore
  JOURNAL.md
  README.md
  docs/
    superpowers/
      specs/
      plans/
  backend/
    pyproject.toml
    .env
    data/
      cards.json
      offers.json
      transactions.json
    phase1_raw_react/
      agent.py
```

## .gitignore

Root level `.gitignore` covering: `.venv/`, `.env`, `__pycache__/`, `*.pyc`, `.DS_Store`, and standard Python build artefacts (`*.egg-info/`, `dist/`, `build/`).

## Mock data

Three JSON files under `backend/data/`:

- `cards.json`: six real Indian credit cards (HDFC Millennia, Axis Bank Magnus, ICICI Amazon Pay, HDFC Infinia, SBI Cashback, IDFC FIRST Select), each with an issuer, annual fee, and reward rates per spending category (dining, groceries, travel, online shopping, fuel, other), simplified from each card's real, published reward structure rather than invented. This includes realistic details such as most cards excluding fuel from reward earning entirely, and IDFC FIRST Select earning far less than the cashback focused cards, its real advantage being a zero annual fee rather than reward rate.
- `offers.json`: six time limited offers, one per card so every card in the catalogue has a promotion, each tied to a real merchant and grounded in a real promotion type for that card, with a validity window.
- `transactions.json`: twenty four past transactions, each with a date, merchant, category, amount and the card used, spanning all six cards and all six spending categories across twelve distinct Indian merchants.

## Phase 1: the ReAct loop

`backend/phase1_raw_react/agent.py`:

- Uses the `groq` Python client (`Groq(api_key=os.environ["GROQ_API_KEY"])`) and plain `chat.completions.create`. Does not use Groq's native `tools=` parameter; the tool calling protocol is defined by hand in the system prompt, as a strict JSON schema (either `{"action": "<tool name>", "args": {...}}` or `{"action": "final_answer", "answer": "..."}`).
- Two tool functions that read the JSON data files directly: `check_card_rewards(category)` and `check_offers(merchant)`.
- A `while` loop that sends the running message history to Groq, parses the response as JSON, executes the named tool if one is requested and appends its result to the message history, or prints the final answer and stops.
- A hard cap of six loop iterations per query, to guard against runaway loops.
- If the model's output fails to parse as valid JSON or names an unknown tool, one corrective message is sent back into the loop. If it fails a second time, that query is aborted with a clear error message rather than looping indefinitely.
- Runs as an interactive terminal loop: prompts the user for a purchase description, prints every reasoning step, tool call and tool result as they happen, then the final recommendation, then prompts again. Exits on Ctrl+C.

## Verification

Manual only for Phase 1. Run the script against two or three sample purchase descriptions covering different categories (for example a dining purchase, a travel purchase, and one that matches an active offer) and confirm the trace and final recommendation are sensible given the mock data.

## Open questions

None outstanding. All prior questions in this design conversation have been resolved.
