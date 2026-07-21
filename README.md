# SpendWeiss

An agentic AI card and spend optimiser for the Indian market, built phase by phase to learn agentic AI from first principles: from a hand-rolled ReAct loop to a deployed multi-agent system with human-in-the-loop approval.

**Live app:** [spendweiss.netlify.app](https://spendweiss.netlify.app/)

## How it is built

![SpendWeiss demo](./assets/Demo.gif)

This is a learning project: each phase re-implements or extends the same agent using a different technique, and every phase's code is kept in the repository (not deleted or rewritten) as a record of the progression. The path runs from a manual ReAct loop, through LangChain, persistent memory, an explicit LangGraph state machine, a critic/reflection node, a multi-agent supervisor, human-in-the-loop approval and a served API, to this website. Full details of what was built and learned at each step are in [`JOURNAL.md`](./JOURNAL.md).

**Backend:** Python, FastAPI, LangGraph, LangChain, Groq (`llama-3.3-70b-versatile` for reasoning, `llama-3.1-8b-instant` for classification), and ChromaDB for transaction memory. Multi-key round-robin across `GROQ_API_KEY`/`GROQ_API_KEY1`/`GROQ_API_KEY2` provides resilience on the free tier.

**Frontend:** React, TypeScript, Vite, Tailwind, and React Flow for the decision diagram.

## What it does

Ask it a real spending question, such as "which card should I use for a ₹3000 grocery run?" or "am I wasting money on subscriptions?", and a multi-agent system reasons through it using real card, offer and transaction data:

- **Card recommendations** compare every card's actual reward rate and active offers for the purchase, cite the specific numbers, and explain why the winner beat the alternatives, rather than giving a one-line verdict.
- **Subscription auditing** scans real transaction history for recurring charges and flags which one is most worth reconsidering.
- **A critic reviews every recommendation** before it is shown, and can send it back for one revision if the reasoning does not hold up against the actual data.
- **Large purchases and cancellations pause for human approval** before finalising, using LangGraph's `interrupt()`/resume.
- **The decision diagram shows the real reasoning path**: every node the agent actually visited, in order, with the real tool calls and critic verdicts behind each step, rather than a generic flowchart.

All amounts are shown in Indian Rupees. Nothing is a keyword-matched shortcut; every answer comes from the actual LangGraph agent making real tool calls and real model decisions.

## Running it locally

Requires `uv` (Python) and `npm` (Node).

### Backend

```bash
cd backend
uv sync
```

Create `backend/.env` (copy from `.env.example`) with your Groq API key(s):

```
GROQ_API_KEY=your-key-here
GROQ_API_KEY1=your-second-key-here   # optional, for rate-limit fallback
GROQ_API_KEY2=your-third-key-here    # optional
```

Then run the API:

```bash
uv run uvicorn phase8_api.app:app --reload --host 127.0.0.1 --port 8000
```

Run the test suite (no API key required):

```bash
uv run pytest
```

### Frontend

```bash
cd frontend
npm install
```

Create `frontend/.env` (copy from `.env.example`):

```
VITE_API_BASE_URL=http://localhost:8000
```

Then run the dev server:

```bash
npm run dev
```

Open `http://localhost:5173`.

### Running an individual phase's agent (CLI)

Earlier phases (1 to 7) also run as standalone interactive CLI agents, from `backend/`:

```bash
uv run python -m phase7_human_loop.agent
```

(Swap `phase7_human_loop` for any earlier phase package to see that phase's version of the agent.)

## Deployment

The live app is deployed as two separate services:

- **Frontend** on Netlify, built from `frontend/` (`npm run build`, publishing `frontend/dist`), with `VITE_API_BASE_URL` pointing at the backend.
- **Backend** on Render, running `uv run uvicorn phase8_api.app:app --host 0.0.0.0 --port $PORT` from `backend/`, with `GROQ_API_KEY`(s) and `CORS_ORIGINS` (set to the Netlify URL) as environment variables.

## Project structure

```
backend/
  phase1_raw_react/       # hand-rolled ReAct loop
  phase2_langchain/       # rebuilt on langchain.agents.create_agent
  phase3_memory/          # adds Chroma vector memory over past transactions
  phase4_langgraph/       # rebuilt as an explicit LangGraph StateGraph
  phase5_critic/          # adds a critic/reflection node
  phase6_multiagent/      # adds a Supervisor dispatching to specialist agents
  phase7_human_loop/      # adds a human-in-the-loop approval gate
  phase8_api/             # served as a FastAPI API
  data/                   # mock cards, offers and transaction data (real Indian banks and merchants)
frontend/
  src/
    components/           # QueryInput, AnswerCard, DecisionDiagram, DecisionLogPanel
    api.ts                # typed client for the Phase 8 API
    decisionPath.ts        # pure logic building the diagram from a query's trace
JOURNAL.md                 # running log of what was built, decided and learned per phase
```
