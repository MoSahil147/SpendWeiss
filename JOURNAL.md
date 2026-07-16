# SpendWeiss Journal

A running log of what got built, decided and learned at each phase of the SpendWeiss project. One entry per phase, added once that phase is verified and merged to main.

---

## Template

```
## Phase N: <name> (YYYY-MM-DD)

**What I built:**

**Key decisions:**

**Gotchas and bugs hit:**

**What I learned:**

**Next up:**
```

---

## Phase 0: Setup (2026-07-16)

**What I built:**
Scaffolded the backend project with `uv init backend`, removed the generated stub script, and added `groq` and `python-dotenv` as dependencies via `uv add`. Created `backend/.env.example` and `backend/.env` (the latter holding a placeholder key until a real Groq key is added). Created the three mock data files, all reflecting the Indian market: `backend/data/cards.json` (six real Indian credit cards: HDFC Millennia, Axis Bank Magnus, ICICI Amazon Pay, HDFC Infinia, SBI Cashback, IDFC FIRST Select, with reward rates researched via web search and simplified from each card's actual published structure, not invented), `backend/data/offers.json` (six offers, one per card, each grounded in a real promotion type researched for that card: BigBasket, IndiGo via the EDGE travel portal, Amazon.in, MakeMyTrip via SmartBuy, Myntra, and the District app BOGO movie offer), and `backend/data/transactions.json` (twenty four past transactions spanning all six cards and all six categories, across twelve Indian merchants including BigBasket, DMart, Swiggy, Zomato, IndiGo, MakeMyTrip, Ola Cabs, Amazon.in, Myntra, Croma, Netflix, District and two petrol pumps, amounts in rupees). All three JSON files verified to parse correctly and to match what the plan document specifies byte for byte.

**Key decisions:**
Working on branch `phase-0-1-setup`, checked out from `main`, not `phase-0-setup` and `phase-1-raw-react` as two separate branches. Phase 0 and Phase 1 were designed as a single spec and plan, so they are being built on a single branch and will be merged to `main` in one go, by the user, once both are verified. `.env` and `.env.example` live inside `backend/`, not the repository root, so that `python-dotenv`'s default lookup from the current working directory finds them automatically when running `uv run` from inside `backend/`, and so that backend secrets stay separate from the frontend's own env file once Phase 8 adds one. All mock data was switched from a generic placeholder market to real Indian banks and merchants at the user's request, expanded from three cards to six, and then corrected a second time so the reward rates reflect each card's real, published terms rather than plausible sounding invented numbers. This surfaced a realistic and useful detail: most of these cards genuinely exclude fuel from reward earning (offering only a surcharge waiver instead), and IDFC FIRST Select genuinely earns far less than the cashback focused cards, since its real advantage is being lifetime free rather than high reward rate. `check_card_rewards` does not model this fee trade-off, only raw reward rate, which is a reasonable Phase 1 simplification worth revisiting later.

**Gotchas and bugs hit:**
`uv init` generates a placeholder `main.py` by default, which is not needed here and was deleted.

**What I learned:**
`uv init <name>` both creates the project directory and initialises it in one step, no need for a separate `mkdir` first.

**Next up:**
Phase 1: build `backend/phase1_raw_react/tools.py` (the `check_card_rewards` and `check_offers` functions) and `backend/phase1_raw_react/agent.py` (the hand rolled ReAct loop).

**Phase 0 status: complete**, verified with a final check: dependencies importable from the `uv` venv, all three data files parse and cross reference correctly (every card has exactly one offer), `.env` and `.env.example` both present, and `.env` confirmed git ignored.

---

## Phase 1: The raw ReAct loop (2026-07-16)

**What I built:**
`backend/phase1_raw_react/tools.py`, with `check_card_rewards(category)` and `check_offers(merchant)` reading the JSON data directly. `backend/phase1_raw_react/agent.py`, a hand rolled ReAct loop against the Groq API (`llama-3.3-70b-versatile`), using a JSON action protocol defined by hand in the system prompt rather than Groq's native `tools=` parameter. Capped at six iterations, with one corrective retry on malformed JSON or an unknown action before aborting that query.

**Key decisions:**
Verified `tools.py` directly first (`check_card_rewards('online_shopping')`, `check_offers('bigbasket')`), matching the plan's expected output exactly, including the three way tie at the top of online shopping. Then verified `agent.py` end to end by piping three sample purchase descriptions into it non-interactively, rather than a live terminal session, since this runs inside an agentic coding session without a real TTY.

**Gotchas and bugs hit:**
Piped stdin runs out eventually and `input()` raises `EOFError` once it does. This is expected for the piped test harness, not a bug in the script, a real interactive terminal session exits cleanly on Ctrl+C instead, which is what the `except KeyboardInterrupt` branch is for.

**What I learned:**
Watching the actual trace made the model's reasoning legible in a way that's easy to take for granted once frameworks hide it: for the BigBasket query, the model correctly weighed a 10 percent limited time offer against a higher underlying reward rate on a different card, and picked the offer because it produced more rupees of value on that specific transaction. That kind of comparison is exactly what Phase 1 was meant to make visible.

**Next up:**
The CI pipeline (pytest suite, GitHub Actions workflow, branch protection, auto-merge) is in progress in parallel on this same branch. Once that is finished and the journal entry for it is added, this branch is ready for a pull request. After that, Phase 2 rebuilds this same agent using `langchain-groq`.

**Convention change, 2026-07-16:** the original "no comments in code" rule from the Phase 0 and 1 spec is reversed, from now on code carries explanatory British English comments. This is a learning project, comments explaining the why and the mechanics are part of the point, not clutter. `tools.py` and `agent.py` were retrofitted with comments accordingly. The `test_data.py` and `ci.yml` files from the CI pipeline work predate this change and were left uncommented, they are short and self-explanatory.
