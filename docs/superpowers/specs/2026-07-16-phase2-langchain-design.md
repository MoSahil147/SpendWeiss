# SpendWeiss: Phase 2, the agent with LangChain, design

Date: 2026-07-16
Status: approved, awaiting spec review

## Purpose

Rebuild Phase 1's card recommendation agent using LangChain, so that the hand rolled ReAct loop from Phase 1 and the framework managed version sit side by side for comparison. The behaviour is identical: same purchase recommendation logic, same interactive terminal loop, same visible trace of tool calls and reasoning. Only the mechanism changes, from a hand written JSON protocol and while loop to LangChain's tool calling and `create_agent`.

This is also a documented API correction to the original project brief. That brief named `create_tool_calling_agent` and `AgentExecutor`, which were the standard pattern at the time it was written. LangChain has since moved to `create_agent` from `langchain.agents` as the current recommended entry point, confirmed via Context7 documentation lookup on 2026-07-16. `create_agent` runs on LangGraph internally, which is a preview of Phase 4 before that phase is reached explicitly. This spec uses `create_agent`, not the names in the original brief.

## Goals

- Reuse Phase 1's tool logic without duplicating it. `backend/phase1_raw_react/tools.py` is imported from, not copied.
- Wrap `check_card_rewards` and `check_offers` with LangChain's `@tool` decorator, adding the type hints and docstrings Phase 1's versions lack, since LangChain generates each tool's schema from those.
- Build the agent with `ChatGroq` and `create_agent`, keeping the same model, `llama-3.3-70b-versatile`.
- Keep the interactive terminal loop from Phase 1: prompt for a purchase description, print the trace, print a final recommendation, repeat until Ctrl+C.
- Keep the trace visible. `create_agent`'s `.invoke()` hides intermediate steps by default; this spec requires printing every message added to the conversation since the user's input, tool calls, tool results, and the final answer, in the same style Phase 1 used.

## Non goals

- No memory, short term or long term. That is Phase 3.
- No explicit `StateGraph`, no custom nodes or conditional edges. That is Phase 4. `create_agent` uses LangGraph internally, but this spec does not touch that internal graph directly.
- No changes to Phase 1's code or to the mock data. Phase 1 remains untouched as the reference point for comparison.
- No changes to the system prompt's substance. It says less, since LangChain's tool calling machinery replaces the hand written JSON action protocol, but it asks for the same reasoning.

## Repository layout addition

```
backend/
  phase1_raw_react/
    __init__.py
  phase2_langchain/
    __init__.py
    tools.py
    agent.py
```

Phase 1's `agent.py` only ever imported its sibling `tools.py` in the same directory, so it worked fine as a plain script (`uv run phase1_raw_react/agent.py`), Python adds a script's own directory to `sys.path` regardless of packaging. Phase 2 imports across directories, `phase2_langchain/tools.py` importing from `phase1_raw_react/tools.py`, which needs both directories to be proper Python packages (an `__init__.py` each, `phase1_raw_react` did not have one before) and needs `backend/` itself on `sys.path`. The fix is to run Phase 2 as a module rather than a script, from inside `backend/`: `uv run python -m phase2_langchain.agent`, since `-m` puts the current working directory on `sys.path`, not the module's own directory.

## Dependencies

`langchain` and `langchain-groq`, added via `uv add langchain langchain-groq` from inside `backend/`.

## `backend/phase2_langchain/tools.py`

Imports `check_card_rewards` and `check_offers` from `backend.phase1_raw_react.tools` under different names (for example `_check_card_rewards`, `_check_offers`), then defines two `@tool` decorated wrapper functions with the same public names, each with a type hinted argument (`category: str`, `merchant: str`) and a docstring LangChain uses to generate the tool's description for the model. Each wrapper's body is a single line calling the underlying Phase 1 function and returning its result.

## `backend/phase2_langchain/agent.py`

- `ChatGroq(model="llama-3.3-70b-versatile")`, reading `GROQ_API_KEY` from the environment the same way Phase 1 does, via `python-dotenv`'s `load_dotenv()`.
- `create_agent(model, tools=[check_card_rewards, check_offers], system_prompt=SYSTEM_PROMPT)`.
- `SYSTEM_PROMPT`: describes the assistant's purpose and the categories it should reason about, but does not describe a JSON action protocol, LangChain's tool calling handles that formatting.
- The interactive loop mirrors Phase 1: prompt for a purchase description with `input()`, call `agent.invoke({"messages": [{"role": "user", "content": purchase_description}]})`, then print every message in the returned `messages` list beyond the single user message that was sent in, followed by the final recommendation, then loop. Exits cleanly on `KeyboardInterrupt`, matching Phase 1's behaviour.

## Error handling

LangChain and the underlying model handle malformed tool call arguments internally; Phase 1's hand written retry and abort logic for malformed JSON has no equivalent here, since there is no hand written JSON to malform. This is itself one of the things this phase is meant to make visible: what a framework takes off your hands.

## Testing

The project's now standing convention (introduced with the CI pipeline) is to prefer real automated tests over manual only verification where practical. `backend/tests/test_phase2_tools.py` tests the two `@tool` wrapped functions directly, asserting they return the same results as Phase 1's underlying functions for the same inputs. The agent's end to end behaviour, since it depends on a live Groq API call, is verified manually, the same way Phase 1's `agent.py` was: running a small set of sample purchase descriptions and reading the trace, not asserted in an automated test.

## Verification

Manual: run `backend/phase2_langchain/agent.py` against the same three sample queries used in Phase 1 (BigBasket groceries, IndiGo flight, Croma gadget) and confirm the recommendations match what Phase 1 produced for the same queries, since the underlying data and tools are identical, only the calling mechanism changed.

## Open questions

None outstanding. All prior questions in this design conversation have been resolved.
