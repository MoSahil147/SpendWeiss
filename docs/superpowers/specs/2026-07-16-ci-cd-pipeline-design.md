# SpendWeiss: CI pipeline with PR gated auto-merge design

Date: 2026-07-16
Status: approved, awaiting spec review

## Purpose

Give SpendWeiss a real continuous integration gate on `main`, so that merging a phase branch no longer depends on the user manually reviewing and clicking merge. Every change to `main` goes through a pull request, GitHub Actions runs the automated test suite against it, and once that check is green the user arms GitHub's native auto-merge toggle once per PR and GitHub takes it from there.

This also reverses an earlier decision. The Phase 0 and Phase 1 design explicitly ruled out an automated test suite, verification was manual only. For CI to be a meaningful gate rather than a formality, that non-goal is dropped. A real pytest suite is introduced, starting with the data already on `main` from Phase 0, and extended as Phase 1 code is rebuilt.

## Goals

- A GitHub Actions workflow that runs the pytest suite on every pull request targeting `main`.
- A pytest suite under `backend/tests/` covering the Phase 0 mock data (`cards.json`, `offers.json`, `transactions.json`) and, once rebuilt, the Phase 1 tool functions (`check_card_rewards`, `check_offers`).
- Branch protection on `main`: no direct pushes, a pull request is required, and the CI check must pass before the pull request can be merged.
- The repository setting that allows auto-merge to be armed on a pull request.

## Non goals

- No linting (ruff or otherwise). Not requested, would be scope creep here.
- No fully automatic merging without a human arming it. The user arms GitHub's built in auto-merge toggle once per pull request; after that it is hands off.
- No changes to deployment. This is CI only, not CD in the sense of shipping to Render or Netlify, those arrive in Phase 8.

## Components

### 1. `.github/workflows/ci.yml`

Triggers on `pull_request` events targeting `main`. Steps: check out the repository, install `uv` via the `astral-sh/setup-uv` action, run `uv sync` inside `backend/`, then run `uv run pytest` inside `backend/`. The job must fail if any test fails, which is what makes it usable as a required status check.

### 2. `backend/tests/`

A pytest suite, with `pytest` added as a development dependency via `uv add --dev pytest`.

- `backend/tests/test_data.py`: loads the three JSON files and asserts on their shape and cross references, for example that there are six cards, six offers, twenty four transactions, that every offer's `card_id` refers to an existing card, that every card is referenced by exactly one offer, and that every category used in `transactions.json` and in each card's `reward_rates` is one of the six known categories.
- `backend/tests/test_tools.py`: added once Phase 1's `tools.py` is rebuilt. Tests `check_card_rewards` and `check_offers` directly, replacing the manual `python -c` verification steps in the existing Phase 0 and 1 plan. This file is out of scope for the plan that follows this spec; it belongs to the Phase 1 rebuild, which is tracked separately.

### 3. Branch protection on `main`

Configured via the GitHub CLI (`gh api`) rather than the web UI, so the exact change is visible and reviewable. Requires: pull requests only, no direct pushes to `main`; the CI workflow's job listed as a required status check; the branch must be up to date with `main` before merging is allowed.

### 4. Repository setting: allow auto-merge

A repository level setting, also set via `gh api`, that makes the "Enable auto-merge" option available on pull requests. Without it, GitHub hides the toggle entirely.

## Workflow after this ships

1. Work happens on a phase branch, as it does today.
2. A pull request is opened against `main`.
3. The CI workflow runs `pytest` automatically.
4. The user clicks "Enable auto-merge" (squash) on the pull request, once, whenever they are ready.
5. The moment the CI check goes green, GitHub squashes and merges the pull request into `main` on its own, no further action needed.
6. If CI fails, auto-merge does not fire, the pull request sits open for the user to look at.

## Risk and confirmation

Branch protection rules and the auto-merge repository setting are changes to shared GitHub configuration, not local files. Both will be applied with `gh api` commands shown to the user before they run, consistent with how any other change to shared infrastructure is handled in this project.

## Verification

- A pull request opened from a throwaway branch with a passing test triggers the CI workflow and shows a green check.
- A pull request opened from a throwaway branch with a deliberately failing test shows a red check, and the branch protection rule blocks merging.
- The "Enable auto-merge" option is visible and armable on a pull request against `main`.

## Open questions

None outstanding. All prior questions in this design conversation have been resolved.
