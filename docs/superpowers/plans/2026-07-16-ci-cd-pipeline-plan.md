# CI Pipeline with PR Gated Auto-Merge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give SpendWeiss a real CI gate on `main`: a pytest suite covering the Phase 0 mock data, a GitHub Actions workflow that runs it on every pull request, branch protection that requires that check to pass, and the repository setting that allows GitHub's native auto-merge toggle to be armed.

**Architecture:** `pytest` added as a `uv` dev dependency, tests live under `backend/tests/`. `.github/workflows/ci.yml` runs `uv sync` and `uv run pytest` on every pull request targeting `main`. Branch protection and the auto-merge repository setting are applied with `gh api`, each shown to the user before it runs.

**Tech Stack:** `pytest`, GitHub Actions, `astral-sh/setup-uv` action, `gh` CLI (already authenticated as `MoSahil147`, repo is `MoSahil147/Spendweiss`, default branch `main`).

## Global Constraints

- All prose is British English, no em dashes.
- Code carries explanatory comments in British English (this reverses an earlier "no comments" rule; see `JOURNAL.md`, 2026-07-16). The `test_data.py` and `ci.yml` content already written into this plan predates that change and stays uncommented; it is short and self-explanatory enough not to need retrofitting, but new code from here on should be commented.
- Do not run `git push`, open a pull request, or run any `gh api` command that changes repository settings without first showing the user the exact command and getting explicit confirmation. This is stricter than the general "no add or push" rule already in place: even with a plan approved, these specific actions pause for a live confirmation each time, per the spec's risk section.
- Local file creation, `git add`, local commits, and running `pytest` locally do not need this pause, they are reversible and local.
- Actually, per the user's standing instruction from earlier in this project, do not run `git add` or `git commit` either, unless the user asks. Leave everything unstaged until told otherwise. This plan only stages nothing; it writes files and runs local verification commands.

---

### Task 1: Add pytest and write the data test suite

**Files:**
- Modify: `backend/pyproject.toml` (via `uv add --dev pytest`)
- Create: `backend/tests/__init__.py`
- Create: `backend/tests/test_data.py`

**Interfaces:**
- Produces: a `backend/tests/` package that `uv run pytest` discovers automatically from inside `backend/`. No other task depends on this one's internals, but Task 2's workflow depends on `uv run pytest` succeeding here.

- [ ] **Step 1: Add pytest as a dev dependency**

```bash
cd backend && uv add --dev pytest
```
Expected: `pytest` added under `[dependency-groups]` (or `[tool.uv.dev-dependencies]` depending on `uv` version) in `backend/pyproject.toml`, and installed into `.venv`.

- [ ] **Step 2: Create the test package marker**

Create `backend/tests/__init__.py`, empty file.

- [ ] **Step 3: Write the failing tests first**

Create `backend/tests/test_data.py`:

```python
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CATEGORIES = {"groceries", "dining", "travel", "online_shopping", "fuel", "other"}


def load_json(filename):
    with open(DATA_DIR / filename) as data_file:
        return json.load(data_file)


def test_six_cards():
    cards = load_json("cards.json")
    assert len(cards) == 6


def test_six_offers():
    offers = load_json("offers.json")
    assert len(offers) == 6


def test_twenty_four_transactions():
    transactions = load_json("transactions.json")
    assert len(transactions) == 24


def test_every_card_has_exactly_one_offer():
    cards = load_json("cards.json")
    offers = load_json("offers.json")
    card_ids = {card["id"] for card in cards}
    offer_card_ids = [offer["card_id"] for offer in offers]
    assert set(offer_card_ids) == card_ids
    assert len(offer_card_ids) == len(set(offer_card_ids))


def test_card_reward_categories_are_known():
    cards = load_json("cards.json")
    for card in cards:
        assert set(card["reward_rates"].keys()) == CATEGORIES


def test_transaction_categories_are_known():
    transactions = load_json("transactions.json")
    for transaction in transactions:
        assert transaction["category"] in CATEGORIES


def test_transaction_cards_exist():
    cards = load_json("cards.json")
    transactions = load_json("transactions.json")
    card_ids = {card["id"] for card in cards}
    for transaction in transactions:
        assert transaction["card_used"] in card_ids
```

- [ ] **Step 4: Run the tests and confirm they pass**

```bash
cd backend && uv run pytest tests/test_data.py -v
```
Expected: 7 tests, all `PASSED`. If any fail, the underlying data file has a real inconsistency, fix the data file, not the test, since the test encodes what Task 2 of the Phase 0 plan already established as true.

---

### Task 2: Write the GitHub Actions workflow

**Files:**
- Create: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: `backend/pyproject.toml` and `backend/tests/` from Task 1.
- Produces: a workflow whose job name is referenced by name in Task 3's branch protection configuration. The job id used here is `test`, and it must match exactly what gets passed to `gh api` in Task 3.

- [ ] **Step 1: Write `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  pull_request:
    branches:
      - main

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync
        working-directory: backend
      - run: uv run pytest
        working-directory: backend
```

- [ ] **Step 2: Validate the YAML locally**

```bash
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml')); print('valid yaml')"
```
Expected output: `valid yaml`. If `PyYAML` is not available in the system Python, install it into a throwaway venv rather than skipping this check: `python3 -m venv /tmp/yamlcheck && /tmp/yamlcheck/bin/pip install pyyaml -q && /tmp/yamlcheck/bin/python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml')); print('valid yaml')"`.

This workflow cannot be fully exercised until it exists on a branch with an open pull request against `main`, which is why Task 4 covers end to end verification after Task 3's branch protection is in place.

---

### Task 3: Configure branch protection on main

**Files:** none, this task only runs `gh api` commands against the repository's settings.

**Interfaces:**
- Consumes: the job id `test` from Task 2's workflow, which must appear in the `contexts` list below exactly as written there.

- [ ] **Step 1: Show the user the exact command before running it**

State to the user: this will configure branch protection on `main` for `MoSahil147/Spendweiss`, requiring the `test` status check to pass and disallowing direct pushes. Wait for explicit confirmation before running Step 2.

- [ ] **Step 2: Apply branch protection**

```bash
gh api repos/MoSahil147/Spendweiss/branches/main/protection \
  --method PUT \
  -H "Accept: application/vnd.github+json" \
  -f "required_status_checks[strict]=true" \
  -f "required_status_checks[contexts][]=test" \
  -F "enforce_admins=true" \
  -F "required_pull_request_reviews=null" \
  -F "restrictions=null" \
  -F "allow_force_pushes=false" \
  -F "allow_deletions=false"
```
Expected: a JSON response describing the new protection rule, HTTP 200. If this returns a 404 or a permissions error, the authenticated `gh` account does not have admin rights on the repository, stop and report this to the user rather than trying workarounds.

- [ ] **Step 3: Verify**

```bash
gh api repos/MoSahil147/Spendweiss/branches/main/protection --jq '.required_status_checks.contexts'
```
Expected output: `["test"]`

---

### Task 4: Enable auto-merge on the repository and verify the whole pipeline

**Files:** none for Step 1. A throwaway file is created and removed as part of the end to end verification in Step 3.

**Interfaces:**
- Consumes: Task 1's tests, Task 2's workflow, and Task 3's branch protection, exercised together for the first time here.

- [ ] **Step 1: Show the user the exact command before running it**

State to the user: this will turn on the "Allow auto-merge" repository setting for `MoSahil147/Spendweiss`, which is what makes the "Enable auto-merge" button appear on pull requests. Wait for explicit confirmation before running Step 2.

- [ ] **Step 2: Enable auto-merge**

```bash
gh api repos/MoSahil147/Spendweiss --method PATCH -F allow_auto_merge=true
```
Expected: JSON response with `"allow_auto_merge": true`.

- [ ] **Step 3: Show the user the exact commands before running them**

State to the user: this will push the current branch (`phase-1-raw-react`, which already contains Task 1 and Task 2's new files once committed) and open a real pull request against `main` to prove the pipeline end to end, since GitHub Actions cannot run without a real pull request. Wait for explicit confirmation before running Step 4. Confirm with the user first whether Tasks 1 and 2's new files should be committed for this, since the project's standing instruction has been to leave changes unstaged.

- [ ] **Step 4: Commit, push, and open the pull request**

```bash
git add backend/pyproject.toml backend/uv.lock backend/tests/ .github/workflows/ci.yml
git commit -m "Add CI pipeline: pytest suite and GitHub Actions workflow"
git push -u origin phase-1-raw-react
gh pr create --title "CI: pytest suite and GitHub Actions workflow" --body "Adds a real pytest suite for the Phase 0 mock data, a GitHub Actions workflow that runs it on every pull request, branch protection requiring that check, and the auto-merge repository setting."
```
Expected: the PR is created, its URL is printed. GitHub Actions starts running the `test` job automatically.

- [ ] **Step 5: Confirm the check goes green**

```bash
gh pr checks --watch
```
Expected: the `test` check reports `pass` within a couple of minutes. If it reports `fail`, read the workflow log with `gh run view --log-failed` and fix the underlying issue (most likely a `uv sync` or `pytest` failure that did not reproduce locally, for example a Python version mismatch between the workflow's `ubuntu-latest` image and the local machine) before proceeding.

- [ ] **Step 6: Arm auto-merge and confirm it merges**

```bash
gh pr merge --auto --squash
```
Expected: GitHub reports auto-merge is enabled for the PR. Once Step 5's check is already green, GitHub merges immediately. Confirm with:
```bash
gh pr view --json state,mergedAt
```
Expected: `"state": "MERGED"` with a non-null `mergedAt`.

- [ ] **Step 7: Update the journal**

Append to `JOURNAL.md`, filling in the template with what happened in this task (in particular, whether branch protection or auto-merge needed any adjustment, and what the first real CI run showed):

```
## CI pipeline: pytest, GitHub Actions, branch protection, auto-merge (2026-07-16)

**What I built:**

**Key decisions:**

**Gotchas and bugs hit:**

**What I learned:**

**Next up:**
```
