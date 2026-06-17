# BatchGen PR Merge Policy Contract

> **This is a binding contract.** Every pull request targeting `main` must satisfy
> it before it can be merged — no exceptions by authorship. It applies **identically**
> to AI coding agents and to human contributors. Most PRs in this repo are
> machine-authored; this contract exists to keep the file changes in those PRs clean,
> in-scope, and reviewable.
>
> A green CI run does **not** waive any rule here. The reviewer still checks every
> item below, and the gate in [§8](#8-ci-enforcement) mechanically enforces the
> high-precision subset.

## Contents

- [§0. Scope](#0-scope)
- [§1. File hygiene — what must not be in a PR](#1-file-hygiene)
- [§2. Change scope, placement & permitted edits](#2-placement)
- [§3. Diff discipline](#3-diff-discipline)
- [§4. Commit & branch hygiene](#4-commit--branch-hygiene)
- [§5. Author pre-merge checklist](#5-author-checklist)
- [§6. Reviewer gate](#6-reviewer-gate)
- [§7. Merge authority & blockers](#7-merge-authority)
- [§8. CI enforcement](#8-ci-enforcement)
- [§9. Override authority (hotfix bypass)](#9-override-authority)
- [§10. After merge](#10-after-merge)

---

<a name="0-scope"></a>
## §0. Scope

- Applies to **every** PR into `main`, regardless of author (agent or human) or access level.
- "**Production packages**" means the importable, shipped code: `batchgen/`, `batchgen_kernels/`,
  `core/`, `server/`, `op_builder/`, plus the repository root.
- The contract governs what the PR's **diff and commit history look like** — facts visible in
  the PR itself. It does not govern how you develop locally.
- Where this contract and the prose in `CONTRIBUTING.md` overlap, **this contract is
  authoritative**; `CONTRIBUTING.md` is the friendly on-ramp and links here.

---

<a name="1-file-hygiene"></a>
## §1. File hygiene — what must not be in a PR

A PR must not **add** any of the following. (The gate only fails on **newly-added** lines/files,
so pre-existing instances on `main` don't block PRs; they are cleaned up — relocated, not
deleted — on their own PRs.)

1. **Throwaway scratch / debug scripts in production packages or repo root.**
   No new files matching `debug_*.py`, `check_*.py`, `scratch_*.py`, `tmp_*.py`, or
   `*_scratch.py` under any production package or at the repo root. **Benchmark scripts
   (`bench_*.py`) are exempt** — they are useful reference and may live alongside the kernels
   they exercise.
2. **Tests interleaved in the runtime package.** No new `test_*.py` under the production
   packages. Tests live in the repository's `tests/` tree.
3. **Server-side env-var debug guards.** Do not add `os.environ.get("BATCHGEN_…")` /
   `os.getenv("BATCHGEN_…")` guards to gate debug or diagnostic behavior in production code;
   represent debug behavior as a batch-level `/v1/batches` `batchgen_debug` flag (togglable per
   job, no restart). Any env-guard added while developing **must be removed before the PR is
   merged**. A genuinely-needed **new environment variable** is a deliberate config change —
   introduce it in its **own dedicated PR**, never bundled into a feature/fix. (The only
   sanctioned env-flag module is `batchgen/timing.py`.)
4. **Raw `print()` left in production paths** (decode, prefill, tokenize, MoE, attention,
   scheduler, server loops) — use a named logger instead. `logging.getLogger(__name__).debug(...)`
   is fine; it is a gated logger, not stray output. Remove any `print()` debugging before the PR
   merges.
5. **Commented-out / dead code** introduced by the change (e.g. `# print(request)`). Delete it
   instead of commenting it out.
6. **Generated artifacts.** No `*.log`, profiler traces (`*.nsys-rep`, `*.ncu-rep`,
   `*.sqlite`), benchmark CSVs, checkpoints / weights (`*.pt`, `*.pth`, `*.safetensors`,
   `*.bin`), wheels (`*.whl`), or compiled extensions. Never `git add -f` past `.gitignore`.
7. **Loose utilities at the repository root.** The only `.py` permitted at root is `setup.py`.
   Tooling belongs in `scripts/`.

Every line H1–H7 is mechanically checkable; see [§8](#8-ci-enforcement).

---

<a name="2-placement"></a>
## §2. Change scope, placement & permitted edits

### Placement

1. **Tests** → `tests/`. **Demo / example scripts** → `examples/`. **Docs** → `docs/`.
   Do not scatter these into the runtime package.
2. **Module docs stay current.** If a PR adds, removes, or changes a module's public API or
   entry points, update that module's `MODULE.md` in the **same** PR. Modules that carry a
   `MODULE.md`: `moe/`, `attention/`, `attention/mla/`, `kv_cache/`, `scheduler/`, `server/`,
   `models/`, `cuda_graph/`, `quantization/`.
3. **New files go where their kind already lives.** Match the directory of the nearest
   existing file of the same kind; do not invent new top-level directories without reviewer
   sign-off.

### §2.4 Change types

Every PR has **exactly one change type**, declared via the checkbox in
`.github/PULL_REQUEST_TEMPLATE.md`. The type fixes the set of files the PR may touch (§2.5).
A change that spans two types must be **split into two PRs** (§3.2).

| Type | Meaning |
|------|---------|
| `model`  | Add or extend support for a model. |
| `kernel` | Add or optimize a compute kernel. |
| `core`   | Change the scheduling / serving / runtime **scaffolding** itself. |
| `fix`    | A narrow bug fix. |
| `infra`  | Build, CI, packaging, scripts, Docker. |
| `docs`   | Documentation only. |

### §2.5 Permitted file changes per type

A PR may modify files **only** within its type's permitted set. Touching anything outside it
is a merge blocker (§7) — for a `model` or `kernel` PR that reaches into the scaffolding layer
(§2.6) this is a **hard** block, clearable only by a §9 owner override (and even then the
scaffolding change should be split out).

| Type | May touch — and nothing else | Notable off-limits |
|------|------------------------------|--------------------|
| `model`  | `batchgen/models/**`; the **registration seam** `batchgen/get_initializer.py`, `batchgen/get_parallel_strategy_manager.py`, `batchgen/models/__init__.py`; model-specific kernels under `batchgen_kernels/**`; `configurations/**`; `tests/**`; `docs/**` | the scaffolding layer (§2.6); **shared** in-tree kernels (`batchgen/moe/`, `batchgen/attention/`, …) — split into a `kernel` PR |
| `kernel` | `batchgen_kernels/**`; in-tree kernel dirs `batchgen/moe/`, `batchgen/attention/`, `batchgen/gemm/`, `batchgen/quantization/`, `batchgen/triton_kernels/`, `batchgen/other_kernels/`; `op_builder/`, `batchgen/op_builder/`; `tests/**`; `docs/**` | the scaffolding layer (§2.6); `batchgen/models/**` forward logic — split into a `model` PR |
| `core`   | the scaffolding layer (§2.6) and whatever it requires — **the only type that may** | still bound by §1 hygiene and §3 single-concern |
| `fix`    | the files implicated by the bug **plus a regression test** under `tests/**` | unrelated subsystems; a fix that rewrites scaffolding is a `core` PR |
| `infra`  | `.github/**`, `scripts/**`, `docker/**`, `Makefile`, `setup.py`, `setup.cfg`, `pyproject.toml`, `MANIFEST.in` | application code under `batchgen/**`, `batchgen_kernels/**` |
| `docs`   | `docs/**`, `*.md`, `README*` | any code |

### §2.6 The scaffolding layer is off-limits except to `core`

The **scaffolding layer** is the scheduling / serving / runtime core. **Only a `core` PR may
modify it.** A `model` or `kernel` PR that touches any of it fails (§7).

- Top-level `batchgen/*.py` runtime modules: `batchgen_worker.py`, `server_worker_main_loop.py`,
  `continuous_batching.py`, `scheduler.py`, `task_scheduler.py`, `decode.py`, `decode_task.py`,
  `prefill.py`, `prefill_task.py`, `sampling.py`, `sequence.py`, `migration.py`,
  `pd_orchestrator.py`, `query_book.py`, `query_manager.py`, `parameter_server.py`,
  `parameter_server_client.py`, `node_manager.py`, `inference_runtime.py`, `batch_inference.py`,
  `batch_order.py`, `batchgen_server.py`, `batchgen_server_dev.py`, `batchgen_client.py`,
  `client_optimized.py`, `launch_server.py`, `launch_http_server.py`, `entrypoint.py`,
  `lifespan.py`, `generate.py`, `model_instance.py`.
- Subpackages: `batchgen/scheduler/`, `batchgen/server/`, `batchgen/kv_cache/`,
  `batchgen/worker/`, `batchgen/sequence_manager/`, `batchgen/distributed/`,
  `batchgen/planner/`, `batchgen/cuda_graph/`, `batchgen/core/`.

The **registration seam** (`batchgen/get_initializer.py`, `batchgen/get_parallel_strategy_manager.py`,
`batchgen/models/__init__.py`) is **not** scaffolding — a `model` PR adds its dispatch branch
there, and that is the only top-level `batchgen/` edit a new model needs.

> Enforced for now by the reviewer (§6) and the declared type in the PR template. A path-scope
> CI check (assert the changed-file set ⊆ the declared type's permitted paths) is a planned
> addition to the §8 gate; per-layer `CODEOWNERS` is intentionally not used for this.

---

<a name="3-diff-discipline"></a>
## §3. Diff discipline

1. **Surgical changes only.** Every changed line must trace to the PR's stated task or the
   dispatched spec's acceptance criteria. No drive-by refactors, renames, reformatting, or
   import reordering in unrelated files.
2. **One concern per PR.** Do not mix, e.g., a kernel optimization, a new model integration,
   and benchmark-harness changes in one PR. Split unrelated concerns into separate PRs.
3. **Simplicity first.** No speculative abstractions, configuration knobs, or error handling
   for impossible cases. If it can be smaller, make it smaller.
4. **Fallback paths announce themselves.** Any new path that falls back from an optimized
   kernel to a reference implementation must emit a one-shot `warning_once`-style log on first
   invocation, so a silent regression to the slow path is visible.

---

<a name="4-commit--branch-hygiene"></a>
## §4. Commit & branch hygiene

1. **No `Co-Authored-By` trailers** in any commit.
2. **Stage files by name.** Do not use `git add -A` or `git add .`; a PR must not contain files
   the author did not intend to stage.
3. **No direct commits to `main`.** Work on a branch named `<handle>/<short-description>`
   (e.g. `tairan/query-book-buffer-pool`); dispatched contributors use the dispatcher-assigned
   `<assignee>/<slug>` branch. Open a PR from that branch.
4. **Angular commit format.** Header `<type>(<scope>): <summary>`; `<type>` is
   `build | ci | docs | feat | fix | perf | refactor | test | chore`; `<scope>` (optional) is a
   subsystem (`moe`, `attention`, `kv_cache`, `scheduler`, `glm5`, …). A body is mandatory for
   every type except `docs` and must be ≥ 20 characters. The PR **title** follows the same format.
5. **Sync via git, never file copy.** Do not move files into any checkout of this repo with
   `scp` / `cp` / `rsync`. Use `git push` + `git pull`.

### PR title & message conventions

The title says the **nature** of the change (`<type>`); the **Change Type** checkbox (§2.4) says
the **subsystem scope** and governs which paths you may touch. They are independent — set both.

| Title part | Rule | Example |
|------------|------|---------|
| `type` | `feat` · `fix` · `perf` · `refactor` · `docs` · `test` · `build` · `ci` · `chore` | `perf` |
| `scope` *(optional)* | subsystem or model | `(moe)` |
| `summary` | imperative mood, lowercase start, **no trailing period**, ≤ 72 chars | `fuse grouped-gemm act-quant` |

Title don'ts: no PR number in the title, no `WIP:` (use Draft state), no `Co-Authored-By` anywhere.

**Body sections:** **What** (one paragraph) · **Why** (motivation; link `close #123`) · **Milestone**
*(optional)* · **Type of Change** (tick one, §2.4) · **File changes** (one row per changed/intended
file with a Δ and a note, so scope can be audited against §2.5 at the **draft** stage — before the
diff exists) · **Checklist** (§5). **Test execution is owned by the PR CI**
(maintainer-triggered GPU / MMLU run via the `ci:run` label) — its accuracy levels, baselines, and
harness are internal core-team infrastructure. Do **not** paste test commands, accuracy tables, or
benchmark logs into the PR body; the green CI check is the record. (Author-side unit tests still
belong under `tests/` — see the §5 checklist.)

**Title `type` ↔ Change Type, by example:**

| Change | Title | Change Type |
|--------|-------|-------------|
| New model | `feat: add glm-6 model support` | `model` |
| Kernel speedup | `perf(moe): fuse grouped-gemm act-quant` | `kernel` |
| Scheduler bug | `fix(scheduler): fix decode batch admission` | `core` |
| Narrow bug | `fix: handle empty prompt in tokenizer` | `fix` |
| CI / build | `ci: add PR file-hygiene check` | `infra` |
| Docs only | `docs: document GLM-5 FP8 setup` | `docs` |

---

<a name="5-author-checklist"></a>
## §5. Author pre-merge checklist

Run this before marking a PR **Ready for review**. It is reproduced in
`.github/PULL_REQUEST_TEMPLATE.md`; tick every box.

- [ ] `git diff --stat origin/main` reviewed; **every** file traces to the task — no unrelated files.
- [ ] **Change type** declared in the PR template; all changed files are within that type's permitted set (§2.5) — no scaffolding edits in a `model`/`kernel` PR (§2.6).
- [ ] No `debug_* / check_* / scratch_* / tmp_*` scripts added to a production package or root (§1.1). (`bench_*` is allowed.)
- [ ] No `test_*.py` added inside the runtime package; tests are under `tests/` (§1.2, §2.1).
- [ ] No new `BATCHGEN_*` env-var debug guard; debug behavior is a `batchgen_debug` batch flag (§1.3). A genuinely-needed new env var goes in its own PR.
- [ ] No leftover `print()` or commented-out code in the changed files (§1.4–§1.5). (`logging.debug()` is fine.)
- [ ] No logs, profiler traces, CSVs, checkpoints, wheels, or compiled artifacts staged; nothing added with `-f` (§1.6).
- [ ] Touched modules' `MODULE.md` updated if the public API changed (§2.2).
- [ ] One concern only; diff is surgical (§3).
- [ ] Commits are clean: Angular format, body present, **no `Co-Authored-By`** (§4).
- [ ] CI is green (format, lint, hygiene, tests).

---

<a name="6-reviewer-gate"></a>
## §6. Reviewer gate

The reviewer **approves only when all** of the following hold:

1. The diff matches the task / spec acceptance criteria and nothing beyond it.
2. No §1 hygiene violation appears in the added lines.
3. §2 placement is satisfied and any required `MODULE.md` update is included.
4. §3 scope is satisfied — single concern, surgical, no speculative complexity.
5. §4 commit hygiene is satisfied.
6. CI is green and the §8 hygiene check has passed (or been overridden per §9 with a recorded reason).

A reviewer must **Request changes** — not "Approve with a note" — for any unmet item.
Contributors with `Write` access may Approve / Request changes but must **not** press Merge
(see §7).

---

<a name="7-merge-authority"></a>
## §7. Merge authority & blockers

- **Only the project owner (`@Andrewxu313`) presses Merge.** Write-access contributors,
  including members of `batchgen-core`, open PRs, push, and review, but never self-merge —
  not even their own PRs. Direct pushes to `main` are the owner's alone.
- A PR requires **≥ 1 approving review from a Code Owner** (per `.github/CODEOWNERS`) and
  **green CI** before the owner merges.
- A merge is **blocked** by any of: failing CI; a failing §8 hygiene check without a §9
  override; a missing Code Owner approval; any §1 hygiene violation; changed files outside the
  declared change type's permitted set (§2.5) — including **any** scaffolding edit in a
  `model`/`kernel` PR (§2.6); misplaced files or a missing required `MODULE.md` (§2); an
  out-of-scope or multi-concern diff (§3); a `Co-Authored-By` trailer or other §4 violation in
  the commit history; unresolved review comments; or the PR still being in **draft**.
- **Release PRs** additionally must be clean and mergeable: rebase/merge current `main` first
  and exclude failed CI / Docker-rescue commits, so GitHub does not report a conflict.

---

<a name="8-ci-enforcement"></a>
## §8. CI enforcement

The high-precision subset of §1 and §4 is enforced by the **PR File Hygiene** check
(`.github/workflows/pr-file-hygiene.yml`, scanning logic in
`.github/workflows/scripts/check-pr-hygiene.sh`). It inspects **only the added lines / new
files** in the PR relative to the base branch, so it never trips on legacy code.

**Blocking checks** (fail the PR): H1 throwaway scratch/debug scripts in a package or root (`bench_*` exempt); H2
`test_*.py` in a package; H6 generated artifacts; H7 loose root `.py`; and a `Co-Authored-By`
trailer in any commit in the PR range.

**Advisory checks** (annotate, never block — left to reviewer judgement): newly-added
`BATCHGEN_*` env guards in production code (§1.3) and newly-added `print()` in production
paths (§1.4). These are fuzzy to detect precisely; the reviewer makes the call under §6.

**Change-type boundaries (§2.4–§2.6)** are enforced for now by the reviewer (§6) and the type
declared in the PR template. A path-scope check — assert the changed-file set is a subset of
the declared type's permitted paths, and fail a `model`/`kernel` PR that touches the
scaffolding layer — is a planned addition to this gate; it is straightforward to add to
`check-pr-hygiene.sh` once the type taxonomy in §2.4 is settled. Per current decision, per-layer
`CODEOWNERS` is **not** used for this boundary.

**Rollout — the gate becomes *required* only after the pre-existing violations on `main` are
cleared.** Until then the workflow runs in **report-only** mode (it posts results but does not
block), so existing violations do not wall off every PR. Flipping it to blocking is two steps:

1. Set `HYGIENE_ENFORCE: "1"` in `pr-file-hygiene.yml` (the script exits non-zero on a blocking
   violation only when this is set).
2. Add **PR File Hygiene** to the `main` branch-protection **required status checks**.

Do **not** perform either step until the cleanup is complete.

**Local pre-check.** Anyone (or any agent) can run the same logic before pushing:

```bash
bash .github/workflows/scripts/check-pr-hygiene.sh origin/main
```

**Sanctioned per-line exceptions.** A genuinely-needed line that trips an advisory check may
carry a trailing `# noqa: hygiene` comment; the file allowlist at the top of
`check-pr-hygiene.sh` (currently `batchgen/timing.py`) exempts sanctioned env-flag modules.
Exceptions are reviewer-approved, not author-asserted.

---

<a name="9-override-authority"></a>
## §9. Override authority (hotfix bypass)

Sometimes a hotfix must land before the cleanup is complete or despite a hygiene failure. Override
authority is **strictly ranked** and every use is **logged**:

| Layer | Who | Mechanism | When |
|-------|-----|-----------|------|
| Per-line | Author, reviewer-approved | `# noqa: hygiene` / file allowlist in `check-pr-hygiene.sh` | A single sanctioned line (e.g. a `timing.py` env flag) |
| Whole-PR | **Maintainer** (`batchgen-core`) | Apply the `ci:skip-hygiene` label; the workflow then passes the hygiene check | An urgent PR where the hygiene finding is known and accepted |
| Hard backstop | **Owner only** (`@Andrewxu313`) | Repo-admin branch-protection bypass (merge despite a failing required check) | True emergency; no time to relabel or amend |

Rules that bound every override:

1. **Precedence is owner > maintainer > contributor.** A contributor may never override their
   own PR's gate; only a *different* maintainer may apply `ci:skip-hygiene`.
2. **Every whole-PR or backstop override requires a recorded reason** in the PR description as a
   line `Override-Reason: <why>`, **and** a follow-up cleanup issue linked in the same PR.
3. **An override skips the gate, not the contract.** The reviewer still owns §6; merge authority
   is still the owner's alone (§7).
4. Configure the GitHub side as: hygiene check = required status check on `main`; "Allow
   specified actors to bypass required pull requests" / admin bypass = **owner only**; the
   `ci:skip-hygiene` label = creatable/appliable by maintainers only. This makes GitHub, not the
   honor system, enforce the ranking.

---

<a name="10-after-merge"></a>
## §10. After merge

1. Delete the PR branch once merged.
2. When the pre-existing violations are all cleared, perform the §8 rollout to make the gate
   blocking.
