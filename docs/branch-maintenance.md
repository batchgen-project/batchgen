# Branch Maintenance: Audit & Cleanup Plan

> **Snapshot:** 2026-07-14 (UTC), measured against `origin/main` (`d171a154`).
> **Regenerate before acting** — branch state drifts as PRs merge. This is a point-in-time report, not a live view.

This document audits every branch (local and remote) and classifies it by how its work relates to `main`, then lists a concrete, staged deletion plan. It also scans for **dead/stale** branches by last-commit age.

> [!IMPORTANT]
> All measurements are against **`origin/main`**, **not** the local `main` branch. At snapshot time local `main` was **104 commits behind** `origin/main`; auditing against it would misclassify almost everything. Run `git fetch --prune origin` before regenerating.

---

## TL;DR

| Scope  | Linear-merged | Squash-merged | Unmerged | Total |
|--------|--------------:|--------------:|---------:|------:|
| Remote | 8 | 107 | 98 | 213 |
| Local  | 1 | 13 | 6 | 20 |

- **Safe to delete now:** all **Linear-merged** + **Squash-merged** branches (their work is already in `main`).
- **Keep:** all **Unmerged** branches — they contain unique work not in `main`.
- **Dead/stale flag:** **12** unmerged branches are >180 days old (abandoned), **25** are 90–180 days old. These need an **owner decision**, never auto-deletion.
- This PR **does not delete anything** — it documents the plan for owner review (per the owner-only merge policy below).

---

## Scope & Method

Each branch tip is compared to `origin/main` and placed in exactly one class:

| Class | Definition | Detection | Safe to delete? |
|-------|------------|-----------|-----------------|
| **Linear-merged** | Branch tip **is an ancestor** of `origin/main` — every commit is present in `main`'s history, unchanged. Full linear history preserved. | `git merge-base --is-ancestor <branch> origin/main` | **Yes** — `git branch -d` accepts it. |
| **Squash-merged** | Branch's **net diff** is already in `main`, but as a rewritten (squashed/rebased) commit with a different SHA. Individual commits are **not** linearly in `main`. | Synthetic full-tree commit on the merge-base, then `git cherry origin/main <synthetic>` patch-id match (equiv. to `git-delete-merged-branches --effort=3`). | **Yes**, but `git branch -d` **refuses** it — `-D` required. |
| **Unmerged** | Has commits whose patch is **not** in `main`. | Neither of the above. | **No** — deleting loses work. |

**Dead/stale** is an orthogonal, age-based axis applied to **unmerged** branches (committer-date of the tip):
- **Dead** — no commits in **>180 days** (abandoned).
- **Stale** — no commits in **90–180 days**. (90 days = GitHub's own "Stale branches" UI threshold.)

---

## Policy Context

- **`PR_MERGE_POLICY.md` § 4 (Commit & branch hygiene)** already mandates branch discipline: work on `<handle>/<short-description>` branches, no direct commits to `main`, and — most relevant here — **"Delete the PR branch once merged."** This document **operationalizes** that rule for the accumulated backlog of branches that were merged but never deleted; it does not introduce a new policy.
- **`CONTRIBUTING.md` § Merge Policy** governs *who* merges (owner-only merge button, ≥1 approving review, green CI; direct pushes to `main` reserved for the owner).
- Neither doc pins a merge *method*, but the squash-merge counts below reflect a de-facto **squash-merge** workflow (even `chore/contributing-merge-policy` and `chore/cleanup-for-migration` are squash-merged into `main`).
- There is **no stale-branch automation** in `.github/workflows/` today, so this cleanup is manual. A companion PR proposes a report-only stale-branch workflow to enforce §4's "delete once merged" going forward.

---
## 1. Linear-merged — full linear history (SAFEST to delete)

Tip is a true ancestor of `origin/main`; every commit is already in `main`. `git branch -d` accepts these.

**Local (1):** `main` — an ancestor of `origin/main` but **do not delete**; it is your default branch, just stale (104 behind). Fast-forward it instead: `git fetch origin && git branch -f main origin/main` (when not checked out).

**Remote (8):**
- `ci/sglang-release-test`
- `release/v1.0.10.post3-glm5-stability`
- `tairan/adapt_Pynccl_Comm`
- `tairan/cold_start_opt`
- `tairan/fix_posix_shm`
- `tairan/fix_stateless_comm_init`
- `tairan/full-dsa-cuda-graph-prod`
- `tairan/urgent_fix`

---

## 2. Squash-merged — work in `main`, history rewritten (SAFE, but not linear)

The branch's net diff is already in `main` under a different (squashed) SHA. Safe to delete, but `git branch -d` will **refuse** them (it only sees ancestor reachability), so `-D` is required. Split by confidence:

### 2a. High confidence — single-commit squash (`ahead == 1`)

One original commit, cleanly squashed into `main`. Deletion is unambiguous.

**Local (13):**
- `docs/async-batch-submission-fix`
- `docs/batch-api-cancel-flow`
- `docs/glm51-enable-thinking-fix`
- `docs/install-access-note`
- `docs/install-md-accuracy-fixes`
- `docs/install-md-kernel-count-23`
- `docs/server-flags-gaps`
- `fix/fa2-backend-flash-attn-import-guard`
- `fix/kernels-explicit-gencode`
- `fix/version-fallback-post4`
- `infra/dockerfile-shell-pipefail`
- `infra/install-deps-flash-attn-idempotency-check`
- `infra/install-deps-script-dir`

**Remote (61):**
- `chore/cleanup-for-migration`
- `chore/contributing-merge-policy`
- `chore/remove-dead-legacy-modules`
- `ci/fix-container-shm-size`
- `docs/async-batch-submission-fix`
- `docs/batch-api-cancel-flow`
- `docs/glm51-enable-thinking-fix`
- `docs/install-access-note`
- `docs/install-fix-fa3-verification-import`
- `docs/install-md-accuracy-fixes`
- `docs/install-md-kernel-count-23`
- `docs/server-flags-gaps`
- `feature/remove-attention-mask`
- `fix/fa2-backend-flash-attn-import-guard`
- `fix/version-fallback-post4`
- `infra/dockerfile-shell-pipefail`
- `infra/install-deps-flash-attn-idempotency-check`
- `infra/install-deps-script-dir`
- `main-url-fix`
- `tairan/aot-kernels`
- `tairan/blackwell-01-detect-arch`
- `tairan/blackwell-03-upstream-deps`
- `tairan/blackwell-04-backend-routing`
- `tairan/blackwell-05-configs-docs`
- `tairan/bump-version-metadata`
- `tairan/completion-dist-log`
- `tairan/cuda-graph-contract`
- `tairan/docs-async-batch-submission`
- `tairan/fix_packaging`
- `tairan/kernels-0.3.1-install-fix`
- `tairan/kernels-0.3.1-release`
- `tairan/kimi-k25-mtp-fix`
- `tairan/kimi-k26-support`
- `tairan/minimax-cleanup`
- `tairan/refactor`
- `tairan/remove-length-defaults`
- `tairan/revalidate-5-3-2-gate`
- `tairan/revalidate-5-4a-2-gate`
- `tairan/v1.0.6-post1`
- `tairan/v1.0.8`
- `tairan/worker-decouple-phase-0`
- `tairan/worker-decouple-phase-1-indexing-cleanup`
- `tairan/worker-decouple-phase-1-indexing-gate`
- `tairan/worker-decouple-phase-1-indexing-port`
- `tairan/worker-decouple-phase-2-completion-cleanup`
- `tairan/worker-decouple-phase-2-completion-gate`
- `tairan/worker-decouple-phase-2-completion-port`
- `tairan/worker-decouple-phase-3-sync-cleanup`
- `tairan/worker-decouple-phase-3-sync-gate`
- `tairan/worker-decouple-phase-3-sync-port`
- `tairan/worker-decouple-phase-4-batch-formation-cleanup`
- `tairan/worker-decouple-phase-4-batch-formation-gate`
- `tairan/worker-decouple-phase-4-batch-formation-port`
- `tairan/worker-decouple-phase-5-kv-budget-port`
- `tairan/worker-decouple-phase-5-kv-capacity-cleanup`
- `tairan/worker-decouple-phase-5-kv-capacity-gate`
- `tairan/worker-decouple-phase-5-kv-capacity-port`
- `tairan/worker-decouple-phase-5-kv-stats-gate`
- `tairan/worker-decouple-phase-5-kv-stats-port`
- `tairan/worker-decouple-phase-5-migration-gate`
- `tairan/worker-decouple-phase-5-watermark-gate`

### 2b. Review-first — multi-commit squash (`ahead > 1`)

Detected as squash-merged because the **whole branch's net diff** matches `main`, but each carries multiple commits. The patch-id signal is reliable, yet the commit count warrants a quick human glance before bulk deletion (guards against a branch that was squash-merged and then had unrelated commits reverted to a net-equal state).

**Remote (46), sorted by commit count:**

| ahead | branch |
|------:|--------|
| 117 | `tairan/kimi-k25-decode-opt` |
| 90 | `tairan/support-kimi-k25` |
| 85 | `tairan/cuda-graph` |
| 82 | `tairan/fused-gate-dispatch` |
| 69 | `luzhan/heterogeneous-layer-kv` |
| 68 | `tairan/gpt-oss-kernel` |
| 50 | `tairan/request-pool` |
| 29 | `tairan/support-minimax-m25` |
| 27 | `tairan/kimi-k25-fix` |
| 26 | `tairan/glm5-v109-valid-token-fixes` |
| 23 | `tairan/query-book-buffer-pool` |
| 20 | `tairan/seq-lifespan-monitor` |
| 18 | `tairan/revise-ep-with-offloading` |
| 17 | `tairan/cuda-graph-contract-phase-c` |
| 10 | `feat/fast-init-memfd` |
| 9 | `tairan/runtime-reload` |
| 9 | `tairan/refactor_contiguous_batching` |
| 9 | `tairan/kernel_fusion_Feb11` |
| 9 | `feat/mgn-kernel-project` |
| 8 | `tairan/reorg_code` |
| 7 | `tairan/incremental-result` |
| 5 | `tairan/revise-model-config-abstraction` |
| 5 | `release/v1.0.10-glm5-cudagraph-clean` |
| 5 | `feature/per-sequence-max-completion-tokens` |
| 4 | `tairan/revise-docker-command` |
| 4 | `tairan/reject-overlimit` |
| 4 | `tairan/change_server_logic` |
| 4 | `tairan/batchgen-tokenizer-abstraction` |
| 4 | `t26-per-request-sampling` |
| 4 | `ci/auto-test-mmlu-pro` |
| 4 | `chore/add-dockerfile` |
| 3 | `tairan/glm5-act-quant-v2-integration` |
| 3 | `tairan/fix_bus_error` |
| 3 | `fix/add-debug-flag` |
| 3 | `feat/max-context-length` |
| 2 | `tairan/remove-mgn-kernels` |
| 2 | `tairan/release-v1.0.9.post2` |
| 2 | `tairan/perf_opt_Feb11` |
| 2 | `tairan/k25-config-fix` |
| 2 | `tairan/fix_sigbus_error` |
| 2 | `tairan/dsa_support` |
| 2 | `tairan/cuda-graph-contract-phase-b` |
| 2 | `tairan/content_parse` |
| 2 | `fix/nccl-init-device-binding` |
| 2 | `fix/kernels-explicit-gencode` |
| 2 | `docs/manual-installation-fixes` |

---

## 3. Unmerged — KEEP (unique work not in `main`)

98 remote + 6 local branches have commits whose patch is not in `main`. **Do not delete** without the owner/author confirming the work is abandoned. Among these, the age scan flags:

### 3a. Dead — unmerged & >180 days (abandoned; owner decision required)

| age | last commit | ahead | author | branch |
|----:|-------------|------:|--------|--------|
| 452d | 2025-04-18 | 1 | ZhanLu | `origin/feat/enable-fa` |
| 364d | 2025-07-14 | 2 | Andrewxu313 | `origin/tairan/decoding_opt` |
| 357d | 2025-07-22 | 2 | xly | `origin/xly/qwen` |
| 343d | 2025-08-05 | 29 | Andrewxu313 | `origin/tairan/fix_nccl_bug` |
| 337d | 2025-08-10 | 2 | ZhanLu | `origin/doc/build-instruction-for-mgn-kernel` |
| 336d | 2025-08-11 | 87 | Andrewxu313 | `origin/tairan/kernel_fusion` |
| 320d | 2025-08-27 | 52 | Andrewxu313 | `origin/tairan/add_r1_acc_test` |
| 319d | 2025-08-28 | 2 | lausannel | `origin/ci/fix-auto-test-error` |
| 319d | 2025-08-28 | 36 | Andrewxu313 | `origin/tairan/fix_multi_batch` |
| 313d | 2025-09-04 | 6 | Andrewxu313 | `origin/tairan/add_enable_hugepage_flag` |
| 300d | 2025-09-16 | 9 | lausannel | `origin/feat/torch-native-mla-fp8` |
| 183d | 2026-01-11 | 1 | Tairan Xu | `origin/tairan/CB` |

### 3b. Stale — unmerged & 90–180 days

| age | last commit | ahead | author | branch |
|----:|-------------|------:|--------|--------|
| 179d | 2026-01-16 | 119 | TairanXU | `origin/tairan/support-gpt-oss` |
| 173d | 2026-01-21 | 2 | TairanXU | `origin/tairan/batchgen-ci` |
| 165d | 2026-01-29 | 233 | TairanXU | `origin/tairan/support-gpt-oss-120b` |
| 159d | 2026-02-04 | 79 | TairanXU | `origin/tairan/support-kimi-2dot5` |
| 145d | 2026-02-18 | 84 | TairanXU | `origin/tairan/port_fused_qkv_rope` |
| 144d | 2026-02-19 | 86 | TairanXU | `origin/tairan/fused-int4-wgmma-moe` |
| 135d | 2026-02-28 | 7 | TairanXU | `origin/tairan/kimi-k25-opt` |
| 134d | 2026-03-02 | 16 | TairanXU | `origin/tairan/dynamic-host-kv` |
| 133d | 2026-03-02 | 40 | TairanXU | `origin/tairan/kernel-structure-reorg` |
| 132d | 2026-03-04 | 5 | TairanXU | `origin/tairan/watchdog-improvements` |
| 130d | 2026-03-05 | 6 | TairanXU | `origin/fix/kv-migration-pages` |
| 125d | 2026-03-11 | 4 | TairanXU | `origin/tairan/kv-ablation` |
| 125d | 2026-03-10 | 6 | TairanXU | `origin/tairan/fast-kill` |
| 123d | 2026-03-13 | 3 | TairanXU | `origin/tairan/dsa-kernel-integration` |
| 113d | 2026-03-22 | 27 | TairanXU | `origin/feature/batch-termination-token` |
| 110d | 2026-03-25 | 3 | TairanXU | `origin/tairan/cleanup` |
| 109d | 2026-03-27 | 4 | TairanXU | `origin/hotfix/v1.0.5.post1` |
| 109d | 2026-03-27 | 31 | Tairan Xu | `origin/tairan/minimax-fp8-blockwise` |
| 109d | 2026-03-26 | 32 | TairanXU | `origin/tairan/qkv-fusion` |
| 109d | 2026-03-26 | 30 | TairanXU | `origin/tairan/minimax-gqa-decode` |
| 108d | 2026-03-27 | 4 | TairanXU | `origin/tairan/r1-model` |
| 107d | 2026-03-29 | 36 | TairanXU | `origin/tairan/scheduler-split` |
| 107d | 2026-03-29 | 1 | TairanXU | `origin/tairan/sm120-support` |
| 103d | 2026-04-01 | 9 | TairanXU | `origin/tairan/deterministic-fixes` |
| 91d | 2026-04-14 | 103 | Tairan.Xu | `origin/tairan/worker-reextract` |

**Local unmerged (6) — keep:** `feature/deepseek-v4-kernel-integration` (active, 96 ahead), `fix/v4flash-tokenizer-and-metadata-discovery` (16), `perf/v4flash-sm120-decode` (42), `release-v1.0.10.post5` (pending release, main+1), `release-v1.0.10.post4` (release checkpoint), `docs/manual-installation-fixes` (⚠️ local tip differs from its own squash-merged remote — sync/inspect before deleting).

### Dead/stale ownership (for follow-up)

Unmerged branches ≥90 days old, grouped by author (note: `TairanXU` / `Tairan Xu` / `Tairan.Xu` are the same person under different git configs):

- **TairanXU** — 23 branch(es)
- **Andrewxu313** — 6 branch(es)
- **ZhanLu** — 2 branch(es)
- **Tairan Xu** — 2 branch(es)
- **lausannel** — 2 branch(es)
- **xly** — 1 branch(es)
- **Tairan.Xu** — 1 branch(es)

---

## Delete Plan

> Execute top-to-bottom. Each tier is safe on its own; stop at any point. **Nothing here runs automatically** — copy/paste after review. Remote deletions are owner-gated per `CONTRIBUTING.md`.

### Step 0 — Sync (always first)

```bash
git fetch --prune origin      # drop remote-tracking refs for already-deleted branches
```

### Step 1 — Local branches (this machine)

```bash
# Linear-merged: -d refuses if not truly merged (safety kept on purpose)
git branch -d main            # SKIP — keep default branch; fast-forward instead:
# git fetch origin && git branch -f main origin/main   # (only when 'main' not checked out)

# Squash-merged locals (-D required; -d would refuse):
git branch -D \
  docs/async-batch-submission-fix \
  docs/batch-api-cancel-flow \
  docs/glm51-enable-thinking-fix \
  docs/install-access-note \
  docs/install-md-accuracy-fixes \
  docs/install-md-kernel-count-23 \
  docs/server-flags-gaps \
  fix/fa2-backend-flash-attn-import-guard \
  fix/kernels-explicit-gencode \
  fix/version-fallback-post4 \
  infra/dockerfile-shell-pipefail \
  infra/install-deps-flash-attn-idempotency-check \
  infra/install-deps-script-dir
```

### Step 2 — Remote: Linear-merged (8, full linear history)

```bash
git push origin --delete \
  ci/sglang-release-test \
  release/v1.0.10.post3-glm5-stability \
  tairan/adapt_Pynccl_Comm \
  tairan/cold_start_opt \
  tairan/fix_posix_shm \
  tairan/fix_stateless_comm_init \
  tairan/full-dsa-cuda-graph-prod \
  tairan/urgent_fix
```

### Step 3 — Remote: Squash-merged, high-confidence (`ahead == 1`)

```bash
git push origin --delete \
  chore/cleanup-for-migration \
  chore/contributing-merge-policy \
  chore/remove-dead-legacy-modules \
  ci/fix-container-shm-size \
  docs/async-batch-submission-fix \
  docs/batch-api-cancel-flow \
  docs/glm51-enable-thinking-fix \
  docs/install-access-note \
  docs/install-fix-fa3-verification-import \
  docs/install-md-accuracy-fixes \
  docs/install-md-kernel-count-23 \
  docs/server-flags-gaps \
  feature/remove-attention-mask \
  fix/fa2-backend-flash-attn-import-guard \
  fix/version-fallback-post4 \
  infra/dockerfile-shell-pipefail \
  infra/install-deps-flash-attn-idempotency-check \
  infra/install-deps-script-dir \
  main-url-fix \
  tairan/aot-kernels \
  tairan/blackwell-01-detect-arch \
  tairan/blackwell-03-upstream-deps \
  tairan/blackwell-04-backend-routing \
  tairan/blackwell-05-configs-docs \
  tairan/bump-version-metadata \
  tairan/completion-dist-log \
  tairan/cuda-graph-contract \
  tairan/docs-async-batch-submission \
  tairan/fix_packaging \
  tairan/kernels-0.3.1-install-fix \
  tairan/kernels-0.3.1-release \
  tairan/kimi-k25-mtp-fix \
  tairan/kimi-k26-support \
  tairan/minimax-cleanup \
  tairan/refactor \
  tairan/remove-length-defaults \
  tairan/revalidate-5-3-2-gate \
  tairan/revalidate-5-4a-2-gate \
  tairan/v1.0.6-post1 \
  tairan/v1.0.8 \
  tairan/worker-decouple-phase-0 \
  tairan/worker-decouple-phase-1-indexing-cleanup \
  tairan/worker-decouple-phase-1-indexing-gate \
  tairan/worker-decouple-phase-1-indexing-port \
  tairan/worker-decouple-phase-2-completion-cleanup \
  tairan/worker-decouple-phase-2-completion-gate \
  tairan/worker-decouple-phase-2-completion-port \
  tairan/worker-decouple-phase-3-sync-cleanup \
  tairan/worker-decouple-phase-3-sync-gate \
  tairan/worker-decouple-phase-3-sync-port \
  tairan/worker-decouple-phase-4-batch-formation-cleanup \
  tairan/worker-decouple-phase-4-batch-formation-gate \
  tairan/worker-decouple-phase-4-batch-formation-port \
  tairan/worker-decouple-phase-5-kv-budget-port \
  tairan/worker-decouple-phase-5-kv-capacity-cleanup \
  tairan/worker-decouple-phase-5-kv-capacity-gate \
  tairan/worker-decouple-phase-5-kv-capacity-port \
  tairan/worker-decouple-phase-5-kv-stats-gate \
  tairan/worker-decouple-phase-5-kv-stats-port \
  tairan/worker-decouple-phase-5-migration-gate \
  tairan/worker-decouple-phase-5-watermark-gate
```

### Step 4 — Remote: Squash-merged, review-first (`ahead > 1`)

Glance at each (see §2b table) before running. Same command form:

```bash
git push origin --delete \
  chore/add-dockerfile \
  ci/auto-test-mmlu-pro \
  docs/manual-installation-fixes \
  feat/fast-init-memfd \
  feat/max-context-length \
  feat/mgn-kernel-project \
  feature/per-sequence-max-completion-tokens \
  fix/add-debug-flag \
  fix/kernels-explicit-gencode \
  fix/nccl-init-device-binding \
  luzhan/heterogeneous-layer-kv \
  release/v1.0.10-glm5-cudagraph-clean \
  t26-per-request-sampling \
  tairan/batchgen-tokenizer-abstraction \
  tairan/change_server_logic \
  tairan/content_parse \
  tairan/cuda-graph \
  tairan/cuda-graph-contract-phase-b \
  tairan/cuda-graph-contract-phase-c \
  tairan/dsa_support \
  tairan/fix_bus_error \
  tairan/fix_sigbus_error \
  tairan/fused-gate-dispatch \
  tairan/glm5-act-quant-v2-integration \
  tairan/glm5-v109-valid-token-fixes \
  tairan/gpt-oss-kernel \
  tairan/incremental-result \
  tairan/k25-config-fix \
  tairan/kernel_fusion_Feb11 \
  tairan/kimi-k25-decode-opt \
  tairan/kimi-k25-fix \
  tairan/perf_opt_Feb11 \
  tairan/query-book-buffer-pool \
  tairan/refactor_contiguous_batching \
  tairan/reject-overlimit \
  tairan/release-v1.0.9.post2 \
  tairan/remove-mgn-kernels \
  tairan/reorg_code \
  tairan/request-pool \
  tairan/revise-docker-command \
  tairan/revise-ep-with-offloading \
  tairan/revise-model-config-abstraction \
  tairan/runtime-reload \
  tairan/seq-lifespan-monitor \
  tairan/support-kimi-k25 \
  tairan/support-minimax-m25
```

### Step 5 — Unmerged dead/stale (DO NOT auto-delete)

These hold unique work. Ping the owner (see §3 ownership) and only delete after explicit confirmation. There is no safe bulk command for this tier by design.

---

## Safety & Recovery

- **`git branch -d` vs `-D`:** `-d` refuses to delete a branch that isn't an ancestor of its upstream — keep using it for linear-merged branches as a built-in guard. `-D` (force) is only needed for squash-merged branches, where `-d` cannot see through the rewritten commit.
- **`--force-with-lease`:** if you script remote deletion via a tool that pushes, prefer `--force-with-lease` over `--force` so a colleague's newer commit on the same ref is never silently discarded.
- **Recovery window:** a deleted branch tip is recoverable from the reflog for **~90 days** (`gc.reflogExpire` default; 30 days for unreachable objects). Recover with `git reflog --all | grep <branch>` then `git branch <branch> <sha>`. Beyond that, `git fsck --lost-found` is a last resort until the next `git gc`.
- **Protected/open-PR branches:** GitHub refuses to delete branches tied to open PRs, and branch-protection rules/rulesets can block deletion server-side. Enable **Settings → Automatically delete head branches** to auto-prune PR-merged branches going forward.

---

## References

- Classification method (LINEAR / SQUASH / UNMERGED) mirrors `git-delete-merged-branches --effort` levels 1 / 3 / never.
- `git cherry` (patch-id equivalence, squash/rebase detection) — https://git-scm.com/docs/git-cherry
- `git patch-id` (stable patch hash) — https://git-scm.com/docs/git-patch-id
- `git merge-base --is-ancestor` (linear-merge check) — https://git-scm.com/docs/git-merge-base
- `git branch` (`-d` vs `-D`, `--merged`) — https://git-scm.com/docs/git-branch
- `git reflog` (90-day recovery default) — https://git-scm.com/docs/git-reflog
- Pro Git § Branch Management — https://git-scm.com/book/en/v2/Git-Branching-Branch-Management
- GitHub "Stale branches" = 3 months — https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-branches-in-your-repository/viewing-branches-in-your-repository
- GitHub auto-delete head branches — https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/configuring-pull-request-merges/managing-the-automatic-deletion-of-branches
- `git-delete-merged-branches` (`--effort=3` = synthetic squash + `git cherry`) — https://github.com/hartwork/git-delete-merged-branches

---

*Generated by an automated branch audit. Reproduce with the classification pipeline in the References section against a freshly fetched `origin/main`.*
