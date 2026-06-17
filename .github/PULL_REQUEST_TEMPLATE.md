## Description
Briefly describe your changes.

## Motivation
Explain why this change is needed and what problem it solves.
If it fixes an issue, link it (e.g., `close #123`).

## Type of Change
<!-- Pick EXACTLY ONE. It fixes the files this PR may touch — see PR_MERGE_POLICY.md §2.5. -->
- [ ] `model` — add/extend model support (`models/**` + registration seam + model kernels only)
- [ ] `kernel` — add/optimize a compute kernel (`batchgen_kernels/**` + in-tree kernel dirs)
- [ ] `core` — change scheduling/serving/runtime scaffolding (the **only** type that may)
- [ ] `fix` — narrow bug fix (+ a regression test)
- [ ] `infra` — build / CI / packaging / scripts / Docker
- [ ] `docs` — documentation only

## Checklist
- [ ] I have read the [CONTRIBUTING](https://github.com/batchgen-project/batchgen/blob/main/CONTRIBUTING.md) guide and the [PR Merge Policy Contract](https://github.com/batchgen-project/batchgen/blob/main/PR_MERGE_POLICY.md).
- [ ] I have updated the tests (if applicable).
- [ ] I have updated the documentation (if applicable).

## PR Merge Policy Contract — pre-merge checklist
<!-- Every box must be ticked before "Ready for review". See PR_MERGE_POLICY.md §5. -->
- [ ] `git diff --stat origin/main` reviewed; every file traces to the task — no unrelated files (§3.1).
- [ ] Exactly one **Type of Change** ticked above; all changed files are within that type's permitted set (§2.5) — no scaffolding edits in a `model`/`kernel` PR (§2.6).
- [ ] No `bench_* / debug_* / check_* / scratch_* / tmp_*` scripts added to a production package or repo root (§1.1).
- [ ] No `test_*.py` added inside the runtime package; tests are under `tests/` (§1.2, §2.1).
- [ ] No new `BATCHGEN_*` env-var debug guard; debug behavior is a `batchgen_debug` batch flag (§1.3).
- [ ] No leftover `print()`, `logging.debug()`, or commented-out code in the changed files (§1.4–§1.5).
- [ ] No logs, profiler traces, CSVs, checkpoints, wheels, or compiled artifacts staged; nothing added with `-f` (§1.6).
- [ ] Touched modules' `MODULE.md` updated if the public API changed (§2.2).
- [ ] One concern only; diff is surgical (§3).
- [ ] Commits are clean: Angular format, body present, **no `Co-Authored-By`** (§4).
- [ ] CI is green (format, lint, hygiene, tests).
