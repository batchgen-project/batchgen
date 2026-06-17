# PR rules (digest)

> Operational quick-reference for every PR into `main` — agents and humans alike.
> **Authoritative source: [`../PR_MERGE_POLICY.md`](../PR_MERGE_POLICY.md).** This file never
> diverges from it; if they disagree, the contract wins. Read this before you open or push a PR.

## 1. Declare one change type (PR template)

Tick **exactly one** in the PR description. The type fixes the files you may touch.

| Type | May touch — and nothing else |
|------|------------------------------|
| `model`  | `batchgen/models/**`; the registration seam `get_initializer.py`, `get_parallel_strategy_manager.py`, `models/__init__.py`; model kernels under `batchgen_kernels/**`; `configurations/**`; `tests/**`; `docs/**` |
| `kernel` | `batchgen_kernels/**`; in-tree kernel dirs `batchgen/{moe,attention,gemm,quantization,triton_kernels,other_kernels}/`; `op_builder/`; `tests/**`; `docs/**` |
| `core`   | the scaffolding layer (below) — **the only type that may** |
| `fix`    | the files implicated by the bug + a regression test under `tests/**` |
| `infra`  | `.github/**`, `scripts/**`, `docker/**`, `Makefile`, `setup.py`, `setup.cfg`, `pyproject.toml`, `MANIFEST.in` |
| `docs`   | `docs/**`, `*.md`, `README*` |

**The scaffolding layer is off-limits to `model`/`kernel` PRs (core-only):** top-level
`batchgen/*.py` runtime modules (`batchgen_worker.py`, `server_worker_main_loop.py`,
`continuous_batching.py`, `scheduler.py`, `task_scheduler.py`, `decode*.py`, `prefill*.py`,
`sampling.py`, `sequence.py`, `migration.py`, `pd_orchestrator.py`, `query_*.py`,
`parameter_server*.py`, `node_manager.py`, `inference_runtime.py`, `batch_*.py`,
`batchgen_server*.py`, `launch_*.py`, `entrypoint.py`, `lifespan.py`, `generate.py`,
`model_instance.py`) and the subpackages `scheduler/`, `server/`, `kv_cache/`, `worker/`,
`sequence_manager/`, `distributed/`, `planner/`, `cuda_graph/`, `core/`. Need a core change to
land a model? **Split it into a separate `core` PR** with owner approval.

## 2. Never put these in a PR

- Scratch / bench / debug scripts in a package or repo root (`bench_* / debug_* / check_* / scratch_* / tmp_*`).
- `test_*.py` inside the runtime package — tests live in `tests/`.
- Server-side `BATCHGEN_*` env-var debug guards — use a batch-level `batchgen_debug` flag (`timing.py` excepted).
- Stray `print()` / `logging.debug()` / commented-out dead code in production paths.
- Generated artifacts: `*.log`, `*.nsys-rep`, `*.ncu-rep`, CSVs, checkpoints, wheels, compiled extensions. Never `git add -f`.
- Loose `.py` at repo root (only `setup.py`). `Co-Authored-By` trailers. `git add -A` / `git add .`.

## 3. Keep it clean

- **Surgical & single-concern** — every changed line traces to the task; no drive-by refactors. Split unrelated concerns.
- Update a module's `MODULE.md` in the same PR if its public API changes.
- Commits: Angular format `<type>: <summary>` + body (≥20 chars, except `docs`).

## 4. Check it before you push

```bash
bash .github/workflows/scripts/check-pr-hygiene.sh origin/main      # §1/§4 hygiene
PR_TYPE=model bash .github/workflows/scripts/check-pr-hygiene.sh origin/main   # + §2 boundary
```

CI runs the same check on every PR (report-only until the cleanup ledger clears, then required).

## 5. Merge & override

- **Only the owner (`@Andrewxu313`) merges.** Needs ≥1 Code Owner approval + green CI.
- Override a hygiene/boundary finding only via §9 of the contract: per-line `# noqa: hygiene`,
  a maintainer `ci:skip-hygiene` label, or owner admin-bypass — each with a recorded `Override-Reason`.
