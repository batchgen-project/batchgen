# BatchGen — Troubleshooting / Commonly-Met Problems

Symptom → root cause → fix, for issues hit repeatedly during setup and model bring-up.
Build/install issues first, then model-development (Kimi-Linear / K3 family) issues, then
startup and serving notes for this release.

---

## 1. `core_engine` build fails: `numa.h: No such file or directory`

**Symptom:** The AOT `core_engine` build during installation (`install_deps.sh`, or
`BUILD_OPS=1 pip install .`) fails with:

```
core/Parameter_Server/posix_shm.cpp:41:10: fatal error: numa.h: No such file or directory
```

**Cause:** `core_engine` is built ahead of time at install and `#include <numa.h>`
+ links `-lnuma`. The runtime packages `numactl` / `numactl-libs` do **not** ship the
header; the `-devel` / `-dev` package is required and is easy to miss on a fresh node.

**Fix:** Install the NUMA development headers:

```bash
# RHEL / TencentOS / CentOS
sudo dnf install -y numactl-devel     # or: sudo yum install -y numactl-devel
# Debian / Ubuntu
sudo apt-get install -y libnuma-dev
```

Verify `ls /usr/include/numa.h`. This is now handled automatically by
`scripts/install_deps.sh` (`install_system_deps`), so a clean install no longer hits it.

---

## 2. `[WGMMA grouped] not available (SM90 required)` / MXFP4 MoE kernels missing

**Symptom** — at import:
`Failed to load WGMMA grouped MoE kernels: No module named 'batchgen_kernels.moe._C_grouped_mxfp4_wgmma'`.

**Cause** — the `batchgen_kernels` CUDA extensions were not built (or built for the wrong
arch). On H20 the arch flag must be Hopper-`a`.

**Fix** — build the kernels with the H20 arch flag:

```bash
cd batchgen_kernels
TORCH_CUDA_ARCH_LIST=9.0a BUILD_ARCH=sm90a pip install . --no-build-isolation
```

BF16 models (e.g. the Kimi-Linear-48B testbed) do not need the MXFP4 WGMMA kernels;
MXFP4 models (Kimi-K3) do.

---

## 3. `ImportError` / missing compiled extensions when running from the source tree

**Symptom** — imports fail with missing CUDA extensions even though install succeeded.

**Cause** — running Python from the repo root makes the source `batchgen/` and
`batchgen_kernels/` dirs shadow the installed site-packages (which hold the compiled `.so`).

**Fix** — run from any directory that is **not** the repo root (see `docs/INSTALL.md`).

---

## 4. Startup refuses the kernel package after an upgrade

**Symptom:** The server exits before it serves, with
`batchgen requires batchgen_kernels >= 0.6.0, found <version>`, or, for a GLM-5 family
model, `required runtime module 'batchgen_kernels.attention._C_dsa_fast_topk' failed to
import` (or another `_C_*` extension name).

**Cause:** The installed `batchgen_kernels` was built for an earlier release. This release
builds the GLM-5 DSA indexer kernels and the FP8 WGMMA MoE kernels ahead of time inside
`batchgen_kernels` 0.6.0 instead of compiling them at run time, so an older build lacks
them. The server does not fall back to compiling them.

**Fix:** Rebuild `batchgen_kernels` from the new checkout: re-run
`./scripts/install_deps.sh --all`, or, for a source worktree,
`./scripts/build_source_runtime.sh` without `--skip-kernels`. A stale `core_engine` is
reported separately (`core_engine ... is older than`); see
[Runtime dependency contract](runtime-dependencies.md#interpreting-common-failures).

---

## 5. Kimi-Linear / Kimi-K3 family — `fla` (flash-linear-attention) version

**Symptom** — running the shipped HF modeling code raises
`fused_kda_gate() got an unexpected keyword argument 'g_bias'`, or `chunk_kda` silently
ignores `A_log`/`dt_bias`/`transpose_state_layout` (wrong KDA output).

**Cause** — the Kimi modeling code targets **fla git-main**, but PyPI's latest release
(`fla-core==0.5.2`) is older and has a different KDA gate API. Note also that
**different model releases pin different fla APIs**: Kimi-Linear-48B's `modeling_kimi.py`
uses the *old* `fused_kda_gate(g, A_log, head_dim, g_bias=...)`, while K3's
`modeling_kimi_linear.py` uses the *new* fused-in-`chunk_kda` API (git-main).

**Fix** — use fla git-main (pure Python, no build):

```bash
git clone --depth 1 https://github.com/fla-org/flash-linear-attention.git fla-src
export PYTHONPATH=$PWD/fla-src:$PYTHONPATH   # shadows any pip-installed fla
```

For a trustworthy single oracle, run testbed weights through **K3's**
`modeling_kimi_linear.py` (git-main API) — verified mathematically identical to the
testbed's own `modeling_kimi.py` (only artifact: `A_log` stored `[1,1,H,1]` vs `[H]`,
reshape on load).

---

## 6. Kimi-Linear / Kimi-K3 — running the reference HF modeling code

- **`AttributeError: 'types.UnionType' object has no attribute '__name__'`** in
  `transformers/utils/auto_docstring.py` — the shipped modeling code decorates with
  `@auto_docstring` and older transformers can't parse `X | None` annotations. Docstrings
  are cosmetic: monkeypatch it to identity before importing the modeling code:
  ```python
  import transformers.utils as _tu
  _tu.auto_docstring = lambda obj=None, *a, **k: ((lambda f: f) if obj is None else obj)
  ```
- **`No module named 'flash_attn'`** — the modeling code forces `flash_attention_2`.
  For a ground-truth oracle, force eager instead (fp32 softmax, higher precision, no dep):
  set `config._attn_implementation = "eager"` on **every** module's `.config` *after*
  construction (the model `__init__` overrides it back to flash_attention_2).

---

## 7. First startup on a new node is slower than later ones (Triton prewarm)

**Symptom:** Kimi-K3 or a GLM-5 family model takes noticeably longer to become ready on a
node with an empty Triton cache, and the log shows `[K3_TRITON_WARMUP]` or
`[GLM5_TRITON_WARMUP]` lines with `compiled=` counts above zero.

**Cause:** Expected behaviour. Both models compile the Triton kernels their serving path
can reach before they serve, so the first requests do not stall on compilation. The ranks
of a node split the compiles over Triton's shared cache directory (`TRITON_CACHE_DIR`, by
default `~/.triton/cache`). With a warm cache the same lines report `loaded=` counts and
the step only loads binaries.

**Fix:** None needed. To avoid recompiling after a container is recreated, keep the
Triton cache directory on storage that outlives the container.

---

## 8. `late compile` warnings after startup

**Symptom:** After `late-compile guard armed`, the log shows WARNING lines such as
`[K3_TRITON_WARMUP] late compile kernel=<module.kernel> source=compile ...` (or the
`[GLM5_TRITON_WARMUP]` equivalent).

**Cause:** A Triton specialisation that the prewarm did not cover was compiled while
serving. The request still completes; the line names the kernel and its full
specialisation key so the gap can be reported. `source=load` means the binary came from
the on-disk cache but was still missed by the prewarm. For GLM-5, decode batches larger
than the top CUDA-graph bucket run eagerly, and some eager kernels are keyed on the active
page-table width; those keys are not prewarmed, so late compiles on that path are expected.

**Fix:** Report the line. To make such a compile fail instead (acceptance runs only,
never in production), create the batch with `"batchgen_debug": {"k3_triton_strict": true}`
for Kimi-K3 or `"batchgen_debug": {"glm5_triton_strict": true}` for GLM-5. The error is
raised inside the model forward, so it stops the worker pool, not only that batch.

---

## 9. Known limitation: Kimi-K3 with `--enable-deepep`

Kimi-K3 with `--enable-deepep` is not supported in v1.1.0. Kimi-K3 decode uses the default
NCCL expert-parallel exchange; launch Kimi-K3 without `--enable-deepep`.
