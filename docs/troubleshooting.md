# BatchGen — Troubleshooting / Commonly-Met Problems

Symptom → root cause → fix, for issues hit repeatedly during setup and model bring-up.
Build/install issues first, then model-development (Kimi-Linear / K3 family) issues.

---

## 1. `core_engine` JIT build fails: `numa.h: No such file or directory`

**Symptom** — first server launch (or first `import` that pulls in a parameter server)
raises `RuntimeError: Error building extension 'core_engine'`; running `ninja` in
`~/.cache/torch_extensions/py311_cu128/core_engine/` shows:

```
FAILED: posix_shm.o
core/Parameter_Server/posix_shm.cpp:41:10: fatal error: numa.h: No such file or directory
```

**Cause** — `core_engine` is JIT-compiled by ninja at first launch and `#include <numa.h>`
+ links `-lnuma`. The runtime packages `numactl` / `numactl-libs` do **not** ship the
header; the `-devel` / `-dev` package is required and is easy to miss on a fresh node.

**Fix** — install the NUMA development headers:

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

## 4. Stale / corrupted `core_engine` JIT build

**Symptom** — build errors that persist after fixing the real cause, or after two
concurrent processes both triggered the JIT build (they race on the shared build dir).

**Fix** — clear the cache and let it rebuild single-process:

```bash
rm -rf ~/.cache/torch_extensions/py311_cu128/core_engine
python -c "import batchgen.models.moonshotai.kimi_k25.config"   # rebuilds once
```

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
