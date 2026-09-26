# Runtime dependency contract

BatchGen validates the complete native runtime before it starts the HTTP
server, creates worker processes, allocates model shared memory, or opens the
serving port. This turns a first-request import failure into an actionable
startup error.

## Install a production environment

Use a dedicated Python/conda environment for each incompatible Torch/CUDA
ABI. Install BatchGen non-editably from the worktree that owns the server:

```bash
./scripts/install_deps.sh --all
```

The installer verifies the existing Torch contract (`2.9.0` with the selected
CUDA channel) and fails instead of replacing a mismatched installation. Use
`--repair` only when replacing that Torch installation is intentional:

```bash
./scripts/install_deps.sh --all --repair
```

The installation builds or installs AOT (ahead-of-time) native extensions for
`batchgen_kernels` and `batchgen.core_engine`, then imports representative
extensions outside the checkout. A missing NUMA development header, an
incomplete kernel wheel, or a JIT-built extension is an installation failure.
The installer does not warm a JIT cache and the server never compiles these
extensions on its first request.

Never use an editable install (`pip install -e`) in a serving environment.

## Run from a source worktree

For development, run a worktree directly instead of installing it. Build it
once (and after every C++/CUDA change) and point a `PYTHONPATH` root at it:

```bash
./scripts/build_source_runtime.sh \
    --pyroot /path/to/pyroot_mylane \
    --model openai/gpt-oss-120b
PYTHONPATH=/path/to/pyroot_mylane python -m batchgen.launch_http_server ...
```

The script verifies Torch, builds `batchgen.core_engine` and `batchgen_kernels`
in place, links `pyroot/batchgen` and `pyroot/batchgen_kernels` to the
worktree, and runs the preflight. It refuses to repoint a pyroot that already
links to a different worktree. Several worktrees can share one environment if
they use the same Torch/CUDA ABI and third-party wheels.

| Changed | Rebuild |
| --- | --- |
| Python only | nothing |
| `core/**` | `--skip-kernels` (a stale `core_engine` fails the preflight) |
| `batchgen_kernels/src/**` | `--skip-core` (stale kernels are **not** detected; always rebuild) |
| Torch, CUDA, or a third-party native package | a new environment |

## Provenance rule

```text
batchgen, batchgen_kernels, batchgen.core_engine → the same root
                                                   (one site-packages or one worktree)
flash_attn / flash_attn_interface, flash_mla     → site-packages
```

Mixing roots (for example `batchgen` from one worktree and kernels from
another, or from site-packages) is rejected.

Run the same check without starting a server:

```bash
cd / && PYTHONPATH=<pyroot> python -m batchgen.runtime_preflight --model MODEL
```

It prints the resolved path of every checked module.

## Startup checks

`python -m batchgen.launch_http_server --model MODEL` performs a model-specific
preflight before importing the HTTP server. It checks:

- Torch version and CUDA ABI;
- the selected tokenizer, EOS IDs, and its chat-template rendering (or the
  model's explicit non-Jinja formatter);
- the FlashAttention module the model declares (FA3 or FA2) and FlashMLA
  where required;
- the `libucx` runtime loader and DeepGEMM API where required;
- AOT `batchgen_kernels` extensions, plus the WGMMA decode kernel on H20;
- a native (AOT) `batchgen.core_engine` module, not older than `core/` when
  running from a worktree;
- the provenance rule above.

An unknown model type has no implicit dependency contract and is rejected.
Workers run in the same environment the preflight checked, so they resolve the
same modules. If a declared module cannot be imported, startup fails; on H20 a
WGMMA decode kernel that fails to load is an error, not a switch to FA3.

## Interpreting common failures

| Message | Meaning | Correct action |
| --- | --- | --- |
| `Torch ... unsupported` or CUDA ABI mismatch | The environment contains a different Torch build | Create the matching environment, or explicitly use `--repair` |
| `... same install or worktree` | `batchgen`, kernels, and `core_engine` come from different roots | Rebuild one pyroot from one worktree, or reinstall |
| `... outside site-packages` | A third-party native package is shadowed by a checkout | Run from a neutral directory; remove the shadowing path |
| `core_engine ... is older than` | `core/` changed after the last build | `build_source_runtime.sh --skip-kernels` |
| `required runtime module ... failed to import` | A native dependency or its ABI is incomplete | Fix the install; do not add a request-time fallback |
| `core_engine is not an AOT native module` | The core engine was never built | Re-run the installer, or `build_source_runtime.sh` for a worktree |
| `unknown model type ... no declared runtime contract` | No exact dependency matrix exists for that model | Add and review a model contract before serving it |

The preflight is intentionally fail-closed. A healthy HTTP endpoint is not
reported until these checks pass, so a later request should not be the first
place that discovers FA3/FA2, UCX, Torch ABI, tokenizer, or AOT problems.
