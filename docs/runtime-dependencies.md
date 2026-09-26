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

Do not launch from a source checkout after installation. An editable install or
a checkout on `PYTHONPATH` can shadow the site-packages wheel and select native
extensions built by another worktree. Keep the model's `batchgen` and
`batchgen_kernels` pair from the same worktree/build, and verify their paths
before launch.

## Startup checks

`python -m batchgen.launch_http_server --model MODEL` performs a model-specific
preflight before importing the HTTP server. It checks:

- Torch version and CUDA ABI;
- the selected tokenizer, EOS IDs, and its chat-template rendering (or the
  model's explicit non-Jinja formatter);
- the exact FlashAttention backend (FA3 or FA2) and FlashMLA where required;
- the `libucx` runtime loader and DeepGEMM API where required;
- AOT `batchgen_kernels` extensions and the selected H20 decode backend;
- an installed, native `batchgen.core_engine` module;
- site-packages provenance for native wheels, rejecting source-worktree
  shadowing.

An unknown model type has no implicit dependency contract and is rejected.
The selected backend is inherited by worker processes. If that backend cannot
be imported, startup fails; BatchGen does not silently switch to another
backend or to a vanilla implementation.

`--allow-runtime-fallback` is an explicit diagnostic opt-in. It is not enabled
by default and is not a production repair for a missing dependency.

## Interpreting common failures

| Message | Meaning | Correct action |
| --- | --- | --- |
| `Torch ... unsupported` or CUDA ABI mismatch | The environment contains a different Torch build | Create the matching environment, or explicitly use `--repair` |
| `batchgen_kernels ... outside site-packages` | A source checkout or another worktree shadows the wheel | Remove the shadowing path and reinstall non-editably |
| `required runtime module ... failed to import` | A native dependency or its ABI is incomplete | Fix the install; do not add a request-time fallback |
| `core_engine is not an AOT native module` | The core engine was not built into the installed package | Re-run the AOT installation with the required CUDA/NUMA toolchain |
| `unknown model type ... no declared runtime contract` | No exact dependency matrix exists for that model | Add and review a model contract before serving it |

The preflight is intentionally fail-closed. A healthy HTTP endpoint is not
reported until these checks pass, so a later request should not be the first
place that discovers FA3/FA2, UCX, Torch ABI, tokenizer, or AOT problems.
