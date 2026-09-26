#!/usr/bin/env bash
# Build a source worktree into a runnable BatchGen runtime without pip-installing it.
#
#   1. verify the active Python has the pinned Torch/CUDA ABI
#   2. build batchgen.core_engine and batchgen_kernels AOT extensions in place
#   3. optionally point a pyroot (PYTHONPATH root) at this worktree
#   4. optionally run the model runtime preflight against that pyroot
#
# Nothing is compiled at server start: the preflight rejects a missing or stale
# core_engine and native packages that come from another worktree.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/build_source_runtime.sh [options]

  --pyroot DIR     Create DIR/batchgen and DIR/batchgen_kernels symlinks to this
                   worktree. Fails if DIR already links to a different worktree.
  --model ID       Run `python -m batchgen.runtime_preflight --model ID` after the
                   build (uses --pyroot, or this worktree, as PYTHONPATH).
  --skip-core      Do not rebuild batchgen.core_engine.
  --skip-kernels   Do not rebuild batchgen_kernels.
  -h, --help       Show this help.

Environment: TORCH_CUDA_CHANNEL (default cu128), BUILD_ARCH (default sm90a),
KERNELS_PARALLEL (extensions built at once, default 24), MAX_JOBS (ninja jobs
per extension, default 8). Run inside the BatchGen environment.
EOF
}

PYROOT=""
MODEL=""
BUILD_CORE=1
BUILD_KERNELS=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pyroot) PYROOT="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --skip-core) BUILD_CORE=0; shift ;;
        --skip-kernels) BUILD_KERNELS=0; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
log() { echo "[build-source-runtime] $*"; }
die() { echo "[build-source-runtime] ERROR: $*" >&2; exit 1; }

# 1. Torch/CUDA ABI: building against the wrong Torch produces extensions that
#    import-fail (or worse, load) under the real runtime.
CHANNEL="${TORCH_CUDA_CHANNEL:-cu128}"
python - "$CHANNEL" <<'PY' || die "active Python does not have the pinned Torch ABI"
import sys
import torch

channel = sys.argv[1]
cuda = f"{channel[2:-1]}.{channel[-1]}"
print(f"[build-source-runtime] python={sys.executable} torch={torch.__version__} cuda={torch.version.cuda}")
if not torch.__version__.startswith("2.9.0") or torch.version.cuda != cuda:
    sys.exit(f"expected torch 2.9.0 with CUDA {cuda} ({channel})")
PY

# 2. In-place AOT builds. setuptools only recompiles extensions whose sources
#    changed, so repeated runs during debugging are incremental.
if [[ $BUILD_CORE -eq 1 ]]; then
    log "building batchgen.core_engine in $ROOT"
    (cd "$ROOT" && BUILD_OPS=1 python setup.py build_ext --inplace)
    # setuptools keeps the old artifact when nothing changed; touch it so the
    # preflight freshness check reflects this verified build.
    core_so=$(ls "$ROOT"/batchgen/core_engine*.so 2>/dev/null | head -1) \
        || true
    [[ -n "$core_so" ]] || die "build finished without batchgen/core_engine*.so"
    touch "$core_so"
fi
if [[ $BUILD_KERNELS -eq 1 ]]; then
    log "building batchgen_kernels in $ROOT/batchgen_kernels"
    # Extensions are independent and mostly single-file, so build them
    # concurrently (h200-instance-1, clean: 131 s vs 1142 s serial).
    (cd "$ROOT/batchgen_kernels" \
        && MAX_JOBS="${MAX_JOBS:-8}" python setup.py build_ext \
            --inplace \
            --parallel "${KERNELS_PARALLEL:-24}")
fi

# 3. pyroot: never repoint a pyroot that belongs to another worktree; a server
#    started from it would change code underneath a running lane.
if [[ -n "$PYROOT" ]]; then
    mkdir -p "$PYROOT"
    for pkg in batchgen batchgen_kernels; do
        link="$PYROOT/$pkg"
        target="$ROOT/$pkg"
        if [[ -L "$link" ]]; then
            current="$(cd "$link" && pwd -P)"
            [[ "$current" == "$target" ]] \
                || die "$link already points to $current, not $target"
        elif [[ -e "$link" ]]; then
            die "$link exists and is not a symlink"
        else
            ln -s "$target" "$link"
        fi
    done
    log "pyroot $PYROOT -> $ROOT"
fi

# 4. Preflight from a neutral cwd so the current directory cannot shadow imports.
if [[ -n "$MODEL" ]]; then
    PP="${PYROOT:-$ROOT}"
    log "runtime preflight model=$MODEL PYTHONPATH=$PP"
    (cd / && PYTHONPATH="$PP" python -m batchgen.runtime_preflight --model "$MODEL")
fi
log "done"
