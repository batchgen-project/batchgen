#!/usr/bin/env bash
# Usage: cache_nvcc.sh <ccache|sccache> <nvcc> <nvcc args...>
#
# torch's ninja rule passes nvcc `--generate-dependencies-with-compile
# --dependency-output FILE`. ccache does not know that the long option takes
# an argument, so it treats FILE as a second source (or fails to preprocess)
# and never caches CUDA objects. nvcc accepts the equivalent `-MD -MF FILE`,
# which ccache understands.
launcher="$1"
nvcc="$2"
shift 2
args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --generate-dependencies-with-compile) args+=(-MD) ;;
        --dependency-output) args+=(-MF "$2"); shift ;;
        *) args+=("$1") ;;
    esac
    shift
done
exec "$launcher" "$nvcc" "${args[@]}"
