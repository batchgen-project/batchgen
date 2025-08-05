#!/usr/bin/env bash
set -x

# Usage:
#   ./build.sh 3.10 12.8 [x86_64|aarch64] [--debug]
#
# Examples:
#   ./build.sh 3.10 12.8
#   ./build.sh 3.10 12.8 x86_64 --debug

PYTHON_VERSION=${1:?Need Python version, e.g. 3.10}
CUDA_VERSION=${2:?Need CUDA version, e.g. 12.8}
ARCH=${3:-$(uname -m)}
MODE=${4:-""}

if [[ "${ARCH}" == "aarch64" ]]; then
  BUILDER="pytorch/manylinuxaarch64-builder:cuda${CUDA_VERSION}"
  LIBCUDA_ARCH="sbsa"
else
  BUILDER="pytorch/manylinux2_28-builder:cuda${CUDA_VERSION}"
  LIBCUDA_ARCH="${ARCH}"
fi

# cache directories to speed up builds

# interactive mode: run devshell.sh with menu
if [[ "${MODE}" == "--debug" ]]; then
   docker run --rm -it \
    -v "$(pwd)":/mgn-kernel \
    -e PYTHON_VERSION="${PYTHON_VERSION}" \
    -e MGN_CUDA_VERSION="${CUDA_VERSION}" \
    -e ARCH="${ARCH}" \
    -e LIBCUDA_ARCH="${LIBCUDA_ARCH}" \
    "${BUILDER}" \
    bash -lc '/mgn-kernel/scripts/devshell.sh --menu'
  exit 0
fi

# non-interactive mode: build once
docker run --rm \
  -v "$(pwd)":/mgn-kernel \
  -e PYTHON_VERSION="${PYTHON_VERSION}" \
  -e MGN_CUDA_VERSION="${CUDA_VERSION}" \
  -e ARCH="${ARCH}" \
  -e LIBCUDA_ARCH="${LIBCUDA_ARCH}" \
  "${BUILDER}" \
  bash -lc '/mgn-kernel/scripts/devshell.sh --build-once'