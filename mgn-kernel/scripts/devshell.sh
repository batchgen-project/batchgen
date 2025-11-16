#!/usr/bin/env bash
set -x

# Called inside container by build.sh
# Modes:
#   --menu       : interactive menu
#   --build-once : do one shot build and exit

PY="${PYTHON_VERSION:-3.10}"
CU="${MGN_CUDA_VERSION:-12.8}"
ARCH="${ARCH:-x86_64}"
LIBCUDA_ARCH="${LIBCUDA_ARCH:-${ARCH}}"
PYROOT="/opt/python/cp${PY//.}-cp${PY//.}"
PYSITE="${PYROOT}/lib/python${PY}/site-packages"

ensure_deps() {
  echo "==> Ensuring toolchain & dependencies (idempotent)..."

  # Use local CMake path if already installed
  CMAKE_MAJ=3.31
  CMAKE_MIN=1
  CMAKE_PATH="/opt/cmake/bin/cmake"
  CMAKE_VERSION_OK=false

  if [ -x "$CMAKE_PATH" ]; then
    INSTALLED_VERSION=$($CMAKE_PATH --version | head -n1 | awk '{print $3}')
    if printf '%s\n' "$INSTALLED_VERSION" "3.26.0" | sort -V -C; then
      CMAKE_VERSION_OK=true
    fi
  fi

  if ! $CMAKE_VERSION_OK; then
    echo "Installing CMake ${CMAKE_MAJ}.${CMAKE_MIN}..."
    wget https://cmake.org/files/v${CMAKE_MAJ}/cmake-${CMAKE_MAJ}.${CMAKE_MIN}-linux-${ARCH}.tar.gz
    tar -xzf cmake-${CMAKE_MAJ}.${CMAKE_MIN}-linux-${ARCH}.tar.gz
    mv cmake-${CMAKE_MAJ}.${CMAKE_MIN}-linux-${ARCH} /opt/cmake
    ln -sf /opt/cmake/bin/cmake /usr/local/bin/cmake
    rm -f cmake-${CMAKE_MAJ}.${CMAKE_MIN}-linux-${ARCH}.tar.gz
  fi

  export PATH="/opt/cmake/bin:${PATH}"
  cmake --version

  # Core dependencies
  yum -y install numactl-devel || true
  ln -sf /usr/lib64/libibverbs.so.1 /usr/lib64/libibverbs.so || true

  # PyTorch & build tools
  TORCH_EXTRA=""
  if [[ "${CU}" == "12.8" ]]; then
    TORCH_EXTRA="--extra-index-url https://download.pytorch.org/whl/cu${CU//.}"
  fi

  "${PYROOT}/bin/pip" install --no-cache-dir \
    torch==2.9.0 ${TORCH_EXTRA} \
    ninja setuptools==75.0.0 wheel==0.41.0 numpy uv scikit-build-core

  # CUDA stub lib
  mkdir -p /usr/lib/${ARCH}-linux-gnu/
  ln -sf /usr/local/cuda-${CU}/targets/${LIBCUDA_ARCH}-linux/lib/stubs/libcuda.so \
         /usr/lib/${ARCH}-linux-gnu/libcuda.so || true

  # Set environment
  export LD_LIBRARY_PATH="/lib64:${LD_LIBRARY_PATH:-}"
  export TORCH_CUDA_ARCH_LIST="7.5 8.0 8.9 9.0+PTX"
  export CUDA_VERSION="${CU}"

  echo "==> Toolchain ready."
}

task_build() {
  echo "==> Building wheel..."
  cd /mgn-kernel
  if [[ ! -x ./scripts/rename_wheels.sh ]]; then
    echo "WARN: rename_wheels.sh not found or not executable." >&2
  fi

  # build dir
  PYTHONPATH="${PYSITE}" \
  "${PYROOT}/bin/python" -m uv build --wheel -Cbuild-dir=build . --color=always --no-build-isolation

  if [[ -x ./scripts/rename_wheels.sh ]]; then
    ./scripts/rename_wheels.sh
  fi
  echo "==> Done. Wheels under ./dist"
}

task_rebuild_clean() {
  echo "==> Cleaning and rebuilding..."
  cd /mgn-kernel
  rm -rf build dist
  task_build
}

task_env() {
  echo "---- Env ----"
  echo "Python    : ${PY}"
  echo "CUDA      : ${CU}"
  echo "ARCH      : ${ARCH}"
  echo "PYROOT    : ${PYROOT}"
  echo "PYSITE    : ${PYSITE}"
  which cmake || true
  cmake --version || true
  "${PYROOT}/bin/python" -c 'import torch,sys;print("torch:",torch.__version__,"cuda:",torch.version.cuda,"py:",sys.version.split()[0])' || true
  echo "------------"
}

menu_loop() {
  ensure_deps
  PS3=$'\nSelect action: '
  select opt in "build" "rebuild" "test" "env" "shell" "exit"; do
    case "$opt" in
      build)   task_build ;;
      rebuild) task_rebuild_clean ;;
      env)     task_env ;;
      shell)   echo "==> Entering shell (exit to return)"; bash ;;
      exit)    break ;;
      *)       echo "Invalid." ;;
    esac
  done
}

main() {
  case "${1:-}" in
    --menu)       menu_loop ;;
    --build-once) ensure_deps; task_build ;;
    *)            echo "Usage: devshell.sh [--menu|--build-once]"; exit 1 ;;
  esac
}

main "$@"