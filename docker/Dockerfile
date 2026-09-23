ARG CUDA_VERSION=12.9.1
ARG PYTHON_VERSION=3.11.13
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu22.04
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG PYTHON_VERSION
# Build target arch for FlashMLA + batchgen_kernels:
#   sm90a (default) -> Hopper; FlashMLA SM100 kernels disabled.
#   sm100 / all     -> Blackwell (B200); FlashMLA SM100 kernels enabled
#                      (requires the CUDA 12.9+ base image above).
# Override at build time: docker build --build-arg BUILD_ARCH=sm100 ...
ARG BUILD_ARCH=sm90a
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    HF_ENDPOINT=https://hf-mirror.com

# RDMA Python UV
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget git fish libnuma-dev \
    build-essential cmake \
    libibverbs1 libibverbs-dev ibverbs-providers rdma-core \
    vim tmux htop tree silversearcher-ag lsof unzip \
    && rm -rf /var/lib/apt/lists/*

# Install uv (with retry and proper error handling)
RUN set -o pipefail && \
    for i in 1 2 3 4 5; do \
      curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL=/usr/local/bin sh && break || \
      { echo "Attempt $i failed, retrying in 10s..."; sleep 10; }; \
    done && \
    uv --version

    
WORKDIR /root/moegen
RUN uv venv --python ${PYTHON_VERSION} --seed

ENV PATH="/root/moegen/.venv/bin:$PATH"

# Make sure to install ninja to enable fast builds
RUN uv pip install torch==2.9.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128 \
    setuptools wheel packaging ninja \
    && uv cache clean
    
# Install Flash Attention
RUN git clone --recursive https://github.com/Dao-AILab/flash-attention.git \
    && cd flash-attention \
    && git checkout v2.8.2 \
    && cd hopper \
    && FLASH_ATTENTION_FORCE_BUILD=TRUE uv pip install . --no-build-isolation \
    && uv cache clean

# Install FlashMLA
# SM100 (Blackwell) kernels are only built for sm100/all targets; the default
# Hopper (sm90a) build disables them to match upstream's Hopper-only path.
RUN git clone --recursive https://github.com/deepseek-ai/FlashMLA.git \
    && cd FlashMLA \
    && git checkout 1408756a88e52a25196b759eaf8db89d2b51b5a1 \
    && git submodule update --init --recursive \
    && if [ "$BUILD_ARCH" = "sm90a" ]; then \
         FLASH_MLA_DISABLE_SM100=1 uv pip install -v . --no-build-isolation; \
       else \
         uv pip install -v . --no-build-isolation; \
       fi \
    && uv cache clean

# Install DeepGEMM (sgl-project fork, supported build_sgl_deep_gemm.sh path).
# The build needs apache-tvm-ffi first. Exactly one wheel must be produced; a
# py3-none-any tag is normalised to the manylinux platform tag before install.
# The final check pins the distribution AND the fp8_mqa_logits signature.
RUN git clone --recursive https://github.com/sgl-project/DeepGEMM.git \
    && cd DeepGEMM \
    && git checkout v0.1.5.post3 \
    && git submodule update --init --recursive \
    && uv pip install apache-tvm-ffi==0.1.11 wheel==0.45.1 \
    && bash ./build_sgl_deep_gemm.sh \
    && DG_WHEEL="$(find ./dist -maxdepth 1 -type f -name 'sgl_deep_gemm-*.whl' | sort)" \
    && [ "$(printf '%s\n' "$DG_WHEEL" | grep -c .)" = "1" ] \
    && if [[ "$DG_WHEEL" == *-py3-none-any.whl ]]; then \
         python -m wheel tags --platform-tag manylinux2014_x86_64 --remove "$DG_WHEEL"; \
         DG_WHEEL="${DG_WHEEL%-py3-none-any.whl}-py3-none-manylinux2014_x86_64.whl"; \
       fi \
    && uv pip install "$DG_WHEEL" \
    && python -c "import inspect; from importlib.metadata import version; import deep_gemm; assert version('sgl-deep-gemm') == '0.1.5.post3'; assert 'max_seqlen_k' in inspect.signature(deep_gemm.fp8_mqa_logits).parameters" \
    && uv cache clean

# Install from moe gen source
COPY . /root/moegen

# Install batchgen_kernels (AOT-compiled CUDA extensions)
# BUILD_ARCH selects the kernel target (sm90a default / sm100 for Blackwell).
RUN cd /root/moegen/batchgen_kernels \
    && BUILD_ARCH=${BUILD_ARCH} uv pip install . --no-build-isolation \
    && uv cache clean

# Install BatchGen
RUN uv pip install -r requirements.txt && uv pip install . -v \
    && uv cache clean

ENV NCCL_BUFFSIZE=16777216
 
CMD ["sleep", "infinity"]
