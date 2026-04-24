"""Sanity + perf unit test for the UVA-page-copy kernel used by Phase A's
AsyncAppendDecodeKVToHostBatchedKernel.

Mirrors the on-device behavior of core/KV_Storage/uva_copy_kernel.cu.
Runs a minimal inline build of the same kernel, drives it with the
exact shape pattern the real code issues per decode step (78 layers ×
batch_size × per-token bytes), and compares against the 1248-copy
cudaMemcpyAsync baseline.

Run on H20:
    docker exec tairan-batchgen bash -c 'source /root/miniconda3/etc/profile.d/conda.sh \
      && conda activate batchgen \
      && cd /data2/tairan/workspace/batchgen_unit_test \
      && python tests/cuda_graph/test_uva_page_copy_dtoh.py'
"""
from __future__ import annotations

import argparse
import ctypes
import os
import time

import torch
from torch.utils.cpp_extension import load_inline


# Replica of core/KV_Storage/uva_copy_kernel.cu. Direction-agnostic:
# dst[i] = src[i] with uint4 vectorized loads/stores. For DtoH, src is
# a device ptr and dst is a UVA-mapped pinned-host ptr.
_SRC = r"""
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cstdint>

__global__ void uva_page_copy_kernel(
        uint8_t** src_ptrs, uint8_t** dst_ptrs,
        std::size_t page_size_bytes, int num_pages) {
    const int page_idx = blockIdx.x;
    if (page_idx >= num_pages) return;
    uint8_t* src = src_ptrs[page_idx];
    uint8_t* dst = dst_ptrs[page_idx];
    const int num_words = static_cast<int>(page_size_bytes / sizeof(uint4));
    uint4* src_v = reinterpret_cast<uint4*>(src);
    uint4* dst_v = reinterpret_cast<uint4*>(dst);
    for (int i = threadIdx.x; i < num_words; i += blockDim.x) {
        dst_v[i] = src_v[i];
    }
}

void launch_uva_page_copy(
        uintptr_t src_ptrs_dev, uintptr_t dst_ptrs_dev,
        int64_t page_size_bytes, int num_pages, int64_t stream_ptr) {
    if (num_pages <= 0 || page_size_bytes == 0) return;
    constexpr int kThreadsPerBlock = 256;
    const dim3 grid(static_cast<unsigned int>(num_pages));
    const dim3 block(kThreadsPerBlock);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    uva_page_copy_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<uint8_t**>(src_ptrs_dev),
        reinterpret_cast<uint8_t**>(dst_ptrs_dev),
        static_cast<std::size_t>(page_size_bytes), num_pages);
}
"""


def _build_extension():
    return load_inline(
        name="uva_page_copy_test",
        cpp_sources=[
            "void launch_uva_page_copy(uintptr_t, uintptr_t, int64_t, int, int64_t);",
        ],
        cuda_sources=[_SRC],
        functions=["launch_uva_page_copy"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )


def _register_host(mem_ptr: int, nbytes: int) -> None:
    """cudaHostRegister an existing CPU pointer with default flags so it
    becomes UVA-mapped (device-accessible via zero-copy PCIe)."""
    cudart = ctypes.CDLL("libcudart.so")
    CUDA_HOST_REGISTER_DEFAULT = 0x0
    err = cudart.cudaHostRegister(
        ctypes.c_void_p(mem_ptr),
        ctypes.c_size_t(nbytes),
        ctypes.c_uint(CUDA_HOST_REGISTER_DEFAULT),
    )
    if err != 0:
        raise RuntimeError(f"cudaHostRegister failed, rc={err}")


def _unregister_host(mem_ptr: int) -> None:
    cudart = ctypes.CDLL("libcudart.so")
    cudart.cudaHostUnregister(ctypes.c_void_p(mem_ptr))


def run_correctness(ext, num_layers: int, batch: int, token_bytes: int):
    """End-to-end correctness: fill GPU K tensors with known data, launch
    the UVA kernel writing to scattered host-pinned destinations,
    verify every (layer, seq) slot matches bit-exactly.
    """
    assert token_bytes % 16 == 0, "uint4 vectorization requires 16-byte alignment"
    num_pages = num_layers * batch

    # Source: per-layer GPU K tensor, shape [batch, token_bytes / elem_size],
    # but we operate at byte granularity for the kernel — allocate one
    # big GPU buffer per layer and carve per-seq pointers.
    gpu_layers = [
        torch.empty(batch * token_bytes, dtype=torch.uint8, device="cuda")
        for _ in range(num_layers)
    ]
    # Fill with deterministic pattern derived from (layer, seq, byte_idx)
    for li, t in enumerate(gpu_layers):
        flat = torch.arange(batch * token_bytes, dtype=torch.int32, device="cuda")
        flat = (flat + li * 1_000_003).to(torch.uint8)  # cheap pseudo-random per-byte
        t.copy_(flat)

    # Destination: a CPU pinned tensor per layer, each of shape [batch, token_bytes],
    # but addresses per-seq are scattered across pages (simulate by permuting
    # which "page row" each seq writes to, so dst pointers are truly scattered).
    # We allocate a single big CPU buffer and hand out scattered slots.
    total_bytes = num_pages * token_bytes * 4  # 4x slop so we can scatter
    # torch pin_memory=True already cudaHostRegisters internally — UVA-mapped.
    cpu_buf = torch.empty(total_bytes, dtype=torch.uint8, pin_memory=True)
    cpu_ptr_base = cpu_buf.data_ptr()
    assert cpu_buf.is_pinned(), "pin_memory=True did not pin the allocation"
    try:
        # Build per-(layer, seq) dst offsets — intentionally non-contiguous
        torch.manual_seed(0)
        dst_offsets = torch.randperm(num_pages) * (token_bytes * 3) + 128
        dst_offsets = dst_offsets.tolist()

        src_ptrs = []
        dst_ptrs = []
        expected_map = {}  # (layer_idx, seq_idx) -> expected bytes tensor
        for li in range(num_layers):
            layer_gpu_ptr = gpu_layers[li].data_ptr()
            for b in range(batch):
                src = layer_gpu_ptr + b * token_bytes
                dst = cpu_ptr_base + dst_offsets[li * batch + b]
                src_ptrs.append(src)
                dst_ptrs.append(dst)
                expected_map[(li, b)] = (
                    gpu_layers[li][b * token_bytes : (b + 1) * token_bytes]
                    .cpu()
                    .clone()
                )

        # Upload src/dst ptr arrays to device
        src_ptrs_tensor = torch.tensor(src_ptrs, dtype=torch.int64, device="cuda")
        dst_ptrs_tensor = torch.tensor(dst_ptrs, dtype=torch.int64, device="cuda")

        stream = torch.cuda.current_stream().cuda_stream
        ext.launch_uva_page_copy(
            src_ptrs_tensor.data_ptr(),
            dst_ptrs_tensor.data_ptr(),
            token_bytes,
            num_pages,
            stream,
        )
        torch.cuda.synchronize()

        # Verify
        mismatches = 0
        for li in range(num_layers):
            for b in range(batch):
                dst_off = dst_offsets[li * batch + b]
                got = cpu_buf[dst_off : dst_off + token_bytes].clone()
                exp = expected_map[(li, b)]
                if not torch.equal(got, exp):
                    mismatches += 1
                    if mismatches <= 3:
                        print(
                            f"  mismatch at (layer={li}, seq={b}): "
                            f"first 8 got={got[:8].tolist()} exp={exp[:8].tolist()}"
                        )
        if mismatches > 0:
            raise AssertionError(
                f"{mismatches}/{num_pages} (layer, seq) slots mismatched"
            )
        print(
            f"[CORRECTNESS] {num_pages} slots × {token_bytes}B — "
            f"{num_layers} layers × {batch} seqs — all bit-exact ✓"
        )
    finally:
        pass  # torch pinned allocation auto-unregisters on dealloc


def run_perf(ext, num_layers: int, batch: int, token_bytes: int,
             warmup: int = 3, iters: int = 20):
    """Compare UVA kernel vs the 1248-cudaMemcpyAsync baseline."""
    num_pages = num_layers * batch

    gpu_layers = [
        torch.empty(batch * token_bytes, dtype=torch.uint8, device="cuda")
        for _ in range(num_layers)
    ]
    for t in gpu_layers:
        t.fill_(0xA5)

    total_bytes = num_pages * token_bytes * 4
    cpu_buf = torch.empty(total_bytes, dtype=torch.uint8, pin_memory=True)
    cpu_ptr_base = cpu_buf.data_ptr()
    assert cpu_buf.is_pinned()
    try:
        torch.manual_seed(0)
        dst_offsets = (torch.randperm(num_pages) * (token_bytes * 3) + 128).tolist()

        src_ptrs, dst_ptrs = [], []
        for li in range(num_layers):
            layer_gpu_ptr = gpu_layers[li].data_ptr()
            for b in range(batch):
                src_ptrs.append(layer_gpu_ptr + b * token_bytes)
                dst_ptrs.append(cpu_ptr_base + dst_offsets[li * batch + b])

        src_ptrs_tensor = torch.tensor(src_ptrs, dtype=torch.int64, device="cuda")
        dst_ptrs_tensor = torch.tensor(dst_ptrs, dtype=torch.int64, device="cuda")

        stream_handle = torch.cuda.current_stream().cuda_stream

        # ---- UVA kernel path ----
        for _ in range(warmup):
            ext.launch_uva_page_copy(
                src_ptrs_tensor.data_ptr(),
                dst_ptrs_tensor.data_ptr(),
                token_bytes,
                num_pages,
                stream_handle,
            )
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(iters):
            ext.launch_uva_page_copy(
                src_ptrs_tensor.data_ptr(),
                dst_ptrs_tensor.data_ptr(),
                token_bytes,
                num_pages,
                stream_handle,
            )
        torch.cuda.synchronize()
        uva_ms = (time.perf_counter() - t0) * 1000 / iters

        # ---- Baseline: 1248 cudaMemcpyAsync ----
        cudart = ctypes.CDLL("libcudart.so")
        cudaMemcpyDeviceToHost = 2
        # Prepare raw ptrs
        def _issue_baseline():
            for li in range(num_layers):
                layer_gpu_ptr = gpu_layers[li].data_ptr()
                for b in range(batch):
                    cudart.cudaMemcpyAsync(
                        ctypes.c_void_p(cpu_ptr_base + dst_offsets[li * batch + b]),
                        ctypes.c_void_p(layer_gpu_ptr + b * token_bytes),
                        ctypes.c_size_t(token_bytes),
                        cudaMemcpyDeviceToHost,
                        ctypes.c_void_p(stream_handle),
                    )

        for _ in range(warmup):
            _issue_baseline()
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(iters):
            _issue_baseline()
        torch.cuda.synchronize()
        baseline_ms = (time.perf_counter() - t0) * 1000 / iters

        speedup = baseline_ms / uva_ms if uva_ms > 0 else float("inf")
        print(
            f"[PERF] shape: {num_layers} layers × {batch} seqs × {token_bytes}B "
            f"= {num_pages} slots, {num_pages * token_bytes / 1024:.1f} KiB/iter"
        )
        print(f"[PERF] UVA kernel path    : {uva_ms:7.3f} ms/iter")
        print(f"[PERF] cudaMemcpyAsync × N: {baseline_ms:7.3f} ms/iter  "
              f"({num_pages} calls)")
        print(f"[PERF] speedup            : {speedup:6.1f}×")
    finally:
        pass


def main():
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-layers", type=int, default=78)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--token-bytes-primary", type=int, default=1152,
                        help="primary MLA K per-token bytes (576 × bf16)")
    parser.add_argument("--token-bytes-aux", type=int, default=256,
                        help="aux indexer K per-token bytes (128 × bf16)")
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    print("Building inline CUDA extension...")
    ext = _build_extension()
    print("Extension built.")
    print()

    # Small correctness check first (keeps the ground-truth read cost low)
    print("== Correctness: primary cache shape ==")
    run_correctness(ext, num_layers=4, batch=args.batch,
                    token_bytes=args.token_bytes_primary)
    print("== Correctness: aux cache shape ==")
    run_correctness(ext, num_layers=4, batch=args.batch,
                    token_bytes=args.token_bytes_aux)

    # Full-shape perf on primary + aux
    print()
    print("== Perf: primary cache (MLA K) ==")
    run_perf(ext, args.num_layers, args.batch, args.token_bytes_primary,
             iters=args.iters)
    print()
    print("== Perf: aux cache (indexer K) ==")
    run_perf(ext, args.num_layers, args.batch, args.token_bytes_aux,
             iters=args.iters)


if __name__ == "__main__":
    main()
