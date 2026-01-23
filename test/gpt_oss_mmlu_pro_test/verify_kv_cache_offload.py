"""Verify KV cache prefill → host offload correctness.

This test verifies that:
1. KV cache is correctly written during prefill on GPU
2. KV values are correctly transferred to host paged KV manager
3. GPU and host KV values match at corresponding positions

The KV cache flow in BatchGen:
1. Prefill: Model computes K, V for all input tokens
2. GPU write: K, V written to GPU paged KV cache via update_layer_prefill()
3. Host callback: kv_append_callback sends K, V to host paged KV manager
4. Decode: Attention reads from GPU KV cache

Potential issues this test can catch:
- Wrong positions being written (off-by-one errors)
- Page table mapping issues
- Host KV not receiving correct values
- Shape/layout mismatches between GPU and host

Usage:
    python verify_kv_cache_offload.py --basic
    python verify_kv_cache_offload.py --gpu-only
"""

import argparse
import ctypes
import errno
import os
import random
import string
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

# For shm cleanup on Linux
try:
    _libc = ctypes.CDLL("libc.so.6", use_errno=True)
    _HAS_SHM_UNLINK = True
except OSError:
    _HAS_SHM_UNLINK = False


def _shm_unlink(name: str) -> None:
    """Unlink shared memory region (Linux only)."""
    if not _HAS_SHM_UNLINK or not name:
        return
    res = _libc.shm_unlink(name.encode("utf-8"))
    if res != 0:
        err = ctypes.get_errno()
        if err != errno.ENOENT:
            print(f"Warning: shm_unlink({name}) failed with errno {err}")

# Add BatchGen to path
BATCHGEN_PATH = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BATCHGEN_PATH))


def create_traceable_kv_tensors(
    batch_size: int,
    seq_len: int,
    num_kv_heads: int,
    head_dim: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create K, V tensors with traceable values for testing.

    Values are set to allow easy identification:
    - K[b, s, h, d] = b * 1000 + s * 10 + h + d * 0.01
    - V[b, s, h, d] = -(b * 1000 + s * 10 + h + d * 0.01)

    This makes it easy to verify which batch/seq/head the values came from.
    """
    k = torch.zeros(batch_size, seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)
    v = torch.zeros(batch_size, seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)

    for b in range(batch_size):
        for s in range(seq_len):
            for h in range(num_kv_heads):
                for d in range(head_dim):
                    val = b * 1000 + s * 10 + h + d * 0.01
                    k[b, s, h, d] = val
                    v[b, s, h, d] = -val

    return k, v


def _random_shm_name() -> str:
    """Generate a random shared memory name."""
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=10))
    return f"/batchgen_kv_test_{suffix}"


def test_gpu_kv_write_only():
    """Test KV write to GPU paged cache (without host offload).

    This tests the basic GPU KV cache write functionality.
    """
    print("=" * 60)
    print("GPU KV Cache Write Verification")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU test")
        return

    try:
        from batchgen.kv_cache.gpu_paged_kv_manager import (
            GPUPagedKVCacheManager,
            GPUPagedKVConfig,
        )
    except ImportError as e:
        print(f"Could not import GPU KV manager: {e}")
        return

    # Configuration (GPT-OSS-120B style)
    num_layers = 2
    num_kv_heads = 8
    head_dim = 64
    page_size = 64
    num_pages = 100

    # Test scenario
    batch_size = 3
    seq_lengths = [10, 25, 50]
    max_seq_len = max(seq_lengths)

    print(f"\nConfiguration:")
    print(f"  num_layers: {num_layers}")
    print(f"  num_kv_heads: {num_kv_heads}")
    print(f"  head_dim: {head_dim}")
    print(f"  page_size: {page_size}")
    print(f"  batch_size: {batch_size}")
    print(f"  seq_lengths: {seq_lengths}")

    # Create GPU manager
    config = GPUPagedKVConfig(
        num_layers=num_layers,
        num_pages=num_pages,
        page_size_tokens=page_size,
        num_k_heads=num_kv_heads,
        k_head_dim=head_dim,
        num_v_heads=num_kv_heads,
        v_head_dim=head_dim,
        kv_dtype=torch.bfloat16,
    )

    device = torch.device("cuda:0")
    gpu_manager = GPUPagedKVCacheManager(config=config, device=device)
    gpu_manager.initialize()
    print(f"\nGPU KV Manager created on {device}")

    # Create mock KV tensors
    print("\nCreating mock KV tensors...")
    k_mock, v_mock = create_traceable_kv_tensors(
        batch_size, max_seq_len, num_kv_heads, head_dim, device="cuda:0"
    )
    print(f"  K shape: {k_mock.shape}")
    print(f"  V shape: {v_mock.shape}")

    # Allocate pages for sequences
    print("\nAllocating pages...")
    sequence_ids = [100, 101, 102]
    gpu_manager.allocate_pages_for_sequences(sequence_ids, [max_seq_len] * batch_size)
    gpu_manager.rebuild_page_table(sequence_ids)
    print(f"  Allocated for sequences: {sequence_ids}")

    # Write KV for each sequence at each layer using prefill API
    print("\nWriting KV to GPU cache...")
    for layer_idx in range(num_layers):
        for b, (seq_id, seq_len) in enumerate(zip(sequence_ids, seq_lengths)):
            # For prefill, use update_layer_prefill which writes multiple tokens
            k_seq = k_mock[b:b+1, :seq_len].contiguous()  # [1, seq_len, heads, dim]
            v_seq = v_mock[b:b+1, :seq_len].contiguous()

            # Note: The actual prefill API depends on implementation
            # For now, write token by token using decode API for verification
            for pos in range(seq_len):
                k_token = k_mock[b:b+1, pos:pos+1]  # [1, 1, heads, dim]
                v_token = v_mock[b:b+1, pos:pos+1]
                seq_len_tensor = torch.tensor([pos], dtype=torch.int32, device=device)

                # This writes K, V for token at position `pos`
                # The decode API writes at position = sequence_length[b]
                # So we need sequence_length = pos (0-indexed position we want to write)
                gpu_manager.update_layer_decode_new_token(
                    k_tensor=k_token,
                    v_tensor=v_token,
                    sequence_lengths=seq_len_tensor,
                    layer_idx=layer_idx,
                )

    print("  KV written to GPU cache")

    # Verify GPU KV values
    print("\n" + "-" * 40)
    print("Verifying GPU KV cache contents...")
    print("-" * 40)

    all_pass = True
    for layer_idx in range(num_layers):
        k_cache_layer = gpu_manager._k_cache[layer_idx]  # [num_pages, page_size, heads, dim]
        v_cache_layer = gpu_manager._v_cache[layer_idx] if gpu_manager._v_cache is not None else None

        print(f"\nLayer {layer_idx}:")

        for b, (seq_id, seq_len) in enumerate(zip(sequence_ids, seq_lengths)):
            state = gpu_manager._get_sequence_state(seq_id)
            mismatch_count = 0

            for pos in range(seq_len):
                # Get the GPU page and offset for this position
                gpu_page, offset = gpu_manager._resolve_token_location(
                    state, seq_id, pos, "verify"
                )

                # Read from GPU cache
                k_actual = k_cache_layer[gpu_page, offset]  # [heads, dim]

                # Expected values
                k_expected = k_mock[b, pos]  # [heads, dim]

                # Compare
                if not torch.allclose(k_actual, k_expected, atol=1e-2):
                    mismatch_count += 1
                    if mismatch_count <= 3:  # Print first 3 mismatches
                        print(f"  MISMATCH: seq={seq_id}, pos={pos}, page={gpu_page}, offset={offset}")
                        print(f"    Expected K[0,:4]: {k_expected[0,:4].cpu().tolist()}")
                        print(f"    Actual K[0,:4]: {k_actual[0,:4].cpu().tolist()}")

            if mismatch_count == 0:
                print(f"  Seq {seq_id}: OK ({seq_len} positions verified)")
            else:
                print(f"  Seq {seq_id}: FAILED ({mismatch_count}/{seq_len} mismatches)")
                all_pass = False

    if all_pass:
        print("\n✓ GPU KV cache verification PASSED")
    else:
        print("\n✗ GPU KV cache verification FAILED")

    return all_pass


def test_kv_offload_to_host():
    """Test KV write to GPU then offload to host.

    This is the main test that mimics the actual inference path:
    1. Write KV to GPU during prefill
    2. Offload KV to host paged KV manager
    3. Verify host KV matches GPU KV
    """
    print("=" * 60)
    print("GPU → Host KV Offload Verification")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping offload test")
        return False

    try:
        from batchgen.models.engine_loader import core_engine as bg
    except ImportError as e:
        print(f"Could not import core_engine: {e}")
        print("Make sure BatchGen is properly built with C++ bindings")
        return False

    # Configuration (GPT-OSS-120B style with GQA)
    num_layers = 2
    num_kv_heads = 8
    k_head_dim = 64  # For GQA, K and V have same shape
    v_head_dim = 64
    page_size = 64
    num_pages = 200

    # Test scenario
    batch_size = 3
    seq_lengths = [10, 25, 50]
    max_seq_len = max(seq_lengths)

    print(f"\nConfiguration:")
    print(f"  num_layers: {num_layers}")
    print(f"  num_kv_heads: {num_kv_heads}")
    print(f"  k_head_dim: {k_head_dim}, v_head_dim: {v_head_dim}")
    print(f"  page_size: {page_size}")
    print(f"  batch_size: {batch_size}")
    print(f"  seq_lengths: {seq_lengths}")

    shm_name = _random_shm_name()
    device_idx = 0

    # Create host paged KV config (GQA style, both K and V)
    host_cfg = bg.HostPagedKVConfig()
    host_cfg.shm_name = shm_name
    host_cfg.num_layers = num_layers
    host_cfg.num_pages = num_pages
    host_cfg.page_size_tokens = page_size
    host_cfg.num_k_heads = num_kv_heads
    host_cfg.k_head_dim = k_head_dim
    host_cfg.num_v_heads = num_kv_heads  # GQA: V has same head count as K
    host_cfg.v_head_dim = v_head_dim
    host_cfg.k_element_size_bytes = 2  # bfloat16
    host_cfg.v_element_size_bytes = 2  # bfloat16
    host_cfg.sequence_table_capacity = 1024
    host_cfg.alignment_bytes = 64

    # Create manager and worker
    # Use DefaultHostPagedKVManager for GQA (stores both K and V)
    # Use MLAHostPagedKVManager for MLA (stores only K)
    manager = bg.DefaultHostPagedKVManager(host_cfg)
    manager.initialize(True)  # Create shared memory
    print(f"\nHost KV Manager created (shm={shm_name})")

    worker = bg.DefaultHostPagedKVWorkerView(host_cfg)
    worker.initialize(device_idx, False)  # Attach to existing shm
    print(f"Host KV Worker attached on device {device_idx}")

    # Create mock KV tensors on GPU
    print("\nCreating mock KV tensors...")
    k_mock, v_mock = create_traceable_kv_tensors(
        batch_size, max_seq_len, num_kv_heads, k_head_dim,
        device=f"cuda:{device_idx}"
    )
    print(f"  K shape: {k_mock.shape}")
    print(f"  V shape: {v_mock.shape}")

    # Register sequences and allocate pages
    sequence_ids = [100, 101, 102]
    worker.register_sequences(sequence_ids)

    capacity_tokens = max_seq_len + page_size  # Some extra capacity
    requests = [(seq_id, capacity_tokens) for seq_id in sequence_ids]
    allocations = worker.allocate_pages_for_sequences(requests)
    print(f"\nAllocated pages: {[len(a) for a in allocations]}")

    # Offload KV for each layer and sequence
    print("\nOffloading KV to host...")
    for layer_idx in range(num_layers):
        # Prepare KV tensors: [batch, seq_len, heads, dim]
        # For offload, we need to send each sequence separately
        for b, (seq_id, seq_len) in enumerate(zip(sequence_ids, seq_lengths)):
            k_seq = k_mock[b:b+1, :seq_len].contiguous()  # [1, seq_len, heads, dim]
            v_seq = v_mock[b:b+1, :seq_len].contiguous()

            # Async offload to host
            task = worker.async_offload_layer_kv_to_host(
                layer_idx=layer_idx,
                sequence_ids=[seq_id],
                k_tensor=k_seq,
                v_tensor=v_seq,
                sequence_lengths=[seq_len],
            )
            task.wait()  # Wait for completion

    print("  Offload complete")
    torch.cuda.synchronize()

    # Verify host KV values
    print("\n" + "-" * 40)
    print("Verifying Host KV cache contents...")
    print("-" * 40)

    all_pass = True
    token_k_elems = num_kv_heads * k_head_dim
    token_v_elems = num_kv_heads * v_head_dim
    token_k_bytes = token_k_elems * 2  # bfloat16
    token_v_bytes = token_v_elems * 2

    for layer_idx in range(num_layers):
        print(f"\nLayer {layer_idx}:")

        for b, (seq_id, seq_len) in enumerate(zip(sequence_ids, seq_lengths)):
            # Get page pointers from host manager
            k_ptrs, v_ptrs = manager.get_sequence_layer_page_pointers(seq_id, layer_idx)

            pages_needed = (seq_len + page_size - 1) // page_size
            print(f"  Seq {seq_id}: {seq_len} tokens, {pages_needed} pages, got {len(k_ptrs)} k_ptrs")

            mismatch_count = 0
            for pos in range(seq_len):
                page_idx = pos // page_size
                page_offset = pos % page_size

                # Get K pointer for this token
                k_ptr = k_ptrs[page_idx] + page_offset * token_k_bytes

                # Read K from host memory
                k_array_type = ctypes.c_uint16 * token_k_elems
                k_host_array = k_array_type.from_address(k_ptr)
                k_host_buf = torch.frombuffer(k_host_array, dtype=torch.uint16)
                k_host = k_host_buf.view(torch.bfloat16).reshape(num_kv_heads, k_head_dim)

                # Expected K
                k_expected = k_mock[b, pos].cpu()  # [heads, dim]

                # Compare
                if not torch.allclose(k_host, k_expected, atol=1e-2):
                    mismatch_count += 1
                    if mismatch_count <= 3:
                        print(f"    K MISMATCH at pos {pos}: page={page_idx}, offset={page_offset}")
                        print(f"      Expected[0,:4]: {k_expected[0,:4].tolist()}")
                        print(f"      Host[0,:4]: {k_host[0,:4].tolist()}")

                # Also verify V if pointers available
                if v_ptrs:
                    v_ptr = v_ptrs[page_idx] + page_offset * token_v_bytes
                    v_array_type = ctypes.c_uint16 * token_v_elems
                    v_host_array = v_array_type.from_address(v_ptr)
                    v_host_buf = torch.frombuffer(v_host_array, dtype=torch.uint16)
                    v_host = v_host_buf.view(torch.bfloat16).reshape(num_kv_heads, v_head_dim)

                    v_expected = v_mock[b, pos].cpu()
                    if not torch.allclose(v_host, v_expected, atol=1e-2):
                        mismatch_count += 1
                        if mismatch_count <= 3:
                            print(f"    V MISMATCH at pos {pos}")

            if mismatch_count == 0:
                print(f"    OK: {seq_len} positions verified")
            else:
                print(f"    FAILED: {mismatch_count} mismatches")
                all_pass = False

    # Cleanup
    try:
        manager.free_sequences(sequence_ids)
    except Exception as e:
        print(f"Warning: Failed to free sequences: {e}")

    del worker
    del manager

    # Clean up shared memory
    _shm_unlink(shm_name)

    if all_pass:
        print("\n✓ Host KV offload verification PASSED")
    else:
        print("\n✗ Host KV offload verification FAILED")

    return all_pass


def test_basic_tensor_values():
    """Basic test without any KV managers - just verify mock tensor generation."""
    print("=" * 60)
    print("Basic Tensor Value Verification")
    print("=" * 60)

    batch_size = 2
    seq_len = 5
    num_heads = 4
    head_dim = 8

    k, v = create_traceable_kv_tensors(batch_size, seq_len, num_heads, head_dim)

    print(f"\nK shape: {k.shape}")
    print(f"V shape: {v.shape}")

    # Verify encoding scheme
    # Note: bfloat16 has ~3 decimal digits precision, so allow 1% relative tolerance
    print("\nVerifying encoding:")
    all_correct = True
    for b in range(batch_size):
        for s in range(seq_len):
            expected_val = b * 1000 + s * 10 + 0 + 0 * 0.01  # h=0, d=0
            actual_val = k[b, s, 0, 0].item()
            # Use relative tolerance for large values, absolute for small
            if expected_val == 0:
                match = abs(actual_val) < 0.1
            else:
                rel_error = abs(actual_val - expected_val) / abs(expected_val)
                match = rel_error < 0.01  # 1% relative error for bfloat16
            print(f"  K[{b},{s},0,0] = {actual_val:.2f} (expected: {expected_val:.2f}) {'✓' if match else '✗'}")
            if not match:
                all_correct = False

    if all_correct:
        print("\n✓ Basic tensor encoding PASSED")
    else:
        print("\n✗ Basic tensor encoding FAILED")

    return all_correct


def main():
    parser = argparse.ArgumentParser(description="Verify KV cache prefill and offload")
    parser.add_argument("--basic", action="store_true",
                        help="Run basic tensor verification only")
    parser.add_argument("--gpu-only", action="store_true",
                        help="Test GPU KV write only (no host offload)")
    parser.add_argument("--full", action="store_true",
                        help="Run full GPU → Host offload test")
    args = parser.parse_args()

    results = {}

    if args.basic or (not args.gpu_only and not args.full):
        results["basic"] = test_basic_tensor_values()

    if args.gpu_only:
        results["gpu"] = test_gpu_kv_write_only()

    if args.full:
        results["offload"] = test_kv_offload_to_host()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name}: {status}")

    if not results:
        print("  No tests run. Use --basic, --gpu-only, or --full")


if __name__ == "__main__":
    main()
