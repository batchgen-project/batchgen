import gc

import torch
from quantization import dequant_per_token_triton


def run_tuner(batch_size=8, seq_len=13000, dim=576):
    """
    Run performance tuning for the dequantization kernel with different block sizes.

    Args:
        batch_size: Batch size for test tensors
        seq_len: Sequence length for test tensors
        dim: Dimension size for test tensors
    """
    print(
        f"Running tuner with shape: batch={batch_size}, seq_len={seq_len}, dim={dim}"
    )

    # Define block sizes to test
    block_sizes = [128, 64, 32]

    # Create test data
    q = torch.randn(
        (batch_size, seq_len, dim), device="cuda", dtype=torch.bfloat16
    ).to(torch.float8_e4m3fn)

    best_time = float("inf")
    best_block_size = None
    reference_output = None

    # Run tests for each block size
    for block_size in block_sizes:
        # Calculate number of blocks based on block size
        num_blocks = (dim + block_size - 1) // block_size

        # Create scale tensor with appropriate shape
        scale = torch.ones(
            (batch_size, seq_len, num_blocks),
            device="cuda",
            dtype=torch.float32,
        )

        # Clear GPU cache
        torch.cuda.empty_cache()
        gc.collect()

        print(f"\nTesting block_size = {block_size}")

        try:
            # Run warmup iterations
            for _ in range(5):
                _ = dequant_per_token_triton(q, scale, BLOCK_SIZE=block_size)
            torch.cuda.synchronize()

            # Create timing events
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            # Time the kernel
            iterations = 100  # More iterations for more stable timing
            start.record()
            for _ in range(iterations):
                output = dequant_per_token_triton(
                    q, scale, BLOCK_SIZE=block_size
                )
            end.record()
            torch.cuda.synchronize()

            elapsed_time = start.elapsed_time(end) / iterations

            # Save first result as reference
            if reference_output is None:
                reference_output = output.clone()
                print(
                    f"  Reference implementation (block_size={block_size}): {elapsed_time:.4f} ms"
                )
            else:
                # Check accuracy
                max_diff = (output - reference_output).abs().max().item()
                mean_diff = (output - reference_output).abs().mean().item()
                is_close = torch.allclose(
                    output, reference_output, atol=1e-2, rtol=1e-2
                )

                print(f"  Time: {elapsed_time:.4f} ms")
                print(
                    f"  Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}, Allclose: {is_close}"
                )

            # Update best configuration if this is faster
            if elapsed_time < best_time:
                best_time = elapsed_time
                best_block_size = block_size

        except Exception as e:
            print(f"  Error with block_size={block_size}: {e}")

    print("\nResults Summary:")
    if best_block_size:
        print(f"Best block size: {best_block_size}")
        print(f"Best time: {best_time:.4f} ms")
    else:
        print("No valid configuration found!")

    # Test a practical use case: dequantize and then matmul
    print("\nTesting practical use case: dequantize + matmul")

    # Create a random BF16 matrix for matmul
    A = torch.randn(
        (batch_size * seq_len, dim), device="cuda", dtype=torch.bfloat16
    )

    # Run reference with PyTorch dequantize (if available)
    try:
        # This is a placeholder - PyTorch doesn't have a built-in FP8 dequantize, so we use our implementation
        print("Running reference with block_size=128...")
        num_blocks = (dim + 128 - 1) // 128
        scale_ref = torch.ones(
            (batch_size, seq_len, num_blocks),
            device="cuda",
            dtype=torch.float32,
        )

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        dequant_ref = dequant_per_token_triton(q, scale_ref, BLOCK_SIZE=128)
        ref_result = torch.matmul(
            A, dequant_ref.view(batch_size * seq_len, dim).T
        )
        end.record()
        torch.cuda.synchronize()

        ref_time = start.elapsed_time(end)
        print(f"Reference time (dequant + matmul): {ref_time:.4f} ms")

        # Test each block size in the practical scenario
        for block_size in block_sizes:
            num_blocks = (dim + block_size - 1) // block_size
            scale_test = torch.ones(
                (batch_size, seq_len, num_blocks),
                device="cuda",
                dtype=torch.float32,
            )

            # Clear cache
            torch.cuda.empty_cache()
            gc.collect()

            # Warmup
            for _ in range(5):
                dequant_out = dequant_per_token_triton(
                    q, scale_test, BLOCK_SIZE=block_size
                )
                _ = torch.matmul(
                    A, dequant_out.view(batch_size * seq_len, dim).T
                )
            torch.cuda.synchronize()

            # Timing
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            dequant_out = dequant_per_token_triton(
                q, scale_test, BLOCK_SIZE=block_size
            )
            test_result = torch.matmul(
                A, dequant_out.view(batch_size * seq_len, dim).T
            )
            end.record()
            torch.cuda.synchronize()

            test_time = start.elapsed_time(end)

            # Check accuracy
            max_diff = (test_result - ref_result).abs().max().item()
            mean_diff = (test_result - ref_result).abs().mean().item()
            is_close = torch.allclose(
                test_result, ref_result, atol=1e-2, rtol=1e-2
            )

            print(f"Block size {block_size}:")
            print(f"  Time: {test_time:.4f} ms")
            print(f"  Speedup vs reference: {ref_time / test_time:.2f}x")
            print(
                f"  Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}, Allclose: {is_close}"
            )

    except Exception as e:
        print(f"Error in practical test: {e}")


if __name__ == "__main__":
    # Run the tuner with different model sizes

    # Small model size
    run_tuner()

    # # Medium model size
    # run_tuner()

    # # Large model size (typical transformer)
    # run_tuner()

    # # Extra large (for very large models)
    # run_tuner()
