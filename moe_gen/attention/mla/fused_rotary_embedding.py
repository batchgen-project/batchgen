import math

import torch
import triton
import triton.language as tl


@triton.jit
def rope_kernel(
    # Input/Output tensors
    x_ptr,  # Input tensor [batch * heads * seq_len, head_dim]
    cos_ptr,  # Cosine values [max_seq_len, head_dim]
    sin_ptr,  # Sine values [max_seq_len, head_dim]
    out_ptr,  # Output tensor [batch * heads * seq_len, head_dim]
    position_ids_ptr,  # Position IDs [seq_len]
    # Sizes
    num_vectors: tl.constexpr,  # Number of vectors (batch * heads * seq_len)
    head_dim: tl.constexpr,  # Head dimension (must be even)
    half_dim: tl.constexpr,  # Half of the head dimension (head_dim // 2)
    seq_len: tl.constexpr,  # Sequence length
    # Strides
    stride_x_batch: tl.constexpr,  # Stride for batch dimension in x
    stride_x_head: tl.constexpr,  # Stride for heads dimension in x
    cos_stride_seq: tl.constexpr,  # Stride for sequence dimension in cos
    cos_stride_dim: tl.constexpr,  # Stride for head dimension in cos
    sin_stride_seq: tl.constexpr,  # Stride for sequence dimension in sin
    sin_stride_dim: tl.constexpr,  # Stride for head dimension in sin
    stride_out_batch: tl.constexpr,  # Stride for batch dimension in out
    stride_out_head: tl.constexpr,  # Stride for heads dimension in out
    position_ids_stride: tl.constexpr,  # Stride for sequence dimension in position_ids
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,  # Block size for batch dimension
):
    """
    Kernel to apply Rotary Position Embedding (RoPE) to the input tensor.
    Args:
            x_ptr (tl.pointer): Pointer to the input tensor of shape [batch * heads * seq_len, head_dim].
            cos_ptr (tl.pointer): Pointer to the cosine values of shape [max_seq_len, head_dim].
            sin_ptr (tl.pointer): Pointer to the sine values of shape [max_seq_len, head_dim].
            out_ptr (tl.pointer): Pointer to the output tensor of shape [batch * heads * seq_len, head_dim].
            position_ids_ptr (tl.pointer): Pointer to the position IDs of shape [seq_len].
    """
    pid = tl.program_id(axis=0)
    # Calculate the start index for this program
    start_idx = pid * BLOCK_SIZE_M
    # Calculate the end index for this program
    end_idx = min(start_idx + BLOCK_SIZE_M, num_vectors)
    # Create offsets for the input and output tensors
    for i in range(start_idx, end_idx):
        # Compute the sequence index for the current index i
        seq_idx = (i // head_dim) % seq_len
        # Load the position ID for the current index
        position_id = tl.load(position_ids_ptr + seq_idx * position_ids_stride)
        # Calculate the offsets for cos and sin based on the position ID
        first_half_cos_offset = (
            position_id * cos_stride_seq
            + tl.arange(0, half_dim) * cos_stride_dim
        )
        second_half_cos_offset = position_id * cos_stride_seq + (
            tl.arange(half_dim, head_dim) * cos_stride_dim
        )
        # Load the cosine and sine values
        first_half_cos_values = tl.load(cos_ptr + first_half_cos_offset)
        second_half_cos_values = tl.load(cos_ptr + second_half_cos_offset)

        first_half_sin_offset = (
            position_id * sin_stride_seq
            + tl.arange(0, half_dim) * sin_stride_dim
        )
        second_half_sin_offset = position_id * sin_stride_seq + (
            tl.arange(half_dim, head_dim) * sin_stride_dim
        )
        first_half_sin_values = tl.load(sin_ptr + first_half_sin_offset)
        second_half_sin_values = tl.load(sin_ptr + second_half_sin_offset)

        # Calculate the offsets for the vector in x
        offsets_x_first_half = (
            i * stride_x_batch + tl.arange(0, half_dim) * stride_x_head
        )
        offsets_x_second_half = i * stride_x_batch + (
            tl.arange(half_dim, head_dim) * stride_x_head
        )
        # Load the vector from x
        first_half_x = tl.load(x_ptr + offsets_x_first_half)
        second_half_x = tl.load(x_ptr + offsets_x_second_half)

        # Apply RoPE
        first_half_intermediate = first_half_x * first_half_cos_values
        second_half_intermediate = second_half_x * second_half_cos_values
        second_half_intermediate += first_half_x * second_half_sin_values
        first_half_intermediate -= second_half_x * first_half_sin_values
        # Store the result in the output tensor
        first_half_out_offset = (
            i * stride_out_batch + tl.arange(0, half_dim) * stride_out_head
        )
        second_half_out_offset = i * stride_out_batch + (
            tl.arange(half_dim, head_dim) * stride_out_head
        )
        tl.store(out_ptr + first_half_out_offset, first_half_intermediate)
        tl.store(out_ptr + second_half_out_offset, second_half_intermediate)


def fused_rotary_embedding(
    x: torch.Tensor,  # Input tensor of shape [batch, heads, seq_len, head_dim]
    cos: torch.Tensor,  # Cosine values of shape [max_seq_len, head_dim]
    sin: torch.Tensor,  # Sine values of shape [max_seq_len, head_dim]
    position_ids: torch.Tensor,  # Position IDs of shape [seq_len]
):
    B, H, S, D = x.shape
    x = x.contiguous()  # Ensure x is contiguous in memory
    cos = cos.contiguous()  # Ensure cos is contiguous in memory
    sin = sin.contiguous()  # Ensure sin is contiguous in memory
    position_ids = (
        position_ids.contiguous()
    )  # Ensure position_ids is contiguous in memory
    assert D % 2 == 0, "Head dimension must be even for rotary embedding."
    x = x.view(B * H * S, D)  # Flatten to [batch * heads * seq_len, head_dim]
    result = torch.empty_like(x)

    grid = lambda meta: (math.ceil(B * H * S / meta["BLOCK_SIZE_M"]),)
    rope_kernel[grid](
        x,
        cos,
        sin,
        result,
        position_ids,
        num_vectors=B * H * S,
        head_dim=D,
        half_dim=D // 2,
        seq_len=S,
        stride_x_batch=x.stride(0),
        stride_x_head=x.stride(1),
        cos_stride_seq=cos.stride(0),
        cos_stride_dim=cos.stride(1),
        sin_stride_seq=sin.stride(0),
        sin_stride_dim=sin.stride(1),
        stride_out_batch=result.stride(0),
        stride_out_head=result.stride(1),
        position_ids_stride=position_ids.stride(0),
        BLOCK_SIZE_M=64,  # Adjust block size as needed
    )
    return result.view(
        B, H, S, D
    )  # Reshape back to original shape [batch, heads, seq_len, head_dim]


@triton.jit
def rope_inplace_kernel(
    # Input/Output tensors
    x_ptr,  # Input tensor [batch * heads * seq_len, head_dim]
    cos_ptr,  # Cosine values [max_seq_len, head_dim]
    sin_ptr,  # Sine values [max_seq_len, head_dim]
    out_ptr,  # Output tensor [batch * heads * seq_len, head_dim]
    position_ids_ptr,  # Position IDs [seq_len]
    # Sizes
    num_vectors: tl.constexpr,  # Number of vectors (batch * heads * seq_len)
    head_dim: tl.constexpr,  # Head dimension (must be even)
    half_dim: tl.constexpr,  # Half of the head dimension (head_dim // 2)
    seq_len: tl.constexpr,  # Sequence length
    # Strides
    stride_x_batch: tl.constexpr,  # Stride for batch dimension in x
    stride_x_head: tl.constexpr,  # Stride for heads dimension in x
    cos_stride_seq: tl.constexpr,  # Stride for sequence dimension in cos
    cos_stride_dim: tl.constexpr,  # Stride for head dimension in cos
    sin_stride_seq: tl.constexpr,  # Stride for sequence dimension in sin
    sin_stride_dim: tl.constexpr,  # Stride for head dimension in sin
    stride_out_batch: tl.constexpr,  # Stride for batch dimension in out
    stride_out_head: tl.constexpr,  # Stride for heads dimension in out
    position_ids_stride: tl.constexpr,  # Stride for sequence dimension in position_ids
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,  # Block size for batch dimension
    result_offset: tl.constexpr = 0,  # Offset for result tensor (default is 0, can be adjusted if needed)
):
    """
    Kernel to apply Rotary Position Embedding (RoPE) to the input tensor.
    Args:
            x_ptr (tl.pointer): Pointer to the input tensor of shape [batch * heads * seq_len, head_dim].
            cos_ptr (tl.pointer): Pointer to the cosine values of shape [max_seq_len, head_dim].
            sin_ptr (tl.pointer): Pointer to the sine values of shape [max_seq_len, head_dim].
            out_ptr (tl.pointer): Pointer to the output tensor of shape [batch * heads * seq_len, head_dim].
            position_ids_ptr (tl.pointer): Pointer to the position IDs of shape [seq_len].
    """
    pid = tl.program_id(axis=0)
    # Calculate the start index for this program
    start_idx = pid * BLOCK_SIZE_M
    # Calculate the end index for this program
    end_idx = min(start_idx + BLOCK_SIZE_M, num_vectors)
    # Create offsets for the input and output tensors
    for i in range(start_idx, end_idx):
        # Compute the sequence index for the current index i
        seq_idx = (i // head_dim) % seq_len
        # Load the position ID for the current index
        position_id = tl.load(position_ids_ptr + seq_idx * position_ids_stride)
        # Calculate the offsets for cos and sin based on the position ID
        first_half_cos_offset = (
            position_id * cos_stride_seq
            + tl.arange(0, half_dim) * cos_stride_dim
        )
        second_half_cos_offset = position_id * cos_stride_seq + (
            tl.arange(half_dim, head_dim) * cos_stride_dim
        )
        # Load the cosine and sine values
        first_half_cos_values = tl.load(cos_ptr + first_half_cos_offset)
        second_half_cos_values = tl.load(cos_ptr + second_half_cos_offset)

        first_half_sin_offset = (
            position_id * sin_stride_seq
            + tl.arange(0, half_dim) * sin_stride_dim
        )
        second_half_sin_offset = position_id * sin_stride_seq + (
            tl.arange(half_dim, head_dim) * sin_stride_dim
        )
        first_half_sin_values = tl.load(sin_ptr + first_half_sin_offset)
        second_half_sin_values = tl.load(sin_ptr + second_half_sin_offset)

        # Calculate the offsets for the vector in x
        offsets_x_first_half = (
            i * stride_x_batch + tl.arange(0, half_dim) * stride_x_head
        )
        offsets_x_second_half = i * stride_x_batch + (
            tl.arange(half_dim, head_dim) * stride_x_head
        )
        # Load the vector from x
        first_half_x = tl.load(x_ptr + offsets_x_first_half)
        second_half_x = tl.load(x_ptr + offsets_x_second_half)

        # Apply RoPE
        first_half_intermediate = first_half_x * first_half_cos_values
        second_half_intermediate = second_half_x * second_half_cos_values
        second_half_intermediate += first_half_x * second_half_sin_values
        first_half_intermediate -= second_half_x * first_half_sin_values
        # Store the result in the output tensor
        first_half_out_offset = (
            i * stride_out_batch
            + (result_offset + tl.arange(0, half_dim)) * stride_out_head
        )
        second_half_out_offset = i * stride_out_batch + (
            result_offset + (tl.arange(half_dim, head_dim)) * stride_out_head
        )
        tl.store(out_ptr + first_half_out_offset, first_half_intermediate)
        tl.store(out_ptr + second_half_out_offset, second_half_intermediate)


def fused_rotary_embedding_inplace(
    x: torch.Tensor,  # Input tensor of shape [batch, heads, seq_len, head_dim]
    cos: torch.Tensor,  # Cosine values of shape [max_seq_len, head_dim]
    sin: torch.Tensor,  # Sine values of shape [max_seq_len, head_dim]
    position_ids: torch.Tensor,  # Position IDs of shape [seq_len]
    result: torch.Tensor,  # Optional output tensor to store results
    result_offset: int,
):
    B, H, S, D = x.shape
    x = x.contiguous()  # Ensure x is contiguous in memory
    cos = cos.contiguous()  # Ensure cos is contiguous in memory
    sin = sin.contiguous()  # Ensure sin is contiguous in memory
    position_ids = (
        position_ids.contiguous()
    )  # Ensure position_ids is contiguous in memory
    assert D % 2 == 0, "Head dimension must be even for rotary embedding."
    x = x.view(B * H * S, D)  # Flatten to [batch * heads * seq_len, head_dim]
    # result = torch.empty_like(x)
    RB, RS, RD = result.shape
    result = result.view(RB * RS, RD)  # Flatten result to match x shape

    grid = lambda meta: (math.ceil(B * H * S / meta["BLOCK_SIZE_M"]),)
    rope_inplace_kernel[grid](
        x,
        cos,
        sin,
        result,
        position_ids,
        num_vectors=B * H * S,
        head_dim=D,
        half_dim=D // 2,
        seq_len=S,
        stride_x_batch=x.stride(0),
        stride_x_head=x.stride(1),
        cos_stride_seq=cos.stride(0),
        cos_stride_dim=cos.stride(1),
        sin_stride_seq=sin.stride(0),
        sin_stride_dim=sin.stride(1),
        stride_out_batch=result.stride(0),
        stride_out_head=result.stride(1),
        position_ids_stride=position_ids.stride(0),
        BLOCK_SIZE_M=16,  # Adjust block size as needed
        result_offset=result_offset,  # Offset for result tensor
    )
    return result.view(
        RB, RS, RD
    )  # Reshape back to original shape [batch, heads, seq_len, head_dim]
