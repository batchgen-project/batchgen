"""Attention kernels: paged decode, RoPE, RMSNorm, QKV split."""

try:
    from batchgen_kernels.attention.decode import attention_decode_bf16
    __all__ = ["attention_decode_bf16"]
except (ImportError, ModuleNotFoundError):
    # Decode .so is pre-compiled and may not be present on all machines
    __all__ = []
