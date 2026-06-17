"""M1 KV bridge — Slice 0 read-adapter.

BatchGen prefill writes BF16 KV into its own paged caches. This adapter
duck-types the *minimal* subset of SGLang's ``token_to_kv_pool`` interface that
SGLang's NSA (DSA) decode path calls, so SGLang can read BatchGen-produced KV
without a copy of the whole pipeline.

Two SGLang entry points are reproduced here:

  * ``get_key_buffer(layer_id)`` — the primary (MLA) compressed-KV buffer.
    SGLang's ``MLATokenToKVPool`` stores this as a dense per-token tensor
    ``(num_total_tokens + page_size, 1, kv_lora_rank + qk_rope_head_dim)`` BF16
    (references/sglang/.../mem_cache/memory_pool.py:1436-1442, get_key_buffer
    at :1465-1472). BatchGen stores the same data paged as
    ``[num_pages, page_size, 1, 576]`` BF16. The two are byte-identical when
    flattened over (page, token): SGLang token index == page * page_size +
    offset. We therefore expose a zero-copy reshape view
    ``[num_pages * page_size, 1, 576]``.

  * ``get_index_k_with_scale_buffer(layer_id)`` — the NSA indexer FP8 KV buffer.
    SGLang's ``NSATokenToKVPool`` stores this paged as
    ``(num_pages, page_size * (index_head_dim + index_head_dim//quant_block_size*4))``
    uint8 (memory_pool.py:1788-1807). For GLM-5/DSP config
    (index_head_dim=128, quant_block_size=128, page_size=64) that is
    ``(num_pages, 64*(128+4)) = (num_pages, 8448)`` uint8. Per page:
        bytes [0 : 64*128]            -> FP8 e4m3 K, view(float8_e4m3fn) as
                                         [64, 128] (64 tokens x 128 elems)
        bytes [64*128 : 64*128+64*4]  -> fp32 scales, view(float32) as [64]
                                         (one scale per token)
    Byte offsets verified against the SGLang write kernel
    ``_set_k_and_s_triton_kernel`` (index_buf_accessor.py:401-441):
        K   : page*BUF_NUMEL_PER_PAGE + tok*128 + arange(128)   (fp8 view)
        scale: page*(BUF_NUMEL_PER_PAGE//4) + (64*128)//4 + tok (fp32 view)
    BatchGen stores indexer K as BF16 paged ``[num_pages, 64, 1, 128]``. This
    adapter quantizes that BF16 to FP8 e4m3 + per-token fp32 scale and packs it
    into the SGLang uint8 layout byte-for-byte.

Scope (Slice 0): READ adapter, materialized eagerly. No incremental writes, no
CUDA-graph capture, no fnuz/HIP path. Single-block indexer (128 == quant_block).

Layout source of record:
  references/sglang/python/sglang/srt/mem_cache/memory_pool.py:1727-1857
  references/sglang/python/sglang/srt/layers/attention/nsa/index_buf_accessor.py:328-441
"""

from __future__ import annotations

import torch

# SGLang NSA constants for the GLM-5 / DeepSeek-V3.2 indexer config.
# (memory_pool.py:1728 quant_block_size, :1777 index_head_dim assertion,
#  :1782 page_size for non-HIP.)
_INDEX_HEAD_DIM = 128
_QUANT_BLOCK_SIZE = 128
_SGLANG_PAGE_SIZE = 64
_SCALE_NBYTES = 4  # fp32 scale, one per (index_head_dim // quant_block_size) block

# FP8 dtype SGLang uses on non-HIP (memory_pool.py / index_buf_accessor.py:373).
# TODO(verify-on-gpu): on HIP/MI3xx SGLang switches to float8_e4m3fnuz and
# page_size==1 (index_buf_accessor.py:356-374). Slice 0 targets H20/H100 only.
_FP8_DTYPE = torch.float8_e4m3fn


class BatchGenNSAKVAdapter:
    """Duck-types the SGLang ``token_to_kv_pool`` subset NSA decode reads.

    Holds references to BatchGen's two paged KV managers and re-presents their
    BF16 contents in SGLang's expected buffer layouts.

    Args:
        gpu_paged_kv_manager: BatchGen primary (MLA) paged KV manager. Its
            ``get_layer_kv_with_page_table(layer)`` returns a K tensor shaped
            ``[num_pages, page_size, 1, 576]`` BF16.
        gpu_paged_kv_manager_aux: BatchGen auxiliary (indexer) paged KV manager.
            Its K tensor is ``[num_pages, page_size, 1, 128]`` BF16.
        page_size: tokens per page (64 for GLM-5; must equal SGLang's
            NSATokenToKVPool.page_size).
        layer_num: number of layers (for bookkeeping / validation only).
    """

    def __init__(
        self,
        gpu_paged_kv_manager,
        gpu_paged_kv_manager_aux,
        page_size: int = _SGLANG_PAGE_SIZE,
        layer_num: int | None = None,
    ) -> None:
        self.gpu_paged_kv_manager = gpu_paged_kv_manager
        self.gpu_paged_kv_manager_aux = gpu_paged_kv_manager_aux
        self.page_size = int(page_size)
        self.layer_num = layer_num

        # SGLang-facing dtypes / constants (duck-typed attributes the NSA pool
        # exposes; some callers read them off the pool object directly).
        self.index_head_dim = _INDEX_HEAD_DIM
        self.quant_block_size = _QUANT_BLOCK_SIZE
        self.store_dtype = torch.bfloat16
        self.index_k_with_scale_buffer_dtype = torch.uint8
        self.index_k_fp8_dtype = _FP8_DTYPE

        # SGLang requires page_size == 64 on non-HIP (memory_pool.py:1782 and
        # the SetKAndS assertion at index_buf_accessor.py:366).
        assert self.page_size == _SGLANG_PAGE_SIZE, (
            f"Slice 0 supports SGLang non-HIP page_size==64 only, got {self.page_size}"
        )

    # ------------------------------------------------------------------ #
    # Primary (MLA) compressed-KV buffer.
    # ------------------------------------------------------------------ #
    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        """Return the primary KV buffer as SGLang sees it: a per-token view.

        BatchGen primary K is ``[num_pages, page_size, 1, 576]`` BF16. SGLang's
        MLA pool indexes by global token slot = page * page_size + offset, i.e.
        shape ``[num_pages * page_size, 1, 576]``. Because the BatchGen tensor
        is contiguous and the (page, offset) axes are the two leading axes, this
        is an exact zero-copy reshape — no data is moved.

        Returns:
            ``[num_pages * page_size, 1, 576]`` BF16 view.
        """
        k_cache, _, _ = self.gpu_paged_kv_manager.get_layer_kv_with_page_table(layer_id)
        # k_cache: [num_pages, page_size, num_k_heads(=1), k_head_dim(=576)] BF16.
        assert k_cache.is_contiguous(), (
            "get_key_buffer requires a contiguous primary K cache for a zero-copy "
            f"reshape; got strides {k_cache.stride()} for shape {tuple(k_cache.shape)}"
        )
        num_pages, page_size, num_heads, head_dim = k_cache.shape
        assert page_size == self.page_size
        # TODO(verify-on-gpu): SGLang's MLA buffer is allocated with a +page_size
        # tail slot (memory_pool.py:1436-1442 uses size + page_size). BatchGen's
        # paged cache has no such tail. Decode reads index into valid tokens
        # only, so the missing tail should be unobservable, but confirm SGLang's
        # NSA decode never dereferences the [size : size+page_size] guard slot.
        return k_cache.reshape(num_pages * page_size, num_heads, head_dim)

    # ------------------------------------------------------------------ #
    # NSA indexer FP8 KV buffer.
    # ------------------------------------------------------------------ #
    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        """Pack BatchGen's BF16 indexer K into SGLang's FP8 paged uint8 buffer.

        BatchGen indexer K: ``[num_pages, page_size, 1, 128]`` BF16. SGLang
        expects ``[num_pages, page_size*(128 + 4)] = [num_pages, 8448]`` uint8
        with the per-page layout documented in this module's docstring.

        FP8 pack (per token, single 128-elem block == 1 quant block):
          1. quantize BF16 -> FP8 e4m3 + fp32 scale via BatchGen's reusable
             ``per_token_blocked_quantize_bf16_to_fp8_1d`` (block_size=128).
             That kernel uses scale = amax(|x| over 128) * (1/448), exactly the
             convention SGLang's ``act_quant`` uses (fp8_quantize.py:315-318).
          2. write the 128 FP8 bytes per token into the page's K section,
          3. write the 4-byte fp32 scale per token into the page's scale section.

        Returns:
            ``[num_pages, 8448]`` uint8, byte-compatible with SGLang's
            ``index_k_with_scale_buffer[layer]``.
        """
        from batchgen_kernels.triton.fp8_quantize import (
            per_token_blocked_quantize_bf16_to_fp8_1d,
        )

        k_cache, _, _ = self.gpu_paged_kv_manager_aux.get_layer_kv_with_page_table(
            layer_id
        )
        # k_cache: [num_pages, page_size, 1, 128] BF16.
        assert k_cache.is_contiguous()
        num_pages, page_size, num_heads, head_dim = k_cache.shape
        assert page_size == self.page_size
        assert num_heads == 1, f"indexer is MQA (1 head), got {num_heads}"
        assert head_dim == self.index_head_dim, (
            f"indexer head_dim must be {self.index_head_dim}, got {head_dim}"
        )

        device = k_cache.device
        # Flatten to [num_pages * page_size, 128] for per-token quantization.
        k_bf16 = k_cache.reshape(num_pages * page_size, head_dim).contiguous()
        # FP8 e4m3 [M, 128] + fp32 scale [M, 1] (num_blocks == 128//128 == 1).
        k_fp8, k_scale = per_token_blocked_quantize_bf16_to_fp8_1d(
            k_bf16, block_size=self.quant_block_size
        )
        assert k_fp8.dtype == _FP8_DTYPE
        assert k_scale.shape == (num_pages * page_size, 1), (
            f"expected single fp32 scale per token, got {tuple(k_scale.shape)}"
        )

        # Per-page byte widths (non-HIP page_size=64): 64*128 K bytes + 64*4 scale.
        k_bytes_per_page = page_size * head_dim  # 8192
        s_bytes_per_page = page_size * _SCALE_NBYTES  # 256
        buf_numel_per_page = k_bytes_per_page + s_bytes_per_page  # 8448

        buf = torch.empty(
            (num_pages, buf_numel_per_page), dtype=torch.uint8, device=device
        )

        # K section: bytes [0 : k_bytes_per_page]. View as fp8 -> [num_pages,
        # page_size, 128]; k_fp8 reshaped to the same shape copies byte-for-byte
        # (1 fp8 elem == 1 byte). Matches _set_k_and_s_triton_kernel K store:
        # page*BUF_NUMEL_PER_PAGE + tok*128 + arange(128) (index_buf_accessor.py:427-431).
        k_section = buf[:, :k_bytes_per_page].view(_FP8_DTYPE).reshape(
            num_pages, page_size, head_dim
        )
        k_section.copy_(k_fp8.view(num_pages, page_size, head_dim))

        # Scale section: bytes [k_bytes_per_page : buf_numel_per_page]. View as
        # fp32 -> [num_pages, page_size]; one fp32 scale per token. Matches the
        # kernel scale store offset (index_buf_accessor.py:434-438):
        # page*(BUF_NUMEL_PER_PAGE//4) + (page_size*128)//4 + tok.
        s_section = buf[:, k_bytes_per_page:].view(torch.float32)
        s_section.copy_(k_scale.view(num_pages, page_size))

        return buf

    @property
    def device(self) -> torch.device:
        """Device of the indexer KV cache (read by SGLang's index_buf_accessor)."""
        k_cache, _, _ = self.gpu_paged_kv_manager_aux.get_layer_kv_with_page_table(0)
        return k_cache.device

    # ------------------------------------------------------------------ #
    # WRITE PATH. In approach (A) SGLang drives the decode forward, so it
    # computes AND writes each decode token's KV (BatchGen's native decode
    # write path is replaced — single writer per phase: BatchGen prefill, then
    # SGLang decode). These mirror the NSA pool's set_* methods, writing through
    # the same zero-copy views the read path exposes so writes persist into
    # BatchGen's paged caches for the next decode step.
    # ------------------------------------------------------------------ #
    def set_kv_buffer(self, layer, loc, cache_k, cache_v=None, *args, **kwargs):
        """Write decode-token primary MLA K into BatchGen's BF16 cache at ``loc``.

        SGLang's ``MLATokenToKVPool.set_kv_buffer`` does ``kv_buffer[loc] = cache_k``.
        ``get_key_buffer`` is a zero-copy ``[N,1,576]`` view of BatchGen's primary
        cache, so writing through it persists into BatchGen KV. MLA stores only K;
        ``cache_v`` is ignored.
        """
        kbuf = self.get_key_buffer(layer.layer_id)  # [N, 1, 576] BF16 view.
        kbuf[loc] = cache_k.reshape(loc.numel(), *kbuf.shape[1:]).to(kbuf.dtype)

    def set_mla_kv_buffer(self, layer, loc, cache_k_nope, cache_k_rope, *args, **kwargs):
        """Write MLA K given separate (nope[512], rope[64]) -> 576 at ``loc``."""
        kbuf = self.get_key_buffer(layer.layer_id)
        cache_k = torch.cat([cache_k_nope, cache_k_rope], dim=-1)
        kbuf[loc] = cache_k.reshape(loc.numel(), *kbuf.shape[1:]).to(kbuf.dtype)

    def set_index_k_scale_buffer(self, layer_id, loc, index_k, index_k_scale):
        """Write decode-token indexer K into BatchGen's BF16 aux cache at ``loc``.

        SGLang computes the indexer K as FP8 e4m3 + per-token fp32 scale
        (nsa_indexer.py:1047, act_quant). BatchGen stores indexer K as BF16, so
        dequantize (fp8 * scale) -> BF16 and write at the token locations. The
        read path re-quantizes BF16 -> FP8, so decode tokens round-trip through
        e4m3 (the lightning indexer is approximate; topk tolerates the step).

        TODO(real-run perf): this round-trip + the read-path re-quant per step is
        the dominant decode cost (design (4)). The eventual design materializes an
        adapter-owned FP8 buffer once at decode start and has SGLang read/write it
        directly; M1 prioritizes correctness.
        """
        aux_k, _, _ = self.gpu_paged_kv_manager_aux.get_layer_kv_with_page_table(
            layer_id
        )
        num_pages, page_size, _num_heads, head_dim = aux_k.shape
        aux_flat = aux_k.reshape(num_pages * page_size, head_dim)
        index_k_fp8 = (
            index_k if index_k.dtype == _FP8_DTYPE else index_k.view(_FP8_DTYPE)
        )
        deq = (
            index_k_fp8.to(torch.float32)
            * index_k_scale.to(torch.float32).reshape(-1, 1)
        ).to(aux_flat.dtype)
        aux_flat[loc] = deq.reshape(loc.numel(), head_dim)

    def get_index_k_scale_buffer(self, layer_id: int, seq_len: int, page_indices):
        """Per-sequence DECODE read: gather (k_fp8, k_scale) for the first
        ``seq_len`` tokens of one sequence from its ``page_indices`` (block table).

        SGLang's NSA decode calls THIS per sequence (nsa_indexer.py:521), not the
        full-buffer ``get_index_k_with_scale_buffer``. Because our packed buffer is
        byte-identical to SGLang's layout (Slice 0, check iv), we reuse SGLang's own
        fused Triton gather ``GetKAndS`` on it — guaranteeing the exact decode read
        semantics.

        Returns:
            (k_fp8 ``(seq_len, index_head_dim)`` uint8, k_scale ``(seq_len, 4)`` uint8).

        TODO(perf): ``get_index_k_with_scale_buffer`` re-quantizes the whole layer
        cache on each call; memoize per (layer, decode-step) — fine for M1 correctness
        but this is the dominant per-step cost (design (4)).
        """
        from sglang.srt.layers.attention.nsa import index_buf_accessor

        buf = self.get_index_k_with_scale_buffer(layer_id)
        return index_buf_accessor.GetKAndS.execute(
            self, buf, seq_len=seq_len, page_indices=page_indices
        )
