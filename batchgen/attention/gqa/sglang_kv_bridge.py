"""GQA KV bridge — read/write adapter for the SGLang decode runner.

Sibling of ``attention/dsa/sglang_kv_bridge.py`` (the MLA/NSA adapter). Where MLA
stores only a latent K (``[N,1,576]``) plus a separate FP8 indexer cache, a GQA
model (e.g. gpt-oss-120b: 8 KV heads, head_dim 64) stores BOTH K and V in full
``[N, num_kv_heads, head_dim]`` form and has no indexer. This adapter duck-types
the subset of SGLang's ``MHATokenToKVPool`` interface that the gpt-oss decode
attention calls (memory_pool.py:929-988), backed by BatchGen's paged KV manager,
so SGLang reads/writes BatchGen-owned KV with no extra copy.

Layout correspondence (zero-copy reshape):
  BatchGen ``_k_cache[layer]`` / ``_v_cache[layer]`` are paged
  ``[num_pages, page_size, num_kv_heads, head_dim]`` BF16. SGLang's MHA pool
  indexes by global token slot = page * page_size + offset, i.e.
  ``[num_pages * page_size, num_kv_heads, head_dim]``. Because the BatchGen tensor
  is contiguous with (page, offset) as the two leading axes, the reshape is exact
  and moves no data — identical to the MLA adapter's primary-K view, just with
  ``num_kv_heads`` > 1 and a parallel V buffer.

Sliding window + attention sinks are NOT this adapter's concern:
  * SINKS — gpt-oss's per-head sink biases (``self_attn.sinks``) are model weights
    fed to the attention kernel via kwargs (gpt_oss.py:350 -> backend), not through
    the KV pool. The weight-feeder supplies them; the adapter never sees them.
  * SLIDING WINDOW — gpt-oss alternates sliding (128) / full layers. SGLang applies
    the window in the attention backend (``layer.sliding_window_size``); the pool
    just stores tokens. This adapter therefore exposes the FULL per-layer buffer for
    every layer and lets the backend mask. Correctness of the sliding-layer read
    hinges on SGLang indexing the buffer with the ForwardBatch page table (the same
    slots BatchGen wrote) rather than a window-compacted SWA layout — VALIDATE this
    on the first coherent decode (it is the #1 gpt-oss bridge risk).

Scope: single (non-dual) KV manager; BF16 store. No FP8 KV-cache scale path yet
(H20 GQA KV is BF16); ``k_scale``/``v_scale`` are honored defensively if passed.
"""

from __future__ import annotations

import torch


class BatchGenGQAKVAdapter:
    """Duck-types the SGLang ``MHATokenToKVPool`` subset gpt-oss decode reads/writes.

    Args:
        gpu_paged_kv_manager: BatchGen paged KV manager holding both K and V.
            ``get_layer_kv_buffer(layer)`` -> K ``[num_pages, page_size, H, D]``;
            ``get_layer_v_buffer(layer)`` -> V (same shape).
        page_size: tokens per page (must equal the value baked into the ForwardBatch
            page table the bridge builds).
        layer_num: layer count (bookkeeping / validation only).
        store_dtype: on-GPU storage dtype of the KV cache (BF16 for H20 GQA).
    """

    def __init__(
        self,
        gpu_paged_kv_manager,
        page_size: int = 64,
        layer_num: int | None = None,
        store_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.gpu_paged_kv_manager = gpu_paged_kv_manager
        self.page_size = int(page_size)
        self.layer_num = layer_num
        # SGLang reads these off the pool object directly in a few places.
        self.store_dtype = store_dtype
        self.dtype = store_dtype

    # ------------------------------------------------------------------ #
    # Read path — flattened per-token views over BatchGen's paged caches.
    # ------------------------------------------------------------------ #
    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        """K buffer as SGLang sees it: ``[num_pages * page_size, H, D]`` BF16 view."""
        k = self.gpu_paged_kv_manager.get_layer_kv_buffer(layer_id)
        # k: [num_pages, page_size, num_kv_heads, head_dim] BF16, contiguous.
        assert k.is_contiguous(), (
            "get_key_buffer requires a contiguous K cache for zero-copy reshape; "
            f"got strides {k.stride()} for shape {tuple(k.shape)}"
        )
        num_pages, page_size, num_heads, head_dim = k.shape
        assert page_size == self.page_size, (
            f"page_size mismatch: adapter={self.page_size}, cache={page_size}"
        )
        return k.reshape(num_pages * page_size, num_heads, head_dim)

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        """V buffer as SGLang sees it: ``[num_pages * page_size, H, D]`` BF16 view."""
        v = self.gpu_paged_kv_manager.get_layer_v_buffer(layer_id)
        assert v.is_contiguous(), (
            "get_value_buffer requires a contiguous V cache for zero-copy reshape; "
            f"got strides {v.stride()} for shape {tuple(v.shape)}"
        )
        num_pages, page_size, num_heads, head_dim = v.shape
        assert page_size == self.page_size
        return v.reshape(num_pages * page_size, num_heads, head_dim)

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    @property
    def device(self) -> torch.device:
        return self.gpu_paged_kv_manager.get_layer_kv_buffer(0).device

    # ------------------------------------------------------------------ #
    # Write path — SGLang drives the decode forward and writes each token's
    # K/V. These persist into BatchGen's paged caches through the same
    # zero-copy views the read path exposes (single writer per phase:
    # BatchGen prefill, then SGLang decode).
    # ------------------------------------------------------------------ #
    def set_kv_buffer(
        self,
        layer,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale=None,
        v_scale=None,
        layer_id_override=None,
        *args,
        **kwargs,
    ) -> None:
        """Write decode-token K and V into BatchGen's BF16 caches at ``loc``.

        Mirrors ``MHATokenToKVPool.set_kv_buffer`` (memory_pool.py:951): ``loc`` are
        global token slots (page * page_size + offset). Writing through the
        flattened ``[N, H, D]`` views persists into BatchGen KV for the next step.
        """
        layer_id = layer_id_override if layer_id_override is not None else layer.layer_id
        kbuf = self.get_key_buffer(layer_id)
        n = loc.numel()
        if k_scale is not None and cache_k.dtype != kbuf.dtype:
            cache_k = cache_k.div(k_scale)
        kbuf[loc] = cache_k.reshape(n, *kbuf.shape[1:]).to(kbuf.dtype)
        if cache_v is not None:
            vbuf = self.get_value_buffer(layer_id)
            if v_scale is not None and cache_v.dtype != vbuf.dtype:
                cache_v = cache_v.div(v_scale)
            vbuf[loc] = cache_v.reshape(n, *vbuf.shape[1:]).to(vbuf.dtype)
