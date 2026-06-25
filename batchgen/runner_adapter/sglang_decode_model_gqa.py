"""Runtime-peel — SGLang-backed decode model for standard GQA models (gpt-oss).

Lean sibling of ``sglang_decode_model.SGLangDecodeModel`` (the MLA/NSA, GLM-5
path, which carries heavy NSA-specific debug scaffolding). This class is a
drop-in for the PSM's ``self.model`` whose forward routes decode to a decode-only
SGLang ``ModelRunner`` reading BatchGen's GQA K+V via ``BatchGenGQAKVAdapter``.

Same worker contract as the MLA model: the worker binds decode state on
``AttnWrapperBase`` each step (``cache_seqlens`` [B] int32, ``position_ids``,
``gpu_paged_kv_manager`` — single, non-dual for GQA), then calls
``self.model(new_tokens, ...)``; the page table ``gpu_table`` is batch/slot-ordered
after ``rebuild_page_table`` so batch row i == slot i.

Off-by-one (BatchGen convention): positions = cache_seqlens - 1; the input token's
KV is written at the slot for position cache_seqlens-1. SGLang's paged attention
reads the block table via ``req_to_token_pool.req_to_token[req_pool_indices,:ctx]``,
so per step we populate it from BatchGen's ``gpu_table`` (same as the MLA path).

dp-attention: gpt-oss runs ``--dp N --enable-dp-attention`` (like GLM-5), so an
idle rank (0 local decode seqs) must still join the pure-TP-MoE all-gather — pad it
with one dummy IDLE token (output discarded), identical to the MLA path.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from batchgen.runner_adapter.sglang_decode_runner import (
    build_decode_forward_batch,
    inject_kv_adapter_gqa,
)

_PAGE = 64


class SGLangGQADecodeModel(nn.Module):
    """Decode-only GQA model (gpt-oss) backed by an SGLang ModelRunner."""

    _batchgen_skip_fp8_unregistration = True

    def __init__(self, runner, core_engine):
        super().__init__()
        self._runner = runner
        self._core_engine = core_engine
        self._adapter_injected = False
        # Worker reads next(self.model.parameters()).device for sampling tensors;
        # real params live in the SGLang runner, so expose a device anchor.
        self._device_anchor = nn.Parameter(
            torch.zeros(1, device="cuda"), requires_grad=False
        )

    def _ensure_adapter(self):
        if self._adapter_injected:
            return
        from batchgen.models.wrappers import AttnWrapperBase

        kv_mgr = getattr(AttnWrapperBase, "gpu_paged_kv_manager", None)
        if kv_mgr is None:
            raise RuntimeError(
                "SGLangGQADecodeModel: AttnWrapperBase.gpu_paged_kv_manager not "
                "bound at first decode forward."
            )
        inject_kv_adapter_gqa(
            self._runner,
            kv_mgr,
            page_size=_PAGE,
            layer_num=getattr(
                self._runner.model_config, "num_hidden_layers", None
            ),
        )
        self._adapter_injected = True

    def _populate_req_to_token(self, page_table, cache_seqlens):
        """req_to_token[row, t] = page_table[row, t//page]*page + t%page."""
        r2t = self._runner.req_to_token_pool.req_to_token
        batch_size = int(page_table.shape[0])
        for row in range(batch_size):
            ctx = int(cache_seqlens[row].item())
            if ctx <= 0:
                continue
            t = torch.arange(ctx, device=page_table.device)
            pages = page_table[row, t // _PAGE].to(torch.int64)
            locs = pages * _PAGE + (t % _PAGE)
            r2t[row, :ctx] = locs.to(r2t.dtype)

    def forward(self, input_ids, attention_mask=None, position_ids=None, use_cache=False):
        from batchgen.models.wrappers import AttnWrapperBase

        self._ensure_adapter()
        input_ids = input_ids.view(-1)
        dev = input_ids.device

        cache_seqlens = AttnWrapperBase.cache_seqlens.view(-1).to(torch.int32)
        batch_size = int(cache_seqlens.numel())

        # Idle dp-attention rank: pad one dummy IDLE token so this rank joins the
        # MoE all-gather; its output is discarded (see module docstring).
        if batch_size == 0:
            from sglang.srt.model_executor.forward_batch_info import ForwardMode

            d_tok = torch.zeros(1, dtype=torch.int64, device=dev)
            d_pos = torch.zeros(1, dtype=torch.int64, device=dev)
            d_cs = torch.ones(1, dtype=torch.int32, device=dev)
            d_pt = torch.zeros(1, 1, dtype=torch.int32, device=dev)
            d_rpi = torch.zeros(1, dtype=torch.int64, device=dev)
            d_ocl = torch.zeros(1, dtype=torch.int64, device=dev)
            self._runner.req_to_token_pool.req_to_token[0, 0] = 0
            fb = build_decode_forward_batch(
                self._runner, d_tok, d_pos, d_cs, d_pt, d_rpi, d_ocl,
                forward_mode=ForwardMode.IDLE,
            )
            out = self._runner.forward(fb)
            nl = out.logits_output.next_token_logits
            empty = torch.empty(0, nl.shape[-1], dtype=nl.dtype, device=dev)
            return SimpleNamespace(logits=empty.unsqueeze(1))

        kv_mgr = AttnWrapperBase.gpu_paged_kv_manager
        # gpu_table: [num_slots, max_pages] physical page indices, slot order.
        _, _, gpu_table = kv_mgr.get_layer_kv_with_page_table(0)
        page_table = gpu_table[:batch_size].to(torch.int32)
        req_pool_indices = torch.arange(batch_size, dtype=torch.int64, device=dev)

        if position_ids is not None:
            positions = position_ids.view(-1).to(torch.int64)
        else:
            positions = cache_seqlens.to(torch.int64) - 1

        # out_cache_loc = write slot for position ctx-1 = phys_page*page + offset.
        pos = cache_seqlens.to(torch.int64) - 1
        page_col = (pos // _PAGE).view(batch_size, 1)
        phys = page_table.to(torch.int64).gather(1, page_col).view(batch_size)
        out_cache_loc = phys * _PAGE + (pos % _PAGE)

        self._populate_req_to_token(page_table, cache_seqlens)

        fb = build_decode_forward_batch(
            self._runner,
            input_ids,
            positions,
            cache_seqlens,
            page_table,
            req_pool_indices,
            out_cache_loc,
        )
        out = self._runner.forward(fb)
        logits = out.logits_output.next_token_logits
        # Worker reads outputs.logits[:, -1, :] -> [B, 1, vocab].
        return SimpleNamespace(logits=logits.unsqueeze(1))
