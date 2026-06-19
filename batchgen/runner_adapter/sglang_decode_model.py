"""M1 runtime-peel — SGLang-backed decode model (PSM-level drop-in).

BatchGen's PSM `configure_decoding` returns `self.model`, and the worker drives
decode generically:

    outputs = self.model(new_tokens, attention_mask=..., position_ids=..., use_cache=False)
    next = self._select_tokens(outputs.logits[:, -1, :], ...)

`SGLangDecodeModel` is a drop-in for that `self.model` whose forward routes to a
decode-only SGLang `ModelRunner` reading BatchGen's KV via `BatchGenNSAKVAdapter`.
The native decode model is NOT built when the sglang backend is on (R4: can't hold
both weight sets). The worker is untouched — it just calls `self.model(...)`.

Decode state is read off `AttnWrapperBase` (the worker binds it every step, model-
agnostically, at `_bind_decode_attention_metadata`) + the GPU paged KV managers:
  cache_seqlens [B] int32, gpu_paged_kv_manager{,_aux}, position_ids [B,1].
The page table `gpu_table` ([num_slots, max_pages] physical page indices) is
batch/slot-ordered after the worker's `rebuild_page_table`, so batch row i == slot i.

Off-by-one (matches BatchGen's convention): cache_seqlens ≡ SGLang seq_lens;
positions = cache_seqlens-1 (the input token's position); the input token's KV is
written at the slot for position cache_seqlens-1.

Page-table bridge (load-bearing): SGLang's NSA backend reads the block table as
`req_to_token_pool.req_to_token[req_pool_indices, :seqlen]` (flat token locs), NOT
ForwardBatch.page_table. So per step we populate req_to_token[slot, t] =
gpu_table[slot, t//page]*page + t%page.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from batchgen.runner_adapter.sglang_decode_runner import (
    build_decode_forward_batch,
    inject_kv_adapter,
)

_PAGE = 64


class SGLangDecodeModel(nn.Module):
    """Decode-only GLM-5 model backed by an SGLang ModelRunner."""

    _logged_once = False  # one-shot decode KV diagnostic guard (class-level)
    _dbg_steps = 0        # token-flow audit step counter
    _DBG_MAX = 8          # log the first N decode steps

    def __init__(self, runner, core_engine):
        super().__init__()
        self._runner = runner
        self._core_engine = core_engine
        self._adapter_injected = False
        # The worker reads `next(self.model.parameters()).device` to place sampling
        # tensors (_build_sampling_tensors). This wrapper owns no real parameters
        # (they live inside the SGLang runner), so expose a tiny device-anchor on
        # the decode device (cuda:local_rank — already set by the worker before
        # configure_decoding) so parameters() is non-empty and reports the right
        # device.
        self._device_anchor = nn.Parameter(
            torch.zeros(1, device="cuda"), requires_grad=False
        )

    def _ensure_adapter(self):
        if self._adapter_injected:
            return
        # The worker binds the live decode KV managers onto AttnWrapperBase before
        # every decode forward (gpu_paged_kv_manager = primary/MLA,
        # gpu_paged_kv_manager_aux = indexer) — the same source the native GLM-5
        # decode reads (wrappers.py:817-818). core_engine.gpu_paged_kv_manager is
        # not populated in this build, so read from AttnWrapperBase.
        from batchgen.models.wrappers import AttnWrapperBase

        primary = getattr(AttnWrapperBase, "gpu_paged_kv_manager", None)
        aux = getattr(AttnWrapperBase, "gpu_paged_kv_manager_aux", None)
        if primary is None or aux is None:
            raise RuntimeError(
                "SGLangDecodeModel: AttnWrapperBase KV managers not bound "
                "(gpu_paged_kv_manager{,_aux} None at first decode forward)."
            )
        inject_kv_adapter(self._runner, primary, aux)
        self._adapter_injected = True

    def forward(self, input_ids, attention_mask=None, position_ids=None, use_cache=False):
        from batchgen.models.wrappers import AttnWrapperBase

        self._ensure_adapter()
        input_ids = input_ids.view(-1)
        dev = input_ids.device

        cache_seqlens = AttnWrapperBase.cache_seqlens.view(-1).to(torch.int32)  # [B]
        batch_size = int(cache_seqlens.numel())

        # --- IDLE dp-attention rank: 0 local decode sequences ----------------- #
        # With dp16 and fewer prompts than ranks, some ranks have an empty decode
        # batch. The worker still runs forward on every rank (MoE all-to-all + the
        # dp-gather collectives need all 16 in lockstep — see batchgen_worker.py
        # ~9746), but this rank's GPU page table is NOT built (worker `if batch:`
        # guard ~9724), so primary.get_layer_kv_with_page_table(0) would raise.
        # Build a TRUE IDLE ForwardBatch (empty tensors): SGLang's
        # prepare_mlp_sync_batch derives the pad from the cross-rank max and
        # forward_idle gates attn-metadata init behind batch_size>0, so no KV/page
        # table is touched. The dp all-gather inside build_decode_forward_batch
        # still runs (contributing this rank's 0), keeping all ranks in lockstep.
        if batch_size == 0:
            from sglang.srt.model_executor.forward_batch_info import ForwardMode

            empty_i64 = torch.empty(0, dtype=torch.int64, device=dev)
            empty_i32 = torch.empty(0, dtype=torch.int32, device=dev)
            fb = build_decode_forward_batch(
                self._runner,
                empty_i64,  # new_tokens
                empty_i64,  # positions
                empty_i32,  # cache_seqlens
                empty_i32,  # page_table (unused; idle reads no KV)
                empty_i64,  # req_pool_indices
                empty_i64,  # out_cache_loc
                forward_mode=ForwardMode.IDLE,
            )
            out = self._runner.forward(fb)
            logits = out.logits_output.next_token_logits  # [0, vocab]
            return SimpleNamespace(logits=logits.unsqueeze(1))  # [0, 1, vocab]
        # ---------------------------------------------------------------------- #

        primary = AttnWrapperBase.gpu_paged_kv_manager
        # gpu_table: [num_slots, max_pages] physical page indices, batch/slot order.
        _, _, gpu_table = primary.get_layer_kv_with_page_table(0)
        page_table = gpu_table[:batch_size].to(torch.int32)                    # [B, max_pages]
        req_pool_indices = torch.arange(batch_size, dtype=torch.int64, device=dev)

        # positions = cache_seqlens - 1 (the input token's position).
        if position_ids is not None:
            positions = position_ids.view(-1).to(torch.int64)
        else:
            positions = cache_seqlens.to(torch.int64) - 1

        # out_cache_loc = write slot for position (ctx-1) = phys_page*64 + offset.
        pos = cache_seqlens.to(torch.int64) - 1                                # [B]
        page_col = (pos // _PAGE).view(batch_size, 1)
        phys = page_table.to(torch.int64).gather(1, page_col).view(batch_size)
        out_cache_loc = phys * _PAGE + (pos % _PAGE)                           # [B]

        # Page-table bridge: SGLang NSA reads req_to_token[req_pool_indices,:ctx].
        self._populate_req_to_token(page_table, cache_seqlens)

        # One-shot decode diagnostic (KV-layout audit): dump metadata + the MLA-K
        # content the adapter reads at the prefill slots. Zeros there => the prefill
        # KV is not resident on the GPU pages (load/slot bug); sane values => a
        # read-flow/semantic issue. Logs once per process.
        if not SGLangDecodeModel._logged_once:
            SGLangDecodeModel._logged_once = True
            try:
                import logging as _lg
                kbuf = self._runner.token_to_kv_pool.get_key_buffer(0)  # [N,1,576] BF16 view
                ctx0 = int(cache_seqlens[0].item())
                # slots for seq 0's tokens 0..ctx0-1
                pcol = (torch.arange(ctx0, device=dev) // _PAGE)
                locs0 = (page_table[0].to(torch.int64)[pcol] * _PAGE
                         + (torch.arange(ctx0, device=dev) % _PAGE))
                prefillK = kbuf[locs0, 0, :]              # [ctx0, 576]
                aux_primary = AttnWrapperBase.gpu_paged_kv_manager
                slot_seq = getattr(aux_primary._gpu_page_table_manager, "slot_to_seq_id", None)
                _lg.getLogger().warning(
                    "[RTPEEL-KVDBG] bsz=%d cache_seqlens=%s positions=%s out_cache_loc=%s "
                    "req_pool_idx=%s page_table[0,:4]=%s slot_to_seq_id=%s | KBUF shape=%s dtype=%s "
                    "| prefillK[seq0] mean=%.4g std=%.4g absmax=%.4g zerofrac=%.3f | "
                    "sample[0,:6]=%s sample[%d,:6]=%s",
                    batch_size, cache_seqlens.tolist(), positions.view(-1).tolist()[:8],
                    out_cache_loc.tolist()[:8], req_pool_indices.tolist()[:8],
                    page_table[0, :4].tolist(), (list(slot_seq)[:8] if slot_seq else None),
                    tuple(kbuf.shape), kbuf.dtype,
                    float(prefillK.float().mean()), float(prefillK.float().std()),
                    float(prefillK.float().abs().max()),
                    float((prefillK == 0).float().mean()),
                    prefillK[0, :6].float().tolist(),
                    max(ctx0 - 1, 0), prefillK[max(ctx0 - 1, 0), :6].float().tolist(),
                )
            except Exception as _e:  # noqa: BLE001
                import logging as _lg
                _lg.getLogger().warning("[RTPEEL-KVDBG] dump failed: %r", _e)

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
        logits = out.logits_output.next_token_logits          # [B, vocab]

        # Token-flow audit (POIS: prefill is unchanged BatchGen code, so the first
        # generated token must be correct; find where decode garbage begins). Log
        # the decode INPUT token + the OUTPUT top-1 for the first few steps.
        if SGLangDecodeModel._dbg_steps < SGLangDecodeModel._DBG_MAX:
            SGLangDecodeModel._dbg_steps += 1
            try:
                import logging as _lg
                top1 = logits.argmax(dim=-1)
                top5 = logits[0].topk(5).indices.tolist() if logits.shape[0] else []
                _lg.getLogger().warning(
                    "[RTPEEL-TOK step=%d] in_tokens=%s out_top1=%s out_top5[seq0]=%s "
                    "logit_absmax=%.4g",
                    SGLangDecodeModel._dbg_steps, input_ids.tolist()[:8],
                    top1.tolist()[:8], top5,
                    float(logits.float().abs().max()) if logits.numel() else -1.0,
                )
            except Exception as _e:  # noqa: BLE001
                import logging as _lg
                _lg.getLogger().warning("[RTPEEL-TOK] dump failed: %r", _e)

        # Worker reads outputs.logits[:, -1, :] -> shape [B, 1, vocab].
        return SimpleNamespace(logits=logits.unsqueeze(1))

    def _populate_req_to_token(self, page_table, cache_seqlens):
        """Fill SGLang's req_to_token[row, :ctx] from BatchGen's gpu_table.

        req_to_token[row, t] = page_table[row, t // page] * page + t % page.
        Row == slot == batch index (req_pool_indices = arange(B)).
        """
        r2t = self._runner.req_to_token_pool.req_to_token       # [max_reqs, max_ctx]
        batch_size = int(page_table.shape[0])
        for row in range(batch_size):
            ctx = int(cache_seqlens[row].item())
            if ctx <= 0:
                continue
            t = torch.arange(ctx, device=page_table.device)
            pages = page_table[row, t // _PAGE].to(torch.int64)
            locs = pages * _PAGE + (t % _PAGE)
            r2t[row, :ctx] = locs.to(r2t.dtype)
