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
    _sel_dumped = False   # one-shot indexer-selection + per-layer-stats probe
    _step_tap_hooks = False  # persistent step-tap hooks installed (layer 0)

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
        # batch. SGLang's forward_idle REQUIRES dp-attention IDLE batches to be
        # PADDED to batch_size>0 (model_runner.py:2350 "In DP Attention, IDLE
        # batches are padded (batch_size > 0) for MLP sync") — a batch_size=0 idle
        # batch contributes 0 to the dp-attention<->pure-TP-MoE all-gather, which
        # corrupts the gathered MoE buffer on ALL ranks (mlp_out diverges, decode
        # garbage). Standalone SGLang never has a truly-idle rank (global_num_tokens
        # has no zeros). So pad this idle rank with ONE dummy decode token at the
        # free scratch slot 0 (idle ranks hold no active KV there) so it joins the
        # MoE collective like every other rank; its output is discarded.
        if batch_size == 0:
            from sglang.srt.model_executor.forward_batch_info import ForwardMode

            d_tok = torch.zeros(1, dtype=torch.int64, device=dev)   # dummy token
            d_pos = torch.zeros(1, dtype=torch.int64, device=dev)   # position 0
            d_cs = torch.ones(1, dtype=torch.int32, device=dev)     # ctx=1 (itself)
            d_pt = torch.zeros(1, 1, dtype=torch.int32, device=dev)  # [1 seq, page 0]
            d_rpi = torch.zeros(1, dtype=torch.int64, device=dev)   # req row 0
            d_ocl = torch.zeros(1, dtype=torch.int64, device=dev)   # write slot 0
            # NSA reads req_to_token[req_pool_indices,:ctx]; map the dummy token to
            # slot 0 so init_forward_metadata builds a valid 1-token page table.
            self._runner.req_to_token_pool.req_to_token[0, 0] = 0
            fb = build_decode_forward_batch(
                self._runner, d_tok, d_pos, d_cs, d_pt, d_rpi, d_ocl,
                forward_mode=ForwardMode.IDLE,
            )
            out = self._runner.forward(fb)
            # Discard the dummy's logits — the worker expects 0 tokens for an idle
            # rank. Return an empty [0,1,vocab] tensor.
            nl = out.logits_output.next_token_logits
            empty = torch.empty(0, nl.shape[-1], dtype=nl.dtype, device=dev)
            return SimpleNamespace(logits=empty.unsqueeze(1))
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
                # Offload/reload faithfulness probe: per-logical-token K fingerprint
                # across pages. The 8 MMLU prompts share a long prefix, so EARLY
                # logical tokens have IDENTICAL fingerprints across seqs and LATE
                # tokens diverge. A faithful reload => the identical->divergent
                # boundary is MONOTONIC in logical position (same physical page span
                # holds the right tokens). A per-page reload scramble => the
                # identical fingerprints appear at SCATTERED logical positions.
                # fp = sum(K[:8]); compare these lines across the 8 ranks/seqs.
                fps = {}
                for p in [0, 1, 32, 64, 128, 256, 512, ctx0 // 2,
                          max(ctx0 - 65, 0), max(ctx0 - 2, 0)]:
                    if p < ctx0:
                        fps[p] = round(float(prefillK[p, :8].float().sum()), 5)
                _lg.getLogger().warning(
                    "[RTPEEL-KVFP] ctx0=%d pages=%s fp(logical_tok->sum K[:8])=%s",
                    ctx0, page_table[0, :4].tolist(), fps,
                )
                # Per-layer reload completeness: the async per-layer host->GPU
                # reload may be incomplete for DEEP layers when decode reads them.
                # I only validated LAYER 0 above. For a shared-prefix token (256)
                # each layer's K must be NONZERO + sane (the prefix hidden state is
                # identical across prompts at every layer, so fp@256 must be equal
                # across the 8 seqs per layer). High zerofrac / fp==0 on a deep
                # layer => that layer's KV was NOT reloaded to GPU.
                nlyr = len(self._runner.model.model.layers)
                lyr = {}
                for L in sorted({0, 1, nlyr // 4, nlyr // 2,
                                 3 * nlyr // 4, nlyr - 1}):
                    try:
                        kb = self._runner.token_to_kv_pool.get_key_buffer(L)
                        ks = kb[locs0, 0, :]  # [ctx0, 576]
                        lyr[L] = (round(float(ks[256, :8].float().sum()), 5),
                                  round(float((ks == 0).float().mean()), 4))
                    except Exception as _e2:  # noqa: BLE001
                        lyr[L] = f"ERR:{_e2!r}"
                _lg.getLogger().warning(
                    "[RTPEEL-KVLYR] nlayers=%d per-layer (fp@tok256, zerofrac)=%s",
                    nlyr, lyr,
                )
            except Exception as _e:  # noqa: BLE001
                import logging as _lg
                _lg.getLogger().warning("[RTPEEL-KVDBG] dump failed: %r", _e)

        # One-shot selection + per-layer-stats probe. For ctx < index_topk the
        # indexer MUST select all ctx tokens; a valid set != {0..ctx-1} localizes
        # the bug to the indexer selection, == {0..ctx-1} points downstream
        # (MLA gather/compute). Per-layer absmax/std flags any layer that blows up.
        _probe = None
        if not SGLangDecodeModel._sel_dumped:
            SGLangDecodeModel._sel_dumped = True
            _probe = self._install_probe_hooks()

        # Cross-path step-tap (layer 0): arm + install persistent hooks routing
        # layer-0 hidden_in/attn_out/hidden_out/indexer_sel through step_tap so the
        # SGLang path dumps the same named points as native for offline diff.
        from batchgen.debug import step_tap
        self._ensure_step_tap_hooks()
        step_tap.begin()

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

        step_tap.tap("logits", logits, layer_id=step_tap.TAP_LAYER)
        step_tap.flush()

        if _probe is not None:
            self._log_probe(_probe, int(cache_seqlens[0].item()))

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

    def _ensure_step_tap_hooks(self):
        """Install persistent layer-0 hooks routing intermediates to step_tap.
        Names match the native taps in Glm5DecoderLayer.forward (hidden_in,
        attn_out, hidden_out) plus indexer_sel. step_tap.tap() self-gates, so the
        hooks are cheap no-ops unless a run is armed via begin()."""
        if SGLangDecodeModel._step_tap_hooks:
            return
        from batchgen.debug import step_tap

        try:
            layers = self._runner.model.model.layers
            n = len(layers)

            def _mk(L):
                def _pre(_m, args, kwargs):
                    hs = args[1] if len(args) > 1 else kwargs.get("hidden_states")
                    step_tap.tap("hidden_in", hs, layer_id=L)

                def _layer_out(_m, _i, out):
                    # decoder layer returns (hidden_states, residual) fused; the
                    # residual-stream value = hidden + residual.
                    if (isinstance(out, (tuple, list)) and len(out) >= 2
                            and isinstance(out[0], torch.Tensor)
                            and isinstance(out[1], torch.Tensor)):
                        step_tap.tap("hidden_out", out[0] + out[1], layer_id=L)
                    else:
                        hs = out[0] if isinstance(out, (tuple, list)) else out
                        step_tap.tap("hidden_out", hs, layer_id=L)

                def _attn_out(_m, _i, out):
                    a = out[0] if isinstance(out, (tuple, list)) else out
                    step_tap.tap("attn_out", a, layer_id=L)

                def _mlp_out(_m, _i, out):
                    o = out[0] if isinstance(out, (tuple, list)) else out
                    step_tap.tap("mlp_out", o, layer_id=L)

                def _mlp_pre(_m, args, kwargs):
                    # MoE input = the dp-gathered buffer (all dp-rank tokens).
                    x = args[0] if args else kwargs.get("hidden_states")
                    step_tap.tap("mlp_in", x, layer_id=L)

                def _gate_out(_m, _i, out):
                    rl = out[0] if isinstance(out, (tuple, list)) else out
                    step_tap.tap("router_logits", rl, layer_id=L)

                def _topk_out(_m, _i, out):
                    ids = getattr(out, "topk_ids", None)
                    if ids is None and isinstance(out, (tuple, list)):
                        ids = out[1] if len(out) > 1 else out[0]
                    step_tap.tap("topk_ids", ids, layer_id=L)

                return (_pre, _layer_out, _attn_out, _mlp_out, _mlp_pre,
                        _gate_out, _topk_out)

            for L in step_tap.TAP_LAYERS:
                if L >= n:
                    continue
                lyr = layers[L]
                pre, lout, aout, mout, mpre, gout, tout = _mk(L)
                lyr.register_forward_pre_hook(pre, with_kwargs=True)
                lyr.register_forward_hook(lout)
                lyr.self_attn.register_forward_hook(aout)
                if hasattr(lyr, "mlp"):
                    lyr.mlp.register_forward_hook(mout)
                    lyr.mlp.register_forward_pre_hook(mpre, with_kwargs=True)
                    if hasattr(lyr.mlp, "gate"):
                        lyr.mlp.gate.register_forward_hook(gout)
                    if hasattr(lyr.mlp, "topk"):
                        lyr.mlp.topk.register_forward_hook(tout)
                if L == 0 and hasattr(lyr.self_attn, "indexer"):
                    lyr.self_attn.indexer.register_forward_hook(
                        lambda _m, _i, o: step_tap.tap(
                            "indexer_sel",
                            o[0] if isinstance(o, (tuple, list)) else o, layer_id=0))
            SGLangDecodeModel._step_tap_hooks = True
        except Exception as _e:  # noqa: BLE001
            import logging as _lg
            _lg.getLogger().warning("[STEP-TAP] sglang hook install failed: %r", _e)

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

    def _install_probe_hooks(self):
        """Hook layer-0 indexer (capture topk_indices) + a few layers' output
        stats. Returns (capture_dict, handles) to log + remove after forward."""
        import logging as _lg

        cap = {"topk": None, "layers": [], "attn0_in": None, "attn0_out": None}
        handles = []
        try:
            layers = self._runner.model.model.layers

            def _idx_hook(_m, _i, out):
                t = out[0] if isinstance(out, (tuple, list)) else out
                cap["topk"] = t.detach() if isinstance(t, torch.Tensor) else None

            def _mk_layer_hook(idx):
                def _h(_m, _i, out):
                    hs = out[0] if isinstance(out, (tuple, list)) else out
                    if isinstance(hs, torch.Tensor) and hs.numel():
                        cap["layers"].append(
                            (idx, float(hs.float().abs().max()),
                             float(hs.float().std())))
                return _h

            def _attn_hook(_m, inp, out):
                # self_attn.forward(positions, hidden_states, forward_batch) ->
                # attn contribution. Capture the decode token's attn IN/OUT. The
                # decode INPUT token is the same (785) for all seqs but each seq's
                # CONTEXT differs, so OUT must DIFFER across seqs if attention uses
                # the KV; identical/near-zero OUT => context ignored (peel clears).
                o = out[0] if isinstance(out, (tuple, list)) else out
                cap["attn0_out"] = o.detach() if isinstance(o, torch.Tensor) else None
                hs_in = inp[1] if len(inp) > 1 else (inp[0] if inp else None)
                cap["attn0_in"] = (hs_in.detach()
                                   if isinstance(hs_in, torch.Tensor) else None)

            handles.append(
                layers[0].self_attn.indexer.register_forward_hook(_idx_hook))
            handles.append(
                layers[0].self_attn.register_forward_hook(_attn_hook))
            n = len(layers)
            for idx in sorted({0, 1, n // 2, n - 2, n - 1}):
                handles.append(
                    layers[idx].register_forward_hook(_mk_layer_hook(idx)))
        except Exception as _e:  # noqa: BLE001
            _lg.getLogger().warning("[RTPEEL-SEL] hook setup failed: %r", _e)
        return cap, handles

    def _log_probe(self, probe, ctx0):
        import logging as _lg

        cap, handles = probe
        for h in handles:
            h.remove()
        try:
            tk = cap["topk"]
            if tk is None:
                _lg.getLogger().warning("[RTPEEL-SEL] indexer returned no topk")
            else:
                row0 = tk[0].reshape(-1)
                valid = row0[row0 >= 0]
                vmin = int(valid.min().item()) if valid.numel() else -1
                vmax = int(valid.max().item()) if valid.numel() else -1
                uniq = int(torch.unique(valid).numel()) if valid.numel() else 0
                # For ctx < index_topk, a correct selection = all ctx tokens.
                expect_all = (valid.numel() == ctx0 and uniq == ctx0
                              and vmin == 0 and vmax == ctx0 - 1)
                _lg.getLogger().warning(
                    "[RTPEEL-SEL] ctx0=%d topk_shape=%s valid_n=%d uniq=%d "
                    "min=%d max=%d select_all_ctx=%s sample=%s",
                    ctx0, tuple(tk.shape), int(valid.numel()), uniq, vmin, vmax,
                    expect_all, valid[:12].tolist(),
                )
            _lg.getLogger().warning(
                "[RTPEEL-SEL] per-layer (idx,absmax,std)=%s", cap["layers"])

            def _fp(t):
                if not isinstance(t, torch.Tensor) or not t.numel():
                    return None
                f = t.float().reshape(-1)
                return (round(float(f.mean()), 5), round(float(f.std()), 5),
                        round(float(f.abs().max()), 4),
                        round(float((f == 0).float().mean()), 3),
                        [round(x, 4) for x in f[:6].tolist()])
            # Decode INPUT token is 785 for all seqs (same embed) but CONTEXT
            # differs per seq. attn0 OUT[:6] identical across seqs => attention
            # ignores the KV (peel clears it). OUT differs => attention uses
            # context => corruption is downstream (MoE/dp-gather/lm_head).
            _lg.getLogger().warning(
                "[RTPEEL-ATTN0] ctx0=%d layer0 self_attn IN(mean,std,absmax,zf,[:6])=%s "
                "OUT(mean,std,absmax,zf,[:6])=%s",
                ctx0, _fp(cap.get("attn0_in")), _fp(cap.get("attn0_out")))
        except Exception as _e:  # noqa: BLE001
            _lg.getLogger().warning("[RTPEEL-SEL] log failed: %r", _e)
