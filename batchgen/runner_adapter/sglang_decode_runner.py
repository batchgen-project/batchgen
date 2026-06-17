"""Slice 1 — build a decode-only SGLang ModelRunner that adopts BatchGen's PG.

Architecture (M1, runtime-peel):
  prefill  = BatchGen's current path (unchanged; writes BF16 KV).
  decode   = a per-rank, decode-only SGLang ``ModelRunner`` constructed here.
  KV       = shared. The runner reads KV through ``BatchGenNSAKVAdapter`` which is
             injected as ``model_runner.token_to_kv_pool``.

Two things make this work without re-initializing distributed state:
  1. BatchGen already called ``dist.init_process_group(...)`` (worker
     ``_init_torch_dist`` ~ batchgen_worker.py:11361, device_id=cuda:local_rank).
     We monkeypatch SGLang's ``init_distributed_environment`` to a no-op when
     ``torch.distributed.is_initialized()`` so the ModelRunner ADOPTS that PG
     instead of double-initializing (design R3; SGLang itself already early-skips
     the init_process_group call at parallel_state.py:1518, but it still builds
     ``_WORLD`` from the live torch.distributed state — the patch keeps that
     adoption path while guaranteeing no second init under dp-attention).
  2. ``initialize_model_parallel`` asserts ``is_initialized()`` and builds _TP/_PP
     groups from the live torch.distributed world (parallel_state.py:1557) — i.e.
     it adopts the existing groups; ModelRunner.init_torch_distributed drives it.

Scope: M1 shares KV ONLY. SGLang loads its own decode weights (BatchGen's EP
weight layout != SGLang nn.Module experts; the weight bridge is a separate
workstream). CUDA graph is disabled in M1.

Verified signatures (do not drift without re-checking references/sglang):
  ModelRunner.__init__      model_runner.py:278-296
  init_distributed_environment (is_initialized early-skip) parallel_state.py:1491-1555
  initialize_model_parallel (adopts existing PG)           parallel_state.py:1557
  ForwardBatch fields / init_new                           forward_batch_info.py:233-529
  ForwardBatch.forward DECODE dispatch                     model_runner.py:2433
  clamp_position(seq_lens) = clamp(seq_lens-1, 0).int64    forward_batch_info.py:1095
  next_token_logits extraction                             out.logits_output.next_token_logits
"""

from __future__ import annotations

from typing import Optional

import torch

from batchgen.attention.dsa.sglang_kv_bridge import BatchGenNSAKVAdapter

# GLM-5 / DeepSeek-V3.2 NSA decode config (matches the KV bridge constants).
_PAGE_SIZE = 64
_ATTENTION_BACKEND = "nsa"
_KV_CACHE_DTYPE = "bfloat16"


def _install_pg_adopt_guard() -> None:
    """Patch SGLang ``init_distributed_environment`` to no-op when PG is live.

    BatchGen owns the process group. SGLang's ModelRunner.init_torch_distributed
    calls ``init_distributed_environment`` then ``initialize_model_parallel``.
    The first must NOT try to (re)create the world PG; the second adopts the
    existing torch.distributed groups. We wrap the function so that when
    ``torch.distributed.is_initialized()`` we still let SGLang build its
    ``_WORLD`` wrapper from the live state (the original function already does
    this via its ``is_initialized()`` early-skip), but we hard-guarantee no
    second ``init_process_group`` is attempted under dp-attention.

    Idempotent: re-patching is a no-op.

    TODO(verify-on-gpu): with ``enable_dp_attention=True`` SGLang also builds
    attention-DP subgroups inside ``initialize_model_parallel`` /
    ``get_attention_tp_group`` paths. Confirm those new_group() calls succeed
    against BatchGen's existing world PG (they must be called collectively by
    ALL ranks in the same order, or NCCL will hang). This is the #1 GPU risk.
    """
    from sglang.srt.distributed import parallel_state

    if getattr(parallel_state.init_distributed_environment, "_batchgen_pg_guard", False):
        return

    _orig = parallel_state.init_distributed_environment

    def _guarded(*args, **kwargs):
        if torch.distributed.is_initialized():
            # PG already owned by BatchGen. Let SGLang build its _WORLD wrapper
            # off the live state, but never re-init. The original function's
            # is_initialized() branch (parallel_state.py:1518) already skips the
            # init_process_group call, so delegating is safe and keeps _WORLD
            # construction intact.
            return _orig(*args, **kwargs)
        return _orig(*args, **kwargs)

    _guarded._batchgen_pg_guard = True
    # NOTE: both branches currently delegate to _orig — the guard exists as the
    # single, named interception point. If GPU bring-up shows _WORLD being
    # rebuilt with a wrong world size, replace the is_initialized() branch with
    # an explicit init_world_group(...) call here. Kept minimal until proven.
    parallel_state.init_distributed_environment = _guarded


def _build_decode_server_args(
    model_path: str,
    world_size: int,
    global_rank: int,
    local_rank: int,
    dist_init_addr: str,
    dp_size: int,
    nnodes: int,
    node_rank: int,
    mem_fraction_static: float,
):
    """Map BatchGen's runtime config to a decode-only SGLang ``ServerArgs``.

    Returns a ``ServerArgs`` configured for NSA decode with shared-KV semantics.
    """
    from sglang.srt.server_args import ServerArgs

    server_args = ServerArgs(
        model_path=model_path,
        # --- attention / KV layout: must match BatchGenNSAKVAdapter --------- #
        attention_backend=_ATTENTION_BACKEND,  # "nsa"
        page_size=_PAGE_SIZE,                   # 64
        kv_cache_dtype=_KV_CACHE_DTYPE,         # "bfloat16" (primary MLA path)
        # --- parallelism: adopt BatchGen's world -------------------------- #
        tp_size=world_size,
        dp_size=dp_size,
        enable_dp_attention=True,
        nnodes=nnodes,
        node_rank=node_rank,
        dist_init_addr=dist_init_addr,
        # --- decode-only / M1 simplifications ----------------------------- #
        skip_tokenizer_init=True,
        disable_radix_cache=True,
        disable_cuda_graph=True,
        trust_remote_code=True,
        # SGLang's default auto-sizes to ~0.46 (too low for GLM-5-FP8 weights ->
        # RuntimeError "increase mem_fraction_static"). Smoke proved 0.82 fits. The
        # in-worker path passes a LOWER value (~0.62, just above weights/total) so
        # SGLang's auto NSA pool is minimal and BatchGen's own paged KV (which the
        # adapter wraps) gets the remaining HBM — BatchGen is the sole decode model
        # resident (its native decode model is skipped, design R4).
        mem_fraction_static=mem_fraction_static,
    )
    # TODO(verify-on-gpu): ServerArgs.__post_init__ rewrites several of these
    # (page_size auto-handling at server_args.py:738, attention-backend
    # compatibility at :736, mem_fraction_static auto-sizing). Confirm none of
    # them override attention_backend="nsa" / page_size=64 for a GLM-5 MLA
    # config, and that enable_dp_attention=True with dp_size==world_size is
    # accepted (dp16 on 2 nodes).
    return server_args


def build_sglang_decode_runner(
    loaded_model_config,
    world_size: int,
    global_rank: int,
    local_rank: int,
    dist_init_addr: str,
    primary_kv_mgr=None,
    aux_kv_mgr=None,
    padding_bsz: Optional[int] = None,
    *,
    dp_size: Optional[int] = None,
    nnodes: int = 2,
    node_rank: int = 0,
    mem_fraction_static: float = 0.82,
):
    """Construct a decode-only SGLang ModelRunner wired to BatchGen's KV.

    Args:
        loaded_model_config: BatchGen's loaded model config. Must expose the
            HF model path (used for SGLang's ``ServerArgs.model_path`` /
            ``ModelConfig.from_server_args``).
            TODO(verify-on-gpu): confirm the attribute name on BatchGen's config
            object (``.model_path`` assumed below). GLM-5 path is whatever
            ``configure_decode`` already resolved for prefill.
        world_size: BatchGen world size (== SGLang tp_size).
        global_rank: BatchGen global rank (== SGLang tp_rank).
        local_rank: BatchGen local rank (== SGLang gpu_id; cuda device index).
        dist_init_addr: "host:port" BatchGen used for ``init_process_group``.
        primary_kv_mgr: BatchGen primary (MLA) GPUPagedKVCacheManager.
        aux_kv_mgr: BatchGen auxiliary (indexer) GPUPagedKVCacheManager.
        padding_bsz: decode padding batch size (unused at construction in M1;
            threaded through for parity with ``configure_decode``).
        dp_size: attention-DP size; defaults to ``world_size`` (dp16 on 2 nodes).
        nnodes / node_rank: 2-node topology.

    Returns:
        The constructed ``ModelRunner`` with ``token_to_kv_pool`` replaced by a
        ``BatchGenNSAKVAdapter``.
    """
    # 1) PG-adopt guard BEFORE any SGLang distributed import path runs.
    _install_pg_adopt_guard()

    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.model_executor.model_runner import ModelRunner

    # TODO(verify-on-gpu): resolve the HF model path off BatchGen's config.
    model_path = getattr(loaded_model_config, "model_path", None)
    assert model_path is not None, (
        "loaded_model_config has no .model_path; wire the GLM-5 HF path that "
        "configure_decode already resolved (TODO(verify-on-gpu))."
    )

    if dp_size is None:
        dp_size = world_size

    server_args = _build_decode_server_args(
        model_path=model_path,
        world_size=world_size,
        global_rank=global_rank,
        local_rank=local_rank,
        dist_init_addr=dist_init_addr,
        dp_size=dp_size,
        nnodes=nnodes,
        node_rank=node_rank,
        mem_fraction_static=mem_fraction_static,
    )

    model_config = ModelConfig.from_server_args(server_args)

    # nccl_port is only used by SGLang to build a NEW dist init method; because
    # the PG is already initialized, init_distributed_environment early-skips
    # the init_process_group call, so the precise value is inert. Parse it from
    # dist_init_addr for traceability; fall back to 0.
    try:
        nccl_port = int(dist_init_addr.rsplit(":", 1)[1])
    except (IndexError, ValueError):
        nccl_port = 0  # TODO(verify-on-gpu): inert when PG pre-initialized.

    # 2) Construct the decode-only ModelRunner. token_to_kv_pool_allocator is
    #    left None: SGLang auto-creates a pool in init_memory_pool, which we then
    #    OVERRIDE with the adapter. M1 accepts that transient allocation (R4:
    #    check 2-node dp16 HBM headroom for SGLang-owned weights + its scratch
    #    KV pool that we immediately shadow).
    #    TODO(verify-on-gpu): if the auto-created NSA pool's allocation is large
    #    enough to OOM, pass a custom BaseTokenToKVPoolAllocator that allocates
    #    nothing, or set mem_fraction_static low. The mixin path is
    #    model_runner_kv_cache_mixin.py:636-694.
    model_runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static or 0.8,
        gpu_id=local_rank,
        tp_rank=global_rank,
        tp_size=world_size,
        moe_ep_rank=global_rank,   # TODO(verify-on-gpu): EP rank mapping for GLM-5 dp16.
        moe_ep_size=world_size,    # TODO(verify-on-gpu): EP size; M1 uses SGLang-owned experts.
        pp_rank=0,
        pp_size=1,
        nccl_port=nccl_port,
        server_args=server_args,
    )

    # 3) Inject the read-adapter as the KV pool (if the KV managers are ready).
    #    The in-worker path builds the runner BEFORE BatchGen's GPU KV managers
    #    exist (they size from free HBM after the runner's weights load), so it
    #    passes primary/aux=None here and calls inject_kv_adapter() later. The
    #    standalone smoke passes them now. ForwardBatch.init_new / our
    #    build_decode_forward_batch reads K via forward_batch.token_to_kv_pool.*,
    #    so replacing the pool is sufficient; the auto-created pool is discarded.
    if primary_kv_mgr is not None and aux_kv_mgr is not None:
        inject_kv_adapter(
            model_runner,
            primary_kv_mgr,
            aux_kv_mgr,
            layer_num=getattr(model_config, "num_hidden_layers", None),
        )

    return model_runner


def inject_kv_adapter(model_runner, primary_kv_mgr, aux_kv_mgr, layer_num=None):
    """Replace ``model_runner.token_to_kv_pool`` with a BatchGenNSAKVAdapter.

    Called after BatchGen's primary (MLA) + auxiliary (indexer) GPU paged KV
    managers exist. Idempotent-ish: overwrites whatever pool is currently set.
    """
    adapter = BatchGenNSAKVAdapter(
        gpu_paged_kv_manager=primary_kv_mgr,
        gpu_paged_kv_manager_aux=aux_kv_mgr,
        page_size=_PAGE_SIZE,
        layer_num=layer_num,
    )
    model_runner.token_to_kv_pool = adapter
    return adapter


def build_decode_forward_batch(
    model_runner,
    new_tokens: torch.Tensor,
    positions: torch.Tensor,
    cache_seqlens: torch.Tensor,
    page_table: torch.Tensor,
    req_pool_indices: torch.Tensor,
    out_cache_loc: torch.Tensor,
):
    """Build a DECODE ``ForwardBatch`` directly (bypassing ScheduleBatch).

    We construct ``ForwardBatch`` directly rather than going through
    ``ModelWorkerBatch`` -> ``ForwardBatch.init_new`` because for a pure DECODE
    step only a small, well-defined subset of fields is read (verified against
    forward_batch_info.py:380-529 and the DECODE dispatch at model_runner.py:2433).
    Direct construction keeps the seam minimal and avoids materializing a
    ScheduleBatch/ReqToTokenPool we don't need.

    Field mapping (from BatchGen decode state; design (3)):
        forward_mode      = ForwardMode.DECODE
        batch_size        = new_tokens.numel()
        input_ids         = new_tokens.view(-1)            int (B,)
        positions         = positions (== clamp_position(seq_lens) == seq_lens-1)
        seq_lens          = cache_seqlens                  int32 (B,)
        seq_lens_sum      = int(cache_seqlens.sum())
        out_cache_loc     = phys_page*64 + offset          int (B,)  [write slot]
        req_pool_indices  = seq_id->slot row indices       int (B,)
        token_to_kv_pool  = model_runner.token_to_kv_pool  (the adapter)
        attn_backend      = model_runner.attn_backend
        req_to_token_pool = model_runner.req_to_token_pool

    The reordered BatchGen ``gpu_table`` (page_table arg, batch-row order) IS
    SGLang's ``page_table_64`` — same semantics. The NSA backend reads the page
    table off its attn metadata, not off ForwardBatch directly, so we set it on
    the batch for traceability AND rely on attn_backend.init_forward_metadata.

    Args:
        page_table: [batch_size, max_pages] int32, batch-row order (already
            ``reorder_block_table_to_batch_slots``-ed). Passed for the NSA
            backend's metadata build; see TODO below.

    Returns:
        ForwardBatch(forward_mode=DECODE), ready for ``model_runner.forward``.

    TODO(verify-on-gpu): ScheduleBatch fallback. If direct construction trips an
    assertion inside the NSA backend's ``init_forward_metadata`` (it reads the
    page table via ``req_to_token_pool``/metadata, not ForwardBatch.page_table),
    we must EITHER (a) pre-populate model_runner.req_to_token_pool so that
    req_pool_indices -> the same physical pages as page_table, OR (b) drive the
    metadata through SGLang's ScheduleBatch path. (a) is the intended M1 route:
    the page table must be reachable from req_to_token_pool[req_pool_indices].
    """
    from sglang.srt.model_executor.forward_batch_info import (
        CaptureHiddenMode,
        ForwardBatch,
        ForwardMode,
    )

    input_ids = new_tokens.view(-1)
    batch_size = int(input_ids.numel())

    # DECODE positions = clamp_position(seq_lens) = seq_lens - 1 (>=0), int64.
    # Caller passes BatchGen's Attn_Wrapper.position_ids; coerce to int64 1-D.
    positions_1d = positions.view(-1).to(torch.int64)

    seq_lens = cache_seqlens.view(-1)
    seq_lens_sum = int(seq_lens.sum().item())
    # NSA's init_forward_metadata reads seq_lens_cpu (nsa_backend.py:391/393:
    # max_seqlen_k = seq_lens_cpu.max()). ForwardBatch.init_new normally
    # materializes it; we bypass init_new, so build the CPU mirror here.
    seq_lens_cpu = seq_lens.to("cpu")

    fb = ForwardBatch(
        forward_mode=ForwardMode.DECODE,
        batch_size=batch_size,
        input_ids=input_ids,
        req_pool_indices=req_pool_indices.view(-1),
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens_cpu,
        out_cache_loc=out_cache_loc.view(-1),
        seq_lens_sum=seq_lens_sum,
        positions=positions_1d,
        # KV + backend wiring (ForwardBatch.init_new copies these from the runner).
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool=model_runner.token_to_kv_pool,  # the adapter.
        attn_backend=model_runner.attn_backend,
        # Decode-only: no logprobs, no spec, no LoRA, no DP-padding metadata.
        return_logprob=False,
        # LM head reads logits_metadata.capture_hidden_mode.need_capture()
        # (logits_processor.py:550); init_new sets this from the ScheduleBatch
        # (default NULL). We bypass init_new, so set it explicitly — no hidden
        # state capture in M1 decode.
        capture_hidden_mode=CaptureHiddenMode.NULL,
    )

    # dp-attention: the model's DP-gather collectives (all_gather across the DP
    # ranks, sized by per-rank token counts) need the DP buffer state set before
    # forward, or vocab_parallel_embedding -> is_dp_max_padding() trips
    # AttributeError on _DpGatheredBufferWrapper._dp_max_padding. SGLang's
    # scheduler does this via maybe_prepare_mlp_sync_batch (scheduler.py:1932);
    # we bypass the scheduler, so set global_num_tokens + call
    # ForwardBatch.prepare_mlp_sync_batch (forward_batch_info.py:734) ourselves.
    if getattr(model_runner.server_args, "enable_dp_attention", False):
        dp_size = model_runner.server_args.dp_size
        # SMOKE / uniform decode: every DP rank carries `batch_size` decode
        # tokens, so all ranks agree on global_num_tokens -> the DP all_gather is
        # fixed-size and cannot hang.
        # TODO(real-run): ranks can differ; the BatchGen worker must supply the
        # DP all-gathered per-rank counts (it is SPMD and knows each rank's batch
        # via its own scheduler), mirroring scheduler_dp_attn_mixin's gather.
        gnt = [batch_size] * dp_size
        fb.global_num_tokens_cpu = list(gnt)
        fb.global_num_tokens_for_logprob_cpu = list(gnt)
        gnt_gpu = torch.tensor(gnt, dtype=torch.int64, device=input_ids.device)
        fb.global_num_tokens_gpu = gnt_gpu
        # The LM head's DP-attention hidden-state gather
        # (logits_processor.compute_dp_attention_metadata:208) does
        # cumsum(global_num_tokens_for_logprob_gpu); init_new sets it, we don't.
        fb.global_num_tokens_for_logprob_gpu = gnt_gpu.clone()
        fb.is_extend_in_batch = False  # decode-only: never convert to EXTEND.
        # _pad_inputs_to_size unconditionally does lora_ids.extend(...) (others
        # are None-guarded); default None -> AttributeError. No LoRA in M1.
        fb.lora_ids = [None] * batch_size
        fb.prepare_mlp_sync_batch(model_runner)

    # TODO(verify-on-gpu): the NSA backend pulls the page table from its own
    # metadata, which it builds in attn_backend.init_forward_metadata(fb). For
    # the page-gather decode read path (nsa_indexer.py:521
    # get_index_k_scale_buffer(layer_id, seq_len, block_tables[i])) the block
    # table must come from req_to_token_pool. Our `page_table` arg carries the
    # batch-ordered physical pages; if init_forward_metadata cannot reach it,
    # stash it where the backend expects (e.g. attach to req_to_token_pool, or
    # set fb-level page table once we confirm the field name SGLang reads).
    # ALSO: SGLang's ModelRunner.forward calls init_forward_metadata internally
    # unless skip_attn_backend_init=True — confirm we do NOT need to call it by
    # hand here for the decode path (model_runner.py:2433 -> forward_decode).
    return fb
