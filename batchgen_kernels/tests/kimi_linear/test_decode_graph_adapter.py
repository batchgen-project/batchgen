"""M5.2 gate: KimiLinearDecodeGraph (Phase-A per-layer decode spans) on a
synthetic 4-layer Kimi-Linear (KDA, MLA, KDA, MLA; layer 0 dense => whole-layer
span, layers 1-3 MoE => attention-span only, MoE eager between replays).

What this exercises (plan M5 item 4/5, task M5.2):
  * bucket selection incl. B that is NOT a bucket (pad-to-bucket) — B=3 -> 4;
  * capture-once / replay-many — a bucket is captured on its first use only;
  * bucket switching across steps (3 -> 8 -> 1 -> ...);
  * static-buffer refresh between replays: the KDA slot buffer keeps its
    address (KDAStateGPUManager's persistent buffer, M5.1) while its contents
    follow the batch, padded rows take the graph's scratch slot (never -1 —
    that would be an fla OOB write before the pool base, plan M5.3); the MLA
    page table is re-copied in batch order each step;
  * compare mode: graph AND eager per span, max|delta| recorded per layer;
  * state evolution: >= 16 decode steps under graphs vs the SAME schedule run
    eagerly through the production wrappers — logits, KDA conv/recurrent pools
    and the paged K cache must match.

Eager reference = the real serving path (KimiLinearKDAWrapper /
KimiLinearAttnWrapper -> serving_modules), reached by flipping the batch-level
`batchgen_debug.kimi_decode_graph_mode` to "eager" on the SAME installed
adapter — so the fallback path is covered too.

Gates: logits/output bf16 <= 1e-2, KDA state <= 2e-2 (plan M5.5); bitwise
agreement is reported when achieved.

Run (GPU): python batchgen_kernels/tests/kimi_linear/test_decode_graph_adapter.py
"""

import sys
import types

import torch

from batchgen.models.moonshotai.kimi_linear.config import KimiLinearConfig
from batchgen.models.moonshotai.kimi_linear.cuda_graph_segments import (
    GRAPH_SCRATCH_SEQ_ID,
    KimiLinearDecodeGraph,
)
from batchgen.models.moonshotai.kimi_linear.model import KimiLinearForCausalLM
from batchgen.models.moonshotai.kimi_linear.serving_modules import (
    kda_decode_serving,
    mla_decoding_nope_with_pagekv,
)
from batchgen.models.moonshotai.kimi_linear.wrappers import (
    KimiLinearAttnWrapper,
    KimiLinearKDAWrapper,
)
from batchgen.models.wrappers import AttnWrapperBase

DEVICE = "cuda:0"
DTYPE = torch.bfloat16

# Synthetic geometry. MLA dims stay at the production values (kv_lora 512 +
# rope 64 = 576, v_head 128) because FlashMLA is fixed to them; only hidden
# size, layer count, vocab and the MoE are shrunk.
HIDDEN = 256
VOCAB = 512
NUM_LAYERS = 4
KDA_LAYERS_1IDX = [1, 3]          # -> layer 0 and 2 are KDA, 1 and 3 are MLA
KDA_HEADS = 4
KDA_HEAD_DIM = 128
CONV_W = 4
MLA_HEADS = 32
NUM_EXPERTS = 8
TOP_K = 2
MOE_INTER = 64

SLOTS = 16                        # KDA state slots (1 reserved as scratch)
PAGE_SIZE = 64                    # FlashMLA page size
NUM_PAGES = 16
MAX_PAGES_PER_SEQ = 2
MAX_KV_SLOTS = 8

BUCKETS = [1, 2, 4, 8]
NUM_SEQS = 8
BASE_CTX = 40                     # tokens already in KV/state after "prefill"
# batch size per step: covers pad-to-bucket (3->4, 5->8, 6->8, 7->8) and
# bucket switching (4, 8, 1, 2, ...). 18 steps >= the 16-step requirement.
BSZ_SCHEDULE = [3, 3, 4, 8, 1, 2, 5, 8, 8, 3, 1, 4, 6, 8, 2, 7, 4, 3]

OUT_TOL = 1e-2
STATE_TOL = 2e-2
COMPARE_TOL = 1e-2

PASS = True


def report(name, ok, detail=""):
    global PASS
    PASS = PASS and ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def check(name, got, ref, tol):
    got32, ref32 = got.float(), ref.float()
    finite = bool(torch.isfinite(got32).all())
    diff = (got32 - ref32).abs().max().item()
    bitwise = torch.equal(got, ref)
    ok = finite and (bitwise or diff <= tol)
    report(
        name, ok,
        f"bitwise={bitwise} max|Δ|={diff:.2e} (tol {tol:.0e})"
        + ("" if finite else " NON-FINITE OUTPUT"),
    )


# ---------------------------------------------------------------------------
#  Fake paged-KV manager (the subset both the graph spans and the eager
#  serving path consume; same page-table semantics as GPUPagedKVManager:
#  graph-stable table storage whose row i IS batch row i).
# ---------------------------------------------------------------------------
class _FakeKVConfig:
    def __init__(self, page_size_tokens):
        self.page_size_tokens = page_size_tokens


class FakePagedKV:
    def __init__(self, num_layers, kv_dim, device, dtype):
        self.device = torch.device(device)
        self.config = _FakeKVConfig(PAGE_SIZE)
        self._k = torch.zeros(
            (num_layers, NUM_PAGES, PAGE_SIZE, 1, kv_dim),
            dtype=dtype, device=self.device,
        )
        self._table = torch.full(
            (MAX_KV_SLOTS, MAX_PAGES_PER_SEQ), -1,
            dtype=torch.int32, device=self.device,
        )
        self._seq_pages = {}
        self._active_rows = 0
        self._order = []

    # -- allocation / page-table order --------------------------------------
    def assign_pages(self, seq_id, pages):
        self._seq_pages[int(seq_id)] = list(pages)

    def set_batch_order(self, seq_ids):
        """Mirror the worker's rebuild: row i holds sequence seq_ids[i]."""
        self._table.fill_(-1)
        for row, seq_id in enumerate(seq_ids):
            pages = self._seq_pages[int(seq_id)]
            self._table[row, : len(pages)] = torch.tensor(
                pages, dtype=torch.int32, device=self.device
            )
        self._active_rows = len(seq_ids)
        self._order = list(seq_ids)

    # -- API used by the graph spans ---------------------------------------
    def ensure_cuda_graph_page_table(self, seq_ids):
        if list(seq_ids) != self._order:
            self.set_batch_order(seq_ids)
        return self._table

    def get_cuda_graph_page_table(self):
        return self._table

    def get_cuda_graph_page_table_storage(self):
        return self._table

    def get_kv_tensors(self):
        return self._k, None

    def get_layer_kv_with_page_table(self, layer_idx):
        return self._k[layer_idx], None, self._table[: self._active_rows]

    # -- API used by the eager serving path --------------------------------
    def update_layer_decode_new_token(self, k_tensor, v_tensor, sequence_lengths,
                                      layer_idx, batch_slice=None,
                                      slot_indices=None):
        positions = sequence_lengths.tolist()
        for row, pos in enumerate(positions):
            page = int(self._table[row, pos // PAGE_SIZE].item())
            self._k[layer_idx, page, pos % PAGE_SIZE, 0] = k_tensor[row, 0, 0]

    def snapshot(self):
        return self._k.clone()

    def restore(self, snap):
        self._k.copy_(snap)


# ---------------------------------------------------------------------------
#  Model + serving wiring (mirrors PSM._config_{attn,kda}_modules)
# ---------------------------------------------------------------------------
def build_config():
    return KimiLinearConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=4 * HIDDEN,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=MLA_HEADS,
        num_key_value_heads=MLA_HEADS,
        kv_lora_rank=512,
        q_lora_rank=None,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        linear_attn_config={
            "kda_layers": KDA_LAYERS_1IDX,
            "num_heads": KDA_HEADS,
            "head_dim": KDA_HEAD_DIM,
            "short_conv_kernel_size": CONV_W,
            "use_full_rank_gate": False,
        },
        n_routed_experts=NUM_EXPERTS,
        num_local_experts=NUM_EXPERTS,
        n_shared_experts=1,
        num_experts_per_tok=TOP_K,
        first_k_dense_replace=1,
        moe_intermediate_size=MOE_INTER,
        n_group=1,
        topk_group=1,
        max_position_embeddings=4096,
        pad_token_id=0,
    )


def build_model(cfg):
    torch.manual_seed(7)
    model = KimiLinearForCausalLM(cfg)
    for name, param in model.named_parameters():
        # RMSNorm-style weights around 1.0 so activations (and therefore the
        # compared logits/state) keep unit scale — an absolute 1e-2 gate is
        # only meaningful if the quantities are O(1).
        if "norm" in name:
            param.data.normal_(1.0, 0.02)
        else:
            param.data.normal_(0.0, 0.05)
        param.requires_grad_(False)
    model = model.to(DEVICE).to(DTYPE)
    # fla wants fp32 for these (the real flow loads them per-tensor).
    for layer_idx in range(cfg.num_hidden_layers):
        if not cfg.is_kda_layer(layer_idx):
            continue
        kda = model.model.layers[layer_idx].self_attn
        kda.A_log.data = kda.A_log.data.float()
        kda.dt_bias.data = kda.dt_bias.data.float()
    model.eval()
    return model


def install_serving_wrappers(model, cfg):
    """PSM method injection + wrapper install (Parallel_Strategy_Manager.py)."""
    for layer_idx in range(cfg.num_hidden_layers):
        layer = model.model.layers[layer_idx]
        attn = layer.self_attn
        if cfg.is_kda_layer(layer_idx):
            attn.kda_decode_serving = types.MethodType(kda_decode_serving, attn)
            layer.self_attn = KimiLinearKDAWrapper(
                attn, layer_idx, None, None, None, persistent=True
            )
        else:
            attn.mla_decoding_nope_with_pagekv = types.MethodType(
                mla_decoding_nope_with_pagekv, attn
            )
            layer.self_attn = KimiLinearAttnWrapper(
                attn, layer_idx, None, None, None, persistent=True
            )


# ---------------------------------------------------------------------------
#  Decode driver
# ---------------------------------------------------------------------------
def decode_step(model, kv, seq_ids, tokens, contexts):
    """One decode step with the worker's per-step wrapper bindings."""
    AttnWrapperBase.phase = "decode"
    AttnWrapperBase.attention_mask = None
    AttnWrapperBase.cur_batch = list(seq_ids)
    AttnWrapperBase.gpu_paged_kv_manager = kv
    cache_seqlens = torch.tensor(
        [contexts[s] + 1 for s in seq_ids], dtype=torch.int32, device=DEVICE
    )
    position_ids = torch.tensor(
        [[contexts[s]] for s in seq_ids], dtype=torch.int64, device=DEVICE
    )
    AttnWrapperBase.cache_seqlens = cache_seqlens
    AttnWrapperBase.position_ids = position_ids
    AttnWrapperBase.max_seqlen = int(cache_seqlens.max().item())
    logits = model(input_ids=tokens, position_ids=position_ids)
    for s in seq_ids:
        contexts[s] += 1
    return logits


def run_schedule(model, kv, mode, live, token_plan, base_contexts):
    """Run BSZ_SCHEDULE decode steps in `mode`; return per-step logits."""
    AttnWrapperBase.batchgen_debug = {"kimi_decode_graph_mode": mode}
    contexts = dict(base_contexts)
    outs = []
    for step, bsz in enumerate(BSZ_SCHEDULE):
        seq_ids = live[:bsz]
        kv.set_batch_order(seq_ids)
        logits = decode_step(model, kv, seq_ids, token_plan[step][:bsz], contexts)
        outs.append(logits.float().clone())
    torch.cuda.synchronize()
    AttnWrapperBase.batchgen_debug = None
    return outs


def pool_snapshot(mgr):
    conv_q, conv_k, conv_v = mgr.get_conv_tensors()
    return (
        conv_q.clone(), conv_k.clone(), conv_v.clone(),
        mgr.get_recurrent_tensors().clone(),
    )


def pool_restore(mgr, snap):
    conv_q, conv_k, conv_v = mgr.get_conv_tensors()
    conv_q.copy_(snap[0])
    conv_k.copy_(snap[1])
    conv_v.copy_(snap[2])
    mgr.get_recurrent_tensors().copy_(snap[3])


def main():
    if not torch.cuda.is_available():
        print("CUDA required")
        sys.exit(1)
    torch.set_grad_enabled(False)
    torch.manual_seed(11)

    cfg = build_config()
    model = build_model(cfg)
    install_serving_wrappers(model, cfg)

    kda_layers = [i for i in range(NUM_LAYERS) if cfg.is_kda_layer(i)]
    mla_layers = [i for i in range(NUM_LAYERS) if not cfg.is_kda_layer(i)]
    report("layout: 2 KDA + 2 MLA, layer 0 dense",
           kda_layers == [0, 2] and mla_layers == [1, 3]
           and hasattr(model.model.layers[0], "mlp")
           and hasattr(model.model.layers[1], "block_sparse_moe"))

    KimiLinearKDAWrapper.reset()
    KimiLinearKDAWrapper.init_state_pools(
        kda_layers, num_slots=SLOTS, num_heads=KDA_HEADS,
        head_dim=KDA_HEAD_DIM, conv_width=CONV_W,
        proj_size=KDA_HEADS * KDA_HEAD_DIM, device=DEVICE, dtype=DTYPE,
    )
    mgr = KimiLinearKDAWrapper.state_manager

    kv = FakePagedKV(NUM_LAYERS, cfg.compressed_kv_dim, DEVICE, DTYPE)
    kv_appends = []
    AttnWrapperBase.kv_append_callback = (
        lambda layer_idx, k, v: kv_appends.append((layer_idx, k))
    )
    AttnWrapperBase.kv_append_callback_aux = None

    # ---- simulate a completed prefill: slots, pages, random state ---------
    live = list(range(1, NUM_SEQS + 1))
    for i, seq_id in enumerate(live):
        KimiLinearKDAWrapper.slot_manager.alloc(seq_id)
        kv.assign_pages(seq_id, [2 * i, 2 * i + 1])
    gen = torch.Generator(device=DEVICE).manual_seed(23)
    for pool in (*mgr.get_conv_tensors(), mgr.get_recurrent_tensors()):
        pool.copy_(
            (torch.randn(pool.shape, generator=gen, device=DEVICE,
                         dtype=torch.float32) * 0.1).to(pool.dtype)
        )
    kv._k.copy_(
        (torch.randn(kv._k.shape, generator=gen, device=DEVICE,
                     dtype=torch.float32) * 0.1).to(DTYPE)
    )
    base_contexts = {s: BASE_CTX for s in live}
    token_plan = [
        torch.randint(0, VOCAB, (NUM_SEQS, 1), generator=gen, device=DEVICE)
        for _ in BSZ_SCHEDULE
    ]

    kda_snap = pool_snapshot(mgr)
    kv_snap = kv.snapshot()

    # ---- adapter ----------------------------------------------------------
    adapter = KimiLinearDecodeGraph(
        model, cfg, device=torch.device(DEVICE), buckets=BUCKETS,
        mode="graph", compare_every=4, rank=0,
    )
    adapter.install()

    # ---- pass 1: graphs ---------------------------------------------------
    graph_outs = run_schedule(model, kv, "graph", live, token_plan,
                              base_contexts)
    graph_pools = pool_snapshot(mgr)
    graph_kv = kv.snapshot()
    graph_appends = len(kv_appends)
    graph_steps = adapter.step
    report("every scheduled step took the graph path (no eager fallback)",
           graph_steps == len(BSZ_SCHEDULE),
           f"{graph_steps}/{len(BSZ_SCHEDULE)} steps replayed")

    # bucket selection / capture-once bookkeeping
    expected_buckets = sorted({adapter.bucketing.get_padded_size(b)
                               for b in BSZ_SCHEDULE})
    stats = adapter.get_capture_stats()
    captured = sorted(stats["kimi_capture"].keys())
    report("bucket selection: pad-to-bucket",
           adapter.bucketing.get_padded_size(3) == 4
           and adapter.bucketing.get_padded_size(5) == 8
           and adapter.bucketing.get_padded_size(8) == 8,
           f"used buckets {expected_buckets}")
    report("capture-once per bucket (replay-many)",
           captured == expected_buckets
           and all(sorted(v) == captured
                   for v in stats["graphs_per_segment"].values()),
           f"captured={captured} segments={len(stats['graphs_per_segment'])}")
    report("every layer captured as its own span",
           len(stats["graphs_per_segment"]) == NUM_LAYERS)
    report("KV offload callback fired per MLA layer per step",
           graph_appends == len(BSZ_SCHEDULE) * len(mla_layers),
           f"{graph_appends} appends")

    # slot-buffer contract: one persistent buffer, refreshed in place, padded
    # rows on the scratch slot (never -1).
    scratch = mgr.get_sequence_state_item(GRAPH_SCRATCH_SEQ_ID)
    slot_ptrs = {s.kda_slots.data_ptr() for s in adapter._statics.values()}
    last_bsz = BSZ_SCHEDULE[-1]
    last_bucket = adapter.bucketing.get_padded_size(last_bsz)
    slots_now = mgr.get_prepared_state_slots()[:last_bucket].tolist()
    expected_slots = [mgr.get_sequence_state_item(s) for s in live[:last_bsz]]
    expected_slots += [scratch] * (last_bucket - last_bsz)
    report("KDA slot buffer: single fixed address, bound not reallocated",
           len(slot_ptrs) == 1
           and slot_ptrs.pop() == mgr._prepared_state_slots.data_ptr())
    report("KDA slot buffer refreshed in place; padding -> scratch slot",
           slots_now == expected_slots and scratch not in expected_slots[:last_bsz],
           f"slots={slots_now} scratch={scratch}")

    # page-table refresh: static rows follow the batch order of the last step
    statics = adapter._statics[last_bucket]
    kv.set_batch_order(live[:last_bsz])
    report("MLA page table refreshed in batch order between replays",
           torch.equal(statics.page_table[:last_bsz],
                       kv.get_cuda_graph_page_table()[:last_bsz]),
           f"pad rows zeroed={bool((statics.page_table[last_bsz:] == 0).all())}")

    # ---- pass 2: eager, identical schedule from the same initial state ----
    pool_restore(mgr, kda_snap)
    kv.restore(kv_snap)
    kv_appends.clear()
    eager_outs = run_schedule(model, kv, "eager", live, token_plan,
                              base_contexts)
    eager_pools = pool_snapshot(mgr)
    eager_kv = kv.snapshot()
    report("eager mode replayed nothing (adapter step counter frozen)",
           adapter.step == graph_steps)
    report("eager fallback took the wrapper path (same KV callback count)",
           len(kv_appends) == graph_appends)

    for step, (got, ref) in enumerate(zip(graph_outs, eager_outs)):
        if step % 6 == 0 or step == len(graph_outs) - 1:
            check(f"step {step:02d} (B={BSZ_SCHEDULE[step]}) logits vs eager",
                  got, ref, OUT_TOL)
    worst = max((g - e).abs().max().item()
                for g, e in zip(graph_outs, eager_outs))
    report(f"all {len(BSZ_SCHEDULE)} steps within logits tol",
           worst <= OUT_TOL, f"worst max|Δ|={worst:.2e}")

    # State lives per slot. The graph path pads to the bucket and parks the
    # padded rows on the scratch slot, which the (unpadded) eager path never
    # touches — so the scratch row legitimately differs and is NOT part of the
    # correctness claim. Compare the LIVE slots, and report the per-slot
    # breakdown so a real divergence can never hide behind that exclusion.
    live_slots = sorted({s for s in expected_slots[:last_bsz]}
                        | {mgr.get_sequence_state_item(sid) for sid in live})
    for name, got, ref in (
        ("conv_q", graph_pools[0], eager_pools[0]),
        ("conv_k", graph_pools[1], eager_pools[1]),
        ("conv_v", graph_pools[2], eager_pools[2]),
        ("recurrent", graph_pools[3], eager_pools[3]),
    ):
        per_slot = {
            s: (got[:, s].float() - ref[:, s].float()).abs().max().item()
            for s in range(got.shape[1])
        }
        offenders = {s: d for s, d in per_slot.items()
                     if d > STATE_TOL and s != scratch}
        report(f"KDA {name}: only the scratch slot may differ",
               not offenders,
               f"scratch={scratch} d_scratch={per_slot[scratch]:.2e} "
               f"live_slots={live_slots} offenders={offenders}")
        check(f"KDA {name} state evolution over {len(BSZ_SCHEDULE)} steps "
              f"(live slots)",
              got[:, live_slots], ref[:, live_slots], STATE_TOL)
    check("paged K cache after full schedule", graph_kv, eager_kv, OUT_TOL)

    # ---- pass 3: compare mode --------------------------------------------
    pool_restore(mgr, kda_snap)
    kv.restore(kv_snap)
    kv_appends.clear()
    adapter.set_mode("compare")
    adapter.step = 0
    contexts = dict(base_contexts)
    AttnWrapperBase.batchgen_debug = {"kimi_decode_graph_mode": "compare"}
    for step, bsz in enumerate(BSZ_SCHEDULE[:8]):
        seq_ids = live[:bsz]
        kv.set_batch_order(seq_ids)
        decode_step(model, kv, seq_ids, token_plan[step][:bsz], contexts)
    torch.cuda.synchronize()
    AttnWrapperBase.batchgen_debug = None

    report("compare mode replayed the already-captured buckets (no recapture)",
           sorted(adapter.get_capture_stats()["kimi_capture"].keys()) == captured)

    deltas = adapter.last_compare
    report("compare mode reported a delta for every span",
           sorted(deltas.keys()) == list(range(NUM_LAYERS)),
           f"layers={sorted(deltas.keys())}")
    worst_span = max(deltas.values()) if deltas else float("inf")
    report("compare mode graph-vs-eager deltas within tol",
           worst_span <= COMPARE_TOL,
           f"worst span max|Δ|={worst_span:.2e} (tol {COMPARE_TOL:.0e}) "
           f"per-layer={ {k: f'{v:.2e}' for k, v in sorted(deltas.items())} }")

    # -- regression: a K-cache geometry change must drop + re-capture --------
    # End-to-end companion to the signature checks below. The old gate never
    # reached this condition: it only ever ran inside a single batch job, so
    # the KV manager never changed shape under a live graph and the bug sailed
    # through 162/162 spans at 0.000e+00 while halving MMLU accuracy. Here the
    # cache is regrown mid-run with the OLD tensor deliberately kept alive, so
    # a graph that failed to drop would replay against still-valid but stale
    # memory — wrong logits, no crash, exactly the production failure.
    pool_restore(mgr, kda_snap)
    kv.restore(kv_snap)
    old_k = kv._k                                   # keep alive on purpose
    grown = torch.zeros(
        (NUM_LAYERS, NUM_PAGES + 3, PAGE_SIZE, 1, old_k.shape[-1]),
        dtype=old_k.dtype, device=old_k.device,
    )
    grown[:, :NUM_PAGES].copy_(old_k)               # same KV contents, new geometry
    kv._k = grown
    report("test setup: per-layer K slice actually relocated",
           (grown[1].data_ptr() - grown.data_ptr())
           != (old_k[1].data_ptr() - old_k.data_ptr()),
           f"layer-1 offset {old_k[1].data_ptr() - old_k.data_ptr()} -> "
           f"{grown[1].data_ptr() - grown.data_ptr()}")
    report("adapter sees the new geometry in its signature",
           tuple(grown.shape) in adapter._signature(kv))

    geo_snap = kv.snapshot()
    geo_graph = run_schedule(model, kv, "graph", live, token_plan, base_contexts)
    report("graphs were re-captured after the geometry change",
           len(adapter._captured) > 0,
           f"captured={sorted(adapter._captured)}")

    pool_restore(mgr, kda_snap)
    kv.restore(geo_snap)
    geo_eager = run_schedule(model, kv, "eager", live, token_plan, base_contexts)
    worst_geo = max((g - e).abs().max().item()
                    for g, e in zip(geo_graph, geo_eager))
    report("logits still match eager after a K-cache geometry change",
           worst_geo <= OUT_TOL,
           f"worst max|Δ|={worst_geo:.2e} (tol {OUT_TOL:.0e})")

    # -- regression: K-cache geometry must be part of the capture signature ---
    # The MLA spans bake `_k_cache[layer]`, whose base address is
    # `data_ptr + layer * num_pages * page_stride`. The worker re-creates the
    # KV manager per batch job and sizes it from that rank's free HBM, so
    # num_pages varies between jobs. A re-init that reuses the same base
    # address with a different page count leaves data_ptr unchanged while every
    # slice above layer 0 moves -- graphs are then never dropped and replay
    # against relocated slices. That silently corrupted MLA output (KDA stayed
    # bitwise clean) and halved MMLU accuracy before it was caught. Task #12.
    sig = adapter._signature(kv)
    report("capture signature carries the K-cache shape",
           tuple(kv._k.shape) in sig,
           f"shape={tuple(kv._k.shape)} sig={sig}")
    report("capture signature carries the K-cache stride",
           tuple(kv._k.stride()) in sig,
           f"stride={tuple(kv._k.stride())} sig={sig}")

    # Proves the above is not a vacuous check: the per-layer slice really does
    # move when only num_pages changes, so a pointer-only signature is blind.
    kv_dim = kv._k.shape[-1]
    k_small = torch.zeros((NUM_LAYERS, NUM_PAGES, PAGE_SIZE, 1, kv_dim),
                          dtype=DTYPE, device=torch.device(DEVICE))
    k_big = torch.zeros((NUM_LAYERS, NUM_PAGES + 1, PAGE_SIZE, 1, kv_dim),
                        dtype=DTYPE, device=torch.device(DEVICE))
    off_small = k_small[1].data_ptr() - k_small.data_ptr()
    off_big = k_big[1].data_ptr() - k_big.data_ptr()
    report("per-layer K slice offset depends on num_pages",
           off_small != off_big,
           f"layer-1 offset {off_small} (num_pages={NUM_PAGES}) vs "
           f"{off_big} (num_pages={NUM_PAGES + 1})")

    adapter.release()
    report("release() restored the eager forwards",
           not adapter._installed and not adapter._captured)
    KimiLinearKDAWrapper.reset()

    print("\n" + ("ALL CHECKS PASSED" if PASS else "SOME CHECKS FAILED"))
    sys.exit(0 if PASS else 1)


if __name__ == "__main__":
    main()
