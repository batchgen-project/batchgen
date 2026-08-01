"""M5 capture smoke: ONE KDA decode step (3x causal_conv1d_update + fla
fused_recurrent_kda_fwd) captured in a torch.cuda.CUDAGraph via
serving_modules.kda_decode_serving on a synthetic single layer, B in {1, 8}.

The step runs over the M5.1-unified pools: a KDAStateGPUManager owns the
fixed-address conv/recurrent pools and the persistent slot-index buffer;
the captured region reads the wrapper's KDALayerState views of them. Between
replays only buffer CONTENTS change (eager, same stream, ordered before
replay): the hidden-input buffer, the slot buffer (refreshed through
manager.prepare_decode_step) and — on simulated admissions — pool slots
zeroed by the manager's F4 zero-on-alloc.

Known capture hazards this harness watches (EXECUTION_PLAN §6 item 10):
  - fla Triton autotune/JIT fires on the FIRST call per shape: warm up
    BEFORE capture (3 iterations on a side stream, same B as the capture).
    A cold shape at capture time compiles host-side inside capture ->
    capture failure or a silently skewed graph.
  - No host syncs inside the captured region: kda_decode_serving builds
    cu_seqlens with a device-side torch.arange and has no .item()/.cpu()/
    value-dependent host logic. If fla ever adds one, capture aborts here —
    that is this smoke's tripwire function.
  - Static addresses only: pools, slot buffer and the hidden input keep
    their data_ptr; replays bake the addresses in.
  - M5.3 padding rows are NOT covered here (full batches only). conv path
    will pass kernel pad_slot_id=-1; the fla path needs the scratch-slot
    design (-1 through ssm_state_indices is an OOB write BEFORE the pool
    base, invisible to pool comparison — 64-step gate runs once under
    compute-sanitizer, plan M5.3/M5.5).

Gate (plan M5.5): 32 replays with varying slot contents vs eager on the
same inputs — outputs bf16 <= 1e-2, conv/recurrent state <= 2e-2; bitwise
match is reported when achieved.

Run (GPU): python batchgen_kernels/tests/kimi_linear/test_kda_segment_capture.py
"""

import sys
from itertools import count

import torch
import torch.nn as nn

from batchgen.models.moonshotai.kimi_linear.serving_modules import (
    kda_decode_serving,
)
from batchgen.models.moonshotai.kimi_linear.wrappers import (
    KDALayerState,
    KimiLinearKDAWrapper,
)

DEVICE = "cuda"
DTYPE = torch.bfloat16

# Synthetic single-layer geometry (head_dim matches the real model's 128)
HIDDEN = 512
HEADS = 4
HEAD_DIM = 128
PROJ = HEADS * HEAD_DIM
W = 4
SLOTS = 24
N_REPLAYS = 32
ADMIT_STEPS = (10, 21)  # free 1 seq + admit a new one (slot recycle + zero)

OUT_TOL = 1e-2    # graph-vs-eager logits/output gate (plan M5.5)
STATE_TOL = 2e-2  # KDA state gate (plan M5.5)

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


class SyntheticKDALayer(nn.Module):
    """Duck-typed stand-in exposing exactly the attributes
    kda_decode_serving consumes (mirrors KimiKDAAttention, model.py:511-563;
    low-rank gate path, no conv bias — the Kimi-Linear configuration)."""

    def __init__(self, hidden, heads, head_dim, conv_width):
        super().__init__()
        proj = heads * head_dim
        self.num_heads = heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden, proj, bias=False)
        self.k_proj = nn.Linear(hidden, proj, bias=False)
        self.v_proj = nn.Linear(hidden, proj, bias=False)
        self.f_a_proj = nn.Linear(hidden, head_dim, bias=False)
        self.f_b_proj = nn.Linear(head_dim, proj, bias=False)
        self.b_proj = nn.Linear(hidden, heads, bias=False)
        self.use_full_rank_gate = False
        self.g_a_proj = nn.Linear(hidden, head_dim, bias=False)
        self.g_b_proj = nn.Linear(head_dim, proj, bias=False)
        # depthwise conv; .weight is (proj, 1, W) like fla ShortConvolution
        self.q_conv1d = nn.Conv1d(proj, proj, conv_width, groups=proj,
                                  bias=False)
        self.k_conv1d = nn.Conv1d(proj, proj, conv_width, groups=proj,
                                  bias=False)
        self.v_conv1d = nn.Conv1d(proj, proj, conv_width, groups=proj,
                                  bias=False)
        self.A_log = nn.Parameter(
            torch.log(torch.empty(heads, dtype=torch.float32).uniform_(1, 16))
        )
        self.dt_bias = nn.Parameter(
            torch.randn(proj, dtype=torch.float32) * 0.1
        )
        self.gate_lower_bound = None
        from fla.modules import FusedRMSNormGated
        self.o_norm = FusedRMSNormGated(head_dim, eps=1e-6,
                                        activation="sigmoid")
        self.o_proj = nn.Linear(proj, hidden, bias=False)


def build_layer():
    layer = SyntheticKDALayer(HIDDEN, HEADS, HEAD_DIM, W).to(DEVICE).to(DTYPE)
    # A_log / dt_bias stay fp32 (the real flow loads them per-tensor;
    # Module.to(dtype) would downcast them)
    layer.A_log.data = layer.A_log.data.float()
    layer.dt_bias.data = layer.dt_bias.data.float()
    return layer


def randn_like_pool(pool, gen):
    return (
        torch.randn(pool.shape, generator=gen, device=DEVICE,
                    dtype=torch.float32) * 0.1
    ).to(pool.dtype)


def run_case(batch_size):
    print(f"\n=== KDA decode segment capture, B={batch_size} ===")
    torch.manual_seed(1000 + batch_size)
    layer = build_layer()

    # unified pools through the wrapper entry point (the PSM call)
    KimiLinearKDAWrapper.reset()
    KimiLinearKDAWrapper.init_state_pools(
        [0], num_slots=SLOTS, num_heads=HEADS, head_dim=HEAD_DIM,
        conv_width=W, proj_size=PROJ, device=DEVICE, dtype=DTYPE,
    )
    mgr = KimiLinearKDAWrapper.state_manager
    slot_mgr = KimiLinearKDAWrapper.slot_manager
    g_state = KimiLinearKDAWrapper.layer_pools[0]

    seq_counter = count(1)
    live = [next(seq_counter) for _ in range(batch_size)]
    for s in live:
        slot_mgr.alloc(s)

    # static input + slot-buffer binding (fixed addresses for capture)
    static_hidden = torch.zeros(batch_size, 1, HIDDEN, dtype=DTYPE,
                                device=DEVICE)
    g_state.set_decode_batch(live)
    slot_buffer_ptr = g_state.cur_decode_slots.data_ptr()

    # ── warm up BEFORE capture (fla Triton autotune/JIT, hazard #1) ──────────
    warm_stream = torch.cuda.Stream()
    warm_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm_stream):
        for _ in range(3):
            kda_decode_serving(layer, static_hidden, g_state)
    torch.cuda.current_stream().wait_stream(warm_stream)
    torch.cuda.synchronize()

    # ── seed varying slot contents (post-warmup, pre-capture) ────────────────
    gen = torch.Generator(device=DEVICE).manual_seed(42 + batch_size)
    for pool in (g_state.conv_q, g_state.conv_k, g_state.conv_v,
                 g_state.recurrent_pool):
        pool.copy_(randn_like_pool(pool, gen))

    # eager mirror: plain tensors, identical initial contents
    e_conv_q = g_state.conv_q.clone()
    e_conv_k = g_state.conv_k.clone()
    e_conv_v = g_state.conv_v.clone()
    e_recurrent = g_state.recurrent_pool.clone()
    e_state = KDALayerState(None, e_conv_q, e_conv_k, e_conv_v, e_recurrent)

    xs = [
        (torch.randn(batch_size, 1, HIDDEN, generator=gen, device=DEVICE,
                     dtype=torch.float32) * 0.1).to(DTYPE)
        for _ in range(N_REPLAYS)
    ]

    # ── capture (records only; no kernels execute, pools untouched) ──────────
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_static = kda_decode_serving(layer, static_hidden, g_state)

    # ── 32 replays with varying inputs, slot indices and slot contents ───────
    graph_outs = []
    slots_per_step = []
    zero_events = {}  # step -> recycled slot (mirrored on the eager side)
    for step in range(N_REPLAYS):
        if step in ADMIT_STEPS:
            victim = live[0]
            recycled = mgr.get_sequence_state_item(victim)
            KimiLinearKDAWrapper.free_sequences([victim])
            new_seq = next(seq_counter)
            got = slot_mgr.alloc(new_seq)  # F4 zero-on-alloc (eager, pre-replay)
            assert got == recycled, "LIFO recycle assumption broken"
            live = live[1:] + [new_seq]
            zero_events[step] = recycled
        # refresh the STATIC slot buffer in place (eager H2D on the capture
        # stream, ordered before replay — plan M5.4 outside-graph contract)
        mgr.prepare_decode_step(live)
        slots_per_step.append(
            torch.tensor([mgr.get_sequence_state_item(s) for s in live],
                         dtype=torch.int32, device=DEVICE)
        )
        static_hidden.copy_(xs[step])
        graph.replay()
        graph_outs.append(out_static.clone())
    torch.cuda.synchronize()

    report(
        f"B={batch_size} slot buffer address stable across replays",
        g_state.cur_decode_slots.data_ptr() == slot_buffer_ptr
        and mgr._prepared_state_slots.data_ptr() == slot_buffer_ptr,
    )

    g_final = (
        g_state.conv_q.clone(), g_state.conv_k.clone(),
        g_state.conv_v.clone(), g_state.recurrent_pool.clone(),
    )

    # ── eager reference: same inputs, slot schedule and zero events ──────────
    eager_outs = []
    for step in range(N_REPLAYS):
        if step in zero_events:
            slot = zero_events[step]
            for pool in (e_conv_q, e_conv_k, e_conv_v, e_recurrent):
                pool[slot].zero_()
        e_state.cur_decode_slots = slots_per_step[step]
        eager_outs.append(kda_decode_serving(layer, xs[step], e_state))
    torch.cuda.synchronize()

    check(
        f"B={batch_size} {N_REPLAYS} replay outputs vs eager",
        torch.stack(graph_outs), torch.stack(eager_outs), OUT_TOL,
    )
    for name, got, ref in (
        ("conv_q", g_final[0], e_conv_q),
        ("conv_k", g_final[1], e_conv_k),
        ("conv_v", g_final[2], e_conv_v),
        ("recurrent", g_final[3], e_recurrent),
    ):
        check(
            f"B={batch_size} final {name} state evolution vs eager",
            got, ref, STATE_TOL,
        )

    KimiLinearKDAWrapper.reset()


def main():
    if not torch.cuda.is_available():
        print("CUDA required")
        sys.exit(1)
    torch.set_grad_enabled(False)

    for batch_size in (1, 8):
        run_case(batch_size)

    print("\n" + ("ALL CHECKS PASSED" if PASS else "SOME CHECKS FAILED"))
    sys.exit(0 if PASS else 1)


if __name__ == "__main__":
    main()
