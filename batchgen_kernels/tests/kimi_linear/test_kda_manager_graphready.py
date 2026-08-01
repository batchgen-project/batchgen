"""M5.1 graph-readiness unit test: unified KDA state pools (manager-owned).

The KDAStateGPUManager is the canonical, CUDA-graph-ready home of the KDA
state: the recurrent pool (L, slots, H, K, K) fp32, the three conv pools
(L, slots, dim, W-1) bf16 and the persistent decode slot-index buffer are
each allocated ONCE with a fixed address. KimiLinearKDAWrapper.
init_state_pools hands per-layer VIEWS of these tensors to the serving
path, and slot alloc/free/zero-on-alloc (the F4 fix) is delegated to the
manager through the KDASlotManager facade.

Checks:
  1. address stability: every pool + the prepared-slot buffer keeps its
     data_ptr across 64 alloc/free/decode-sim cycles (graph replays bake
     the addresses in).
  2. aliasing: wrapper per-layer pools share storage with the manager
     tensors (same data_ptr; writes visible both ways).
  3. unified zero-on-alloc (F4): a recycled slot is zeroed across every
     layer's conv+recurrent pool on FRESH alloc; an idempotent re-alloc of
     a live sequence does NOT wipe its state.
  4. conv contract: the per-layer conv view is a contiguous
     (slots, dim, W-1) 3-D tensor and causal_conv1d_update consumes it
     directly — the in-place state roll lands in the manager's memory
     (no copy).

Run (GPU): python batchgen_kernels/tests/kimi_linear/test_kda_manager_graphready.py
"""

import sys
from itertools import count

import torch

from batchgen.models.moonshotai.kimi_linear.wrappers import (
    KimiLinearKDAWrapper,
)
from batchgen_kernels.conv1d import causal_conv1d_update

DEVICE = "cuda"
DTYPE = torch.bfloat16

# Small synthetic geometry (contract test, not a numerics test)
LAYER_IDS = [1, 4, 7]  # global layer indices, physical 0..2
SLOTS = 16
HEADS = 4
HEAD_DIM = 64
DIM = HEADS * HEAD_DIM
W = 4  # kernel width; conv state width = W - 1

PASS = True


def report(name, ok, detail=""):
    global PASS
    PASS = PASS and ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def slot_is_zero(state, slot):
    return (
        bool((state.conv_q[slot] == 0).all())
        and bool((state.conv_k[slot] == 0).all())
        and bool((state.conv_v[slot] == 0).all())
        and bool((state.recurrent_pool[slot] == 0).all())
    )


def main():
    if not torch.cuda.is_available():
        print("CUDA required")
        sys.exit(1)

    KimiLinearKDAWrapper.reset()
    KimiLinearKDAWrapper.init_state_pools(
        LAYER_IDS, num_slots=SLOTS, num_heads=HEADS, head_dim=HEAD_DIM,
        conv_width=W, proj_size=DIM, device=DEVICE, dtype=DTYPE,
    )
    mgr = KimiLinearKDAWrapper.state_manager
    slot_mgr = KimiLinearKDAWrapper.slot_manager
    pools = KimiLinearKDAWrapper.layer_pools

    conv_q, conv_k, conv_v = mgr.get_conv_tensors()
    recurrent = mgr.get_recurrent_tensors()

    # ── check 2: wrapper pools alias manager memory ──────────────────────────
    ok = True
    for physical, gidx in enumerate(LAYER_IDS):
        st = pools[gidx]
        ok = ok and st.conv_q.data_ptr() == conv_q[physical].data_ptr()
        ok = ok and st.conv_k.data_ptr() == conv_k[physical].data_ptr()
        ok = ok and st.conv_v.data_ptr() == conv_v[physical].data_ptr()
        ok = ok and (
            st.recurrent_pool.data_ptr() == recurrent[physical].data_ptr()
        )
        ok = ok and st.conv_q.shape == (SLOTS, DIM, W - 1)
        ok = ok and st.recurrent_pool.shape == (SLOTS, HEADS, HEAD_DIM,
                                                HEAD_DIM)
    report("wrapper pools are views of manager tensors", ok)

    st0 = pools[LAYER_IDS[0]]
    st0.conv_q[3].fill_(2.0)
    ok = bool((conv_q[0, 3] == 2.0).all())
    st0.conv_q[3].zero_()
    report("write through wrapper view visible in manager tensor", ok)

    # ── check 1: data_ptr stable across 64 alloc/free/decode-sim cycles ─────
    slot_buffer_ptr = mgr._prepared_state_slots.data_ptr()
    base_ptrs = (
        recurrent.data_ptr(), conv_q.data_ptr(), conv_k.data_ptr(),
        conv_v.data_ptr(),
    )
    seq_ids = count(100)
    live = []
    stable = True
    for _cycle in range(64):
        newly = [next(seq_ids) for _ in range(4)]
        for s in newly:
            slot_mgr.alloc(s)
        live.extend(newly)
        # decode-sim: every layer stages the batch through the manager's
        # persistent slot buffer (the wrapper decode path)
        for gidx in LAYER_IDS:
            pools[gidx].set_decode_batch(live)
        view = pools[LAYER_IDS[-1]].cur_decode_slots
        stable = stable and view.data_ptr() == slot_buffer_ptr
        stable = stable and view.numel() == len(live)
        # free all but the two most recent (bounded live set, heavy recycle)
        KimiLinearKDAWrapper.free_sequences(live[:-2])
        live = live[-2:]
        stable = stable and base_ptrs == (
            recurrent.data_ptr(), conv_q.data_ptr(), conv_k.data_ptr(),
            conv_v.data_ptr(),
        )
        stable = stable and (
            mgr._prepared_state_slots.data_ptr() == slot_buffer_ptr
        )
    report("data_ptr stable across 64 alloc/free/decode-sim cycles", stable)

    # ── check 3: unified zero-on-alloc (F4) through the manager ─────────────
    seq_a = next(seq_ids)
    slot_a = slot_mgr.alloc(seq_a)
    for gidx in LAYER_IDS:
        st = pools[gidx]
        st.conv_q[slot_a].fill_(3.0)
        st.conv_k[slot_a].fill_(3.0)
        st.conv_v[slot_a].fill_(3.0)
        st.recurrent_pool[slot_a].fill_(3.0)

    ok = slot_mgr.alloc(seq_a) == slot_a  # idempotent re-alloc
    ok = ok and not any(slot_is_zero(pools[g], slot_a) for g in LAYER_IDS)
    report("idempotent re-alloc preserves live state", ok)

    KimiLinearKDAWrapper.free_sequences([seq_a])
    seq_b = next(seq_ids)
    slot_b = slot_mgr.alloc(seq_b)
    ok = slot_b == slot_a  # LIFO recycle of the just-freed slot
    ok = ok and all(slot_is_zero(pools[g], slot_b) for g in LAYER_IDS)
    report("F4 zero-on-alloc covers conv+recurrent in every layer pool", ok,
           f"(recycled slot {slot_b})")

    # ── check 4: conv view consumed by causal_conv1d_update without copy ────
    st = pools[LAYER_IDS[1]]
    view = st.conv_q
    ok = (view.is_contiguous() and view.dim() == 3
          and view.shape == (SLOTS, DIM, W - 1))
    report("conv view is contiguous (slots, dim, W-1)", ok,
           f"shape={tuple(view.shape)}")

    seq_c = next(seq_ids)
    slot_c = slot_mgr.alloc(seq_c)
    slots_t = torch.tensor([slot_b, slot_c], dtype=torch.int32, device=DEVICE)
    g = torch.Generator(device=DEVICE).manual_seed(7)
    weight = torch.randn(DIM, W, generator=g, device=DEVICE,
                         dtype=torch.float32).mul_(0.1).to(DTYPE)
    init = torch.randn(2, DIM, W - 1, generator=g, device=DEVICE,
                       dtype=torch.float32).mul_(0.1).to(DTYPE)
    view[slots_t.long()] = init
    x_t = torch.randn(2, DIM, generator=g, device=DEVICE,
                      dtype=torch.float32).mul_(0.1).to(DTYPE)
    y = causal_conv1d_update(
        x_t, view, weight, bias=None, conv_state_indices=slots_t,
    )
    # State stores raw inputs: post-update slot content must be the exact
    # left-rolled window [init[..., 1:], x_t] — and it must land in the
    # MANAGER tensor (the view aliases it; no staging copy).
    expected = torch.cat([init[:, :, 1:], x_t.unsqueeze(-1)], dim=-1)
    physical = 1  # pools[LAYER_IDS[1]] views physical layer 1
    ok = torch.equal(conv_q[physical][slots_t.long()], expected)
    ok = ok and bool(torch.isfinite(y.float()).all())
    report("causal_conv1d_update rolls state in place in manager memory", ok)

    KimiLinearKDAWrapper.reset()
    print("\n" + ("ALL CHECKS PASSED" if PASS else "SOME CHECKS FAILED"))
    sys.exit(0 if PASS else 1)


if __name__ == "__main__":
    main()
