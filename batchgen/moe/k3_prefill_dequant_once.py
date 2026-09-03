"""K3 prefill routed-expert path: dequantize the layer's Marlin-order MXFP4
shard ONCE to BF16, then run the chunks through BF16 grouped GEMMs.

The production M16 grouped Marlin kernel is SM-bound on its per-16-row-tile
E2M1 decode (kernel-dev family ``moe/k3_routed_mxfp4``: 25.5 ms per 65K-row
chunk at 172 TFLOP/s). Staging BF16 weights once per layer (7.4 GB per rank
for 112 experts) and running ``torch._grouped_mm`` gate+up -> SiTU -> down
measured 8.7 ms per chunk on H200 with parity against the same reference.
Prefill only: decode keeps the Marlin path.

Contract of :meth:`K3PrefillDequantOnce.expert_path` is the compact
``ResidentEPMXFP4MoELayer._expert_path`` contract: ``(expert_out, topk_pos)``
with ``topk_pos[t*K + k]`` = absolute row of that assignment in
``expert_out`` or -1, so ``_combine_fp32`` consumes it unchanged. Rows are
grouped by expert and each group is padded to 16 rows (grouped_mm alignment).
"""
import torch
import triton
import triton.language as tl

from batchgen.moe.routing import dispatch_count_gather_cuda

GROUP_ALIGN = 16
_TABLES = {}


def _tables(device):
    """Inverse Marlin nibble permutation (tiled position -> marlin nibble) and
    inverse scale permutation, cached per device."""
    key = str(device)
    if key not in _TABLES:
        from batchgen.moe.marlin_weight_prep import _get_scale_perms, get_weight_perm
        perm = get_weight_perm(4)  # raw/tiled -> marlin, 1024 entries
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(perm.numel())
        # q_tiled[i] = q_marlin[inv_perm[i]] in the transform reference, i.e.
        # the marlin nibble of tiled position i is inv[i] where inv is the
        # inverse of get_weight_perm (marlin_to_wgmma_gpu uses the same).
        scale_perm, _ = _get_scale_perms()
        inv_s = [0] * len(scale_perm)
        for i, p in enumerate(scale_perm):
            inv_s[p] = i
        _TABLES[key] = (inv.to(device=device, dtype=torch.int32),
                        torch.tensor(inv_s, device=device, dtype=torch.int32))
    return _TABLES[key]


@triton.jit
def _dequant_marlin_kernel(qw_ptr, s_ptr, invp_ptr, invs_ptr, out_ptr, N, K,
                           BN: tl.constexpr, BK: tl.constexpr):
    """out[n, k] (bf16, [N, K] row-major) from marlin_qw [K//16, N*2] int32 and
    marlin_s [K//32, N] uint8 E8M0. One program: BN n-rows x BK k-cols."""
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    k = tl.program_id(1) * BK + tl.arange(0, BK)
    n2 = n[:, None]
    k2 = k[None, :]
    mask = (n2 < N) & (k2 < K)
    kb = k2 >> 4
    k_in = k2 & 15
    g = n2 >> 6
    tb = (n2 & 63) >> 4
    n_in = n2 & 15
    tiled = tb * 256 + k_in * 16 + n_in                      # position inside the 16x64 tile
    m = tl.load(invp_ptr + tiled)                             # marlin nibble inside the tile
    nib = g * 1024 + m                                        # nibble index inside the K-block row
    word = tl.load(qw_ptr + kb * (N * 2) + (nib >> 3), mask=mask, other=0)
    code = (word >> ((nib & 7) * 4)) & 0xF
    sign = tl.where(((code >> 3) & 1) == 1, -1.0, 1.0)
    e = (code >> 1) & 3
    man = (code & 1).to(tl.float32)
    mag = tl.where(e == 0, man * 0.5, (1.0 + 0.5 * man) * tl.exp2((e - 1).to(tl.float32)))
    kg = k2 >> 5
    sc_col = g * 64 + tl.load(invs_ptr + (n2 & 63))
    s = tl.load(s_ptr + kg * N + sc_col, mask=mask, other=127).to(tl.int32)
    val = sign * mag * tl.exp2((s - 127).to(tl.float32))
    tl.store(out_ptr + n2 * K + k2, val.to(tl.bfloat16), mask=mask)


def dequant_marlin_bf16(marlin_qw, marlin_s, out, BN=32, BK=128):
    """marlin_qw [K//16, N*2] int32 + marlin_s [K//32, N] uint8 -> out [N, K] bf16."""
    K = int(marlin_qw.shape[0]) * 16
    N = int(marlin_qw.shape[1]) // 2
    assert tuple(out.shape) == (N, K), (out.shape, N, K)
    invp, invs = _tables(out.device)
    grid = (triton.cdiv(N, BN), triton.cdiv(K, BK))
    _dequant_marlin_kernel[grid](marlin_qw, marlin_s, invp, invs, out, N, K, BN=BN, BK=BK, num_warps=4)
    return out


@triton.jit
def _situ_kernel(x_ptr, out_ptr, N, stride_x, stride_o, beta, lbeta, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    g = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(x_ptr + row * stride_x + N + offs, mask=mask, other=0.0).to(tl.float32)
    tg = 2.0 * tl.sigmoid(2.0 * g / beta) - 1.0
    tu = 2.0 * tl.sigmoid(2.0 * u / lbeta) - 1.0
    tl.store(out_ptr + row * stride_o + offs, (beta * tg * tl.sigmoid(g) * (lbeta * tu)).to(tl.bfloat16), mask=mask)


def situ_gated(gu, out, beta=4.0, lbeta=25.0, BLOCK=1024):
    """[rows, 2N] bf16 (gate | up) -> out [rows, N] bf16, K3 SiTU (beta 4, linear beta 25)."""
    rows, twoN = gu.shape
    N = twoN // 2
    _situ_kernel[(rows, triton.cdiv(N, BLOCK))](gu, out, N, gu.stride(0), out.stride(0), beta, lbeta, BLOCK=BLOCK, num_warps=4)
    return out


class K3PrefillDequantOnce:
    """BF16 staging of one layer's local expert shard plus the chunk path."""

    def __init__(self, shard, device):
        self.E, self.N, self.K = int(shard.num_local), int(shard.N), int(shard.K_latent)
        self.device = device
        self.w_gu = torch.empty(self.E, 2 * self.N, self.K, dtype=torch.bfloat16, device=device)
        self.w_d = torch.empty(self.E, self.K, self.N, dtype=torch.bfloat16, device=device)
        if hasattr(shard, "marlin_packed"):
            # streamed-SP8 shard (StreamedSP8LayerBuffer._make_shard): stacked
            # Marlin-order views [E, k//16, n*2] int32 and [E, k//32, n] uint8
            packed, scales = shard.marlin_packed, shard.marlin_scales
            per_expert = [
                {p: (packed[p][e], scales[p][e]) for p in ("w1", "w3", "w2")}
                for e in range(self.E)
            ]
        else:
            # build_layer_shard: one dict of (marlin_qw, marlin_s) per expert
            per_expert = shard._tensors
        for e, t in enumerate(per_expert):
            dequant_marlin_bf16(*t["w1"], self.w_gu[e, : self.N])
            dequant_marlin_bf16(*t["w3"], self.w_gu[e, self.N:])
            dequant_marlin_bf16(*t["w2"], self.w_d[e])
        # grouped_mm mat2 is [E, in, out]: transposed views of the [E, out, in] staging
        self.w_gu_t = self.w_gu.transpose(1, 2)
        self.w_d_t = self.w_d.transpose(1, 2)

    def expert_path(self, x_latent, topk_idx_i32, num_rows, expert_start, packed_capacity=None):
        E, N, K = self.E, self.N, self.K
        device = x_latent.device
        top_k = topk_idx_i32.shape[-1]
        cap = max(int(packed_capacity), 1) if packed_capacity is not None else None
        dispatched = torch.empty(cap, K, dtype=torch.bfloat16, device=device) if cap else None
        dispatched, counts, offsets, topk_pos = dispatch_count_gather_cuda(
            x_latent, topk_idx_i32, expert_start, E, dispatched_x=dispatched,
            topk_pos=torch.empty(num_rows * top_k, dtype=torch.int32, device=device),
        )
        # pad every expert group to a multiple of GROUP_ALIGN rows
        cnt = counts.to(torch.int64)
        pad_cnt = (cnt + GROUP_ALIGN - 1) // GROUP_ALIGN * GROUP_ALIGN
        pad_end = torch.cumsum(pad_cnt, 0)
        pad_start = pad_end - pad_cnt
        total = int(pad_end[-1].item())
        src_rows = int(offsets[-1].item())
        e_of_row = torch.repeat_interleave(torch.arange(E, device=device), cnt, output_size=src_rows)
        src = torch.arange(src_rows, device=device)
        dst = pad_start[e_of_row] + (src - offsets[:-1].to(torch.int64)[e_of_row])
        A = torch.zeros(max(total, GROUP_ALIGN), K, dtype=torch.bfloat16, device=device)
        A.index_copy_(0, dst, dispatched[:src_rows])
        if src_rows == 0:
            # No assignment of this chunk routes to this rank's experts (small
            # or padded chunks): topk_pos is already all -1, nothing to compute.
            return torch.zeros(GROUP_ALIGN, K, dtype=torch.bfloat16, device=device), topk_pos
        # map compact positions to padded rows without a data-dependent sync:
        # the appended sentinel row keeps -1 for non-owned assignments
        pos = topk_pos.to(torch.int64)
        dst_ext = torch.cat([dst, dst.new_full((1,), -1)])
        topk_pos_padded = dst_ext[torch.where(pos >= 0, pos, src_rows)].to(torch.int32)
        offs = pad_end.to(torch.int32)
        gu = torch._grouped_mm(A, self.w_gu_t, offs=offs)            # [rows_p, 2N]
        h = torch.empty(A.shape[0], N, dtype=torch.bfloat16, device=device)
        situ_gated(gu, h)
        expert_out = torch._grouped_mm(h, self.w_d_t, offs=offs)     # [rows_p, K]
        return expert_out, topk_pos_padded
