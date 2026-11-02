"""
FP8 WGMMA MoE Pipeline — Production integration for GLM-5-FP8.

Replaces the Triton FP8 grouped GEMM pipeline with WGMMA kernels:
  dispatch_scatter_3d → act_quant_3d → v8c_fused_gateup → fused_silu_quant_3d → v8b_down → reduce

Enable with: BATCHGEN_USE_WGMMA_FP8=1

Kernels:
  - v8b: FP8 WGMMA m64n32k32 grouped GEMM (down projection)
  - v8c: Fused gate+up FP8 WGMMA (two projections in one launch)
  - act_quant_3d: Per-block absmax FP8 quantization with transposed scale output
  - fused_silu_quant_3d: SiLU×gate + FP8 quant with transposed scale output
  - dispatch_scatter_3d: Route tokens from flat [G, H] into 3D [E, mtp, H]
  - reduce_weighted_scatter: Weighted sum from 3D output back to flat [G, H]

Every kernel is an AOT extension declared in ``batchgen_kernels/setup.py`` and
mirrored in ``_jit_registry.py``; nothing is compiled at runtime:
  - v8b, v8c and their TMA descriptor builders:
    ``batchgen_kernels.moe._C_fp8_wgmma_grouped_gemm``
    (``src/moe/fp8_wgmma_grouped_gemm.cu``, built with ``-DNUM_STAGES=4``)
  - act_quant_3d, fused_silu_quant_3d:
    ``batchgen_kernels.moe._C_fp8_wgmma_quant_ops``
    (``src/moe/fp8_wgmma_quant_ops.cu``)
  - dispatch_scatter_3d, reduce_weighted_scatter:
    ``batchgen_kernels.moe._C_dispatch_scatter_3d``, resolved through
    ``batchgen.moe.dispatch_scatter_3d``
"""

import logging

import torch

from batchgen.moe.dispatch_scatter_3d import require_dispatch_scatter_3d_kernels

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
BLOCK_M = 64
BLOCK_N = 32
BLOCK_K = 128
SCALE_BLOCK_K = 128
QUANT_BLOCK = 128
DEFAULT_MTP = 4096

_GROUPED_GEMM_EXTENSION = "batchgen_kernels.moe._C_fp8_wgmma_grouped_gemm"
_QUANT_OPS_EXTENSION = "batchgen_kernels.moe._C_fp8_wgmma_quant_ops"


# ══════════════════════════════════════════════════════════════════════════════
# Module loaders
# ══════════════════════════════════════════════════════════════════════════════

_module_cache = {}


def _load_aot_module(cache_key, extension):
    if cache_key in _module_cache:
        return _module_cache[cache_key]

    # Resolved at call time so tests can substitute the loader.
    from batchgen_kernels import load_extension

    module = load_extension(extension, allow_dev_jit=False)
    _module_cache[cache_key] = module
    return module


def _build_wgmma_module():
    return _load_aot_module("wgmma", _GROUPED_GEMM_EXTENSION)


def _build_fast_module():
    return _load_aot_module("fast", _QUANT_OPS_EXTENSION)


def _build_dispatch_reduce_module():
    # dispatch_scatter_3d owns this extension and its cache.
    return require_dispatch_scatter_3d_kernels()


def build_all_modules():
    """Load all CUDA modules. Call once during init()."""
    wgmma_mod = _build_wgmma_module()
    fast_mod = _build_fast_module()
    dr_mod = _build_dispatch_reduce_module()
    return wgmma_mod, fast_mod, dr_mod


# ══════════════════════════════════════════════════════════════════════════════
# WGMMAMoEBuffers — Pre-allocated buffers for the full pipeline
# ══════════════════════════════════════════════════════════════════════════════

class WGMMAMoEBuffers:
    """Manages all pre-allocated buffers for the WGMMA MoE pipeline.

    Shared across all MoE layers — buffers are reused, only weight pointers
    differ per layer (passed at forward time).
    """

    def __init__(self, wgmma_mod, fast_mod, dr_mod,
                 E_local, mtp, H, N,
                 gate_weights_list, gate_scales_list,
                 up_weights_list, up_scales_list,
                 down_weights_list, down_scales_list,
                 expert_start, top_k, num_global_tokens,
                 num_tokens_per_rank=None,
                 n_routed_experts=256,
                 device="cuda"):
        self.wgmma_mod = wgmma_mod
        self.fast_mod = fast_mod
        self.dr_mod = dr_mod
        self.E = E_local
        self.H = H
        self.N = N
        self.expert_start = expert_start
        self.top_k = top_k
        self.device = device

        self.mtp = mtp
        mtp_padded = ((mtp + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
        self.mtp_padded = mtp_padded

        num_k_h = H // QUANT_BLOCK   # H//128
        num_k_n = N // QUANT_BLOCK   # N//128

        # ── Activation buffers ──
        self.act_buf_bf16 = torch.zeros(
            E_local, mtp_padded, H, dtype=torch.bfloat16, device=device)
        self.act_buf_fp8 = torch.zeros(
            E_local, mtp_padded, H, dtype=torch.uint8, device=device)
        self.act_scale_t = torch.zeros(
            num_k_h, E_local * mtp_padded, dtype=torch.float32, device=device)

        # ── Intermediate buffers ──
        self.gate_out = torch.zeros(
            E_local * mtp_padded, N, dtype=torch.bfloat16, device=device)
        self.up_out = torch.zeros(
            E_local * mtp_padded, N, dtype=torch.bfloat16, device=device)
        self.inter_fp8 = torch.zeros(
            E_local, mtp_padded, N, dtype=torch.uint8, device=device)
        self.inter_scale_t = torch.zeros(
            num_k_n, E_local * mtp_padded, dtype=torch.float32, device=device)

        # ── Output buffer ──
        self.down_out = torch.zeros(
            E_local * mtp_padded, H, dtype=torch.bfloat16, device=device)

        # ── Dispatch/reduce buffers ──
        NK = num_global_tokens * top_k
        self.expert_counts = torch.zeros(E_local, dtype=torch.int32, device=device)
        self.expert_counters = torch.zeros(E_local, dtype=torch.int32, device=device)
        self.topk_pos = torch.full((NK,), -1, dtype=torch.int32, device=device)
        self.tpe = torch.zeros(E_local, dtype=torch.int32, device=device)
        self.global_results = torch.zeros(
            num_global_tokens, H, dtype=torch.bfloat16, device=device)
        self.num_global_tokens = num_global_tokens

        # ── AllGather/Gate pre-allocated buffers ──
        if num_tokens_per_rank is None:
            num_tokens_per_rank = num_global_tokens  # fallback: single-rank
        self.num_tokens_per_rank = num_tokens_per_rank
        self.all_tokens = torch.zeros(
            num_global_tokens, H, dtype=torch.bfloat16, device=device)
        self.padded_hidden_states = torch.zeros(
            num_tokens_per_rank, H, dtype=torch.bfloat16, device=device)
        self.router_logits_bf16 = torch.zeros(
            num_global_tokens, n_routed_experts, dtype=torch.bfloat16, device=device)
        self.router_logits = torch.zeros(
            num_global_tokens, n_routed_experts, dtype=torch.float32, device=device)

        # ── Weight pointers (per layer — stored as list of per-layer dicts) ──
        # For now, store the first layer's weights. Multi-layer support via
        # register_layer_weights() and select_layer().
        self._layer_weights = {}
        self._current_layer_id = None

        # Register the initial layer
        self.register_layer_weights(
            0, gate_weights_list, gate_scales_list,
            up_weights_list, up_scales_list,
            down_weights_list, down_scales_list)

        # ── TMA descriptors (created once, invalidated on resize) ──
        self._create_tma_descriptors()

    def register_layer_weights(self, layer_id,
                               gate_weights, gate_scales,
                               up_weights, up_scales,
                               down_weights, down_scales):
        """Register weight pointers for one MoE layer."""
        dev = self.device
        self._layer_weights[layer_id] = {
            "gate_weight_ptrs": torch.tensor(
                [w.data_ptr() for w in gate_weights], dtype=torch.int64, device=dev),
            "up_weight_ptrs": torch.tensor(
                [w.data_ptr() for w in up_weights], dtype=torch.int64, device=dev),
            "down_weight_ptrs": torch.tensor(
                [w.data_ptr() for w in down_weights], dtype=torch.int64, device=dev),
            "gate_scale_ptrs": torch.tensor(
                [s.data_ptr() for s in gate_scales], dtype=torch.int64, device=dev),
            "up_scale_ptrs": torch.tensor(
                [s.data_ptr() for s in up_scales], dtype=torch.int64, device=dev),
            "down_scale_ptrs": torch.tensor(
                [s.data_ptr() for s in down_scales], dtype=torch.int64, device=dev),
            # Keep references alive
            "_refs": (gate_weights, gate_scales, up_weights, up_scales,
                      down_weights, down_scales),
        }

    def free_buffers(self):
        """Release all GPU buffers. Called during phase transitions to reclaim memory."""
        self.act_buf_bf16 = None
        self.act_buf_fp8 = None
        self.act_scale_t = None
        self.gate_out = None
        self.up_out = None
        self.inter_fp8 = None
        self.inter_scale_t = None
        self.down_out = None
        self.expert_counts = None
        self.expert_counters = None
        self.topk_pos = None
        self.tpe = None
        self.global_results = None
        self.all_tokens = None
        self.padded_hidden_states = None
        self.router_logits_bf16 = None
        self.router_logits = None
        self._layer_weights.clear()
        self._current_tma_layer_id = None

    def _create_tma_descriptors(self, layer_id=0):
        """Create TMA descriptors for v8c (stage 1) and v8b (stage 2).

        Must be called each time the layer changes (weight pointers differ).
        """
        lw = self._layer_weights[layer_id]

        # v8c: 4E descriptors (A + B_gate + B_up + xscale)
        self.tma_v8c = self.wgmma_mod.create_tma_descriptors_v8c(
            self.act_buf_fp8, lw["gate_weight_ptrs"], lw["up_weight_ptrs"],
            self.act_scale_t, self.N, self.N, self.mtp_padded)

        # v8b: 3E descriptors (A + B_down + xscale)
        self.tma_v8b = self.wgmma_mod.create_tma_descriptors_v8b(
            self.inter_fp8, lw["down_weight_ptrs"],
            self.inter_scale_t, self.H, self.mtp_padded)

        self._current_tma_layer_id = layer_id

    def _maybe_resize(self, max_tokens_needed):
        """Resize buffers if max tokens per expert exceeds current mtp."""
        if max_tokens_needed <= self.mtp:
            return
        logger.info(
            "WGMMA pipeline: resizing buffers, mtp %d -> %d",
            self.mtp, max_tokens_needed)
        self.mtp = max_tokens_needed
        mtp_padded = ((max_tokens_needed + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
        self.mtp_padded = mtp_padded

        E = self.E
        H = self.H
        N = self.N
        dev = self.device
        num_k_h = H // QUANT_BLOCK
        num_k_n = N // QUANT_BLOCK

        self.act_buf_bf16 = torch.zeros(E, mtp_padded, H, dtype=torch.bfloat16, device=dev)
        self.act_buf_fp8 = torch.zeros(E, mtp_padded, H, dtype=torch.uint8, device=dev)
        self.act_scale_t = torch.zeros(num_k_h, E * mtp_padded, dtype=torch.float32, device=dev)
        self.gate_out = torch.zeros(E * mtp_padded, N, dtype=torch.bfloat16, device=dev)
        self.up_out = torch.zeros(E * mtp_padded, N, dtype=torch.bfloat16, device=dev)
        self.inter_fp8 = torch.zeros(E, mtp_padded, N, dtype=torch.uint8, device=dev)
        self.inter_scale_t = torch.zeros(num_k_n, E * mtp_padded, dtype=torch.float32, device=dev)
        self.down_out = torch.zeros(E * mtp_padded, H, dtype=torch.bfloat16, device=dev)

        # Recreate TMA descriptors (they encode buffer dimensions)
        self._create_tma_descriptors()

    def _resize_topk_pos_if_needed(self, num_global_tokens):
        """Resize topk_pos and global_results if global token count changed."""
        NK = num_global_tokens * self.top_k
        if self.topk_pos.numel() < NK:
            self.topk_pos = torch.full((NK,), -1, dtype=torch.int32, device=self.device)
            self.num_global_tokens = num_global_tokens
        if self.global_results.shape[0] < num_global_tokens:
            self.global_results = torch.zeros(
                num_global_tokens, self.H, dtype=torch.bfloat16, device=self.device)

    def forward(self, layer_id, global_x, topk_idx, topk_weight):
        """Full WGMMA MoE pipeline forward.

        Args:
            layer_id: Which MoE layer (for weight selection)
            global_x: [G, H] BF16 — all-gathered input
            topk_idx: [G, K] INT32 — expert indices from gating
            topk_weight: [G, K] F32 — routing weights from gating

        Returns:
            global_results: [G, H] BF16 — weighted expert output
        """
        G = global_x.shape[0]
        K_topk = topk_idx.shape[1]
        E = self.E
        H = self.H
        N = self.N
        mtp = self.mtp_padded

        lw = self._layer_weights[layer_id]

        # Recreate TMA descriptors if layer changed (weight pointers differ)
        if getattr(self, '_current_tma_layer_id', None) != layer_id:
            self._create_tma_descriptors(layer_id)

        # Ensure topk_pos is large enough
        self._resize_topk_pos_if_needed(G)

        # ── Step 1: Dispatch scatter 3D ──
        self.act_buf_bf16.zero_()
        self.topk_pos.fill_(-1)

        expert_counts, topk_pos = self.dr_mod.dispatch_scatter_3d(
            global_x, topk_idx.to(torch.int32), self.act_buf_bf16,
            self.expert_start, E, mtp,
            self.expert_counts, self.expert_counters, self.topk_pos)

        self.tpe.copy_(expert_counts)
        # Note: no .item() sync here — buffers are pre-sized to mtp=4096
        # which is always sufficient for decode (typically <100 tokens/expert).

        # ── Step 2: act_quant_3d (write directly into pre-allocated buffers) ──
        self.fast_mod.act_quant_3d(
            self.act_buf_bf16, self.tpe,
            self.act_buf_fp8, self.act_scale_t)

        # ── Step 3: v8c fused gate+up WGMMA ──
        num_k_h = H // QUANT_BLOCK
        self.wgmma_mod.grouped_fp8_moe_gemm_v8c(
            self.tma_v8c, self.tpe,
            lw["gate_scale_ptrs"], lw["up_scale_ptrs"],
            num_k_h,  # w_scale_stride_n
            self.gate_out, self.up_out,
            mtp, N, N, H, E)

        # ── Step 4: fused_silu_quant_3d (write directly into pre-allocated buffers) ──
        gate_3d = self.gate_out.view(E, mtp, N)
        up_3d = self.up_out.view(E, mtp, N)
        self.fast_mod.fused_silu_quant_3d(
            gate_3d, up_3d, self.tpe,
            self.inter_fp8, self.inter_scale_t)

        # ── Step 5: v8b down WGMMA ──
        num_k_n = N // QUANT_BLOCK
        self.wgmma_mod.grouped_fp8_moe_gemm_v8b(
            self.tma_v8b, self.tpe,
            lw["down_scale_ptrs"],
            num_k_n,  # w_scale_stride_n
            self.down_out,
            mtp, H, N, E)

        # ── Step 6: Reduce weighted scatter ──
        self.global_results[:G].zero_()
        topk_weights_flat = topk_weight.reshape(-1)
        self.dr_mod.reduce_weighted_scatter(
            self.down_out, topk_pos, topk_weights_flat,
            G, H, K_topk, self.global_results[:G])

        return self.global_results[:G]
