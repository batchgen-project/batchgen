"""Torch CPU stand-ins for the fla symbols ``modeling_kimi_linear.py`` imports.

The oracle hard-raises without fla (modeling_kimi_linear.py:46-53) and fla is
triton/GPU-only, so CPU parity tests install these into ``sys.modules`` BEFORE
loading the oracle.  Exactly the 7 imported symbols are provided:

    fla.modules:        ShortConvolution, FusedRMSNormGated
    fla.ops.kda:        chunk_kda, fused_recurrent_kda
    fla.ops.utils.index: prepare_cu_seqlens_from_mask, prepare_lens_from_mask
    fla.utils:          tensor_cache

``chunk_kda`` delegates the KDA math to the production-vendored
``kimi_k3/kda_reference.py`` (fla's own torch reference functions).  Stated
caveat: on CPU the same reference backs BOTH stacks, so the kernel interior
cancels in parity tests and is closed only by the staged GPU test.  Everything
else the oracle computes around these symbols (projections, decays, gates,
norms, router, MoE, AttnRes, layer map) is genuinely cross-validated, because
BatchGen's model.py implements those independently.

Prefill-only: caches, varlen and initial states raise loudly.
"""

from __future__ import annotations

import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F


def _load_kda_reference():
    """Import the production-vendored torch KDA core by file path (the tests
    must not import the `batchgen` package — that JIT-builds the core engine)."""
    import importlib.util
    from pathlib import Path
    path = (Path(__file__).resolve().parents[2]
            / "batchgen" / "models" / "moonshotai" / "kimi_k3" / "kda_reference.py")
    name = "_k3_kda_reference_for_shim"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class ShortConvolution(nn.Module):
    """Causal depthwise conv + activation, mirroring fla's
    ``ShortConvolution(hidden_size, kernel_size, activation=..., bias=False)``
    (fla/modules/conv/short_conv.py: nn.Conv1d(D, D, W, groups=D, bias=False,
    padding=W-1) + act).  fp32 interior, stored in input dtype.  Weight is
    created fp32 (the K3 checkpoint ships the conv weights F32)."""

    def __init__(self, hidden_size: int, kernel_size: int,
                 activation: str | None = None, bias: bool = False, **kwargs):
        super().__init__()
        if bias:
            raise NotImplementedError("fla ShortConvolution shim: bias=False only")
        if activation not in (None, "silu"):
            raise NotImplementedError(
                "fla ShortConvolution shim: activation must be None or 'silu', "
                "got {!r}".format(activation))
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.activation = activation
        self.weight = nn.Parameter(
            torch.empty(hidden_size, 1, kernel_size, dtype=torch.float32))

    def forward(self, x: torch.Tensor, cache=None, output_final_state: bool = False,
                cu_seqlens=None, mask=None, **kwargs):
        if cache is not None or output_final_state:
            raise NotImplementedError("fla shim is prefill-only: no conv cache")
        if cu_seqlens is not None:
            raise NotImplementedError("fla shim is prefill-only: no varlen (M4)")
        if mask is not None:
            raise NotImplementedError("fla shim: padding mask not supported")
        seq_len = x.shape[1]
        y = F.conv1d(
            x.transpose(1, 2).float(),
            self.weight.float(),
            bias=None,
            groups=self.hidden_size,
            padding=self.kernel_size - 1,
        )[..., :seq_len]
        if self.activation == "silu":
            y = F.silu(y)
        return y.transpose(1, 2).to(x.dtype), None


class FusedRMSNormGated(nn.Module):
    """Gated RMSNorm over the LAST dim (per head), sigmoid gate — mirroring
    fla's FusedRMSNormGated(hidden, eps, activation='sigmoid').  Entire op
    fp32 including sigmoid(g) (fla fused_norm_gate.py:67-104); weight applied
    to the normalized x BEFORE the gate; stored in input dtype."""

    def __init__(self, hidden_size: int, eps: float = 1e-5,
                 activation: str = "sigmoid", **kwargs):
        super().__init__()
        if activation != "sigmoid":
            raise NotImplementedError(
                "fla FusedRMSNormGated shim: sigmoid activation only")
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        y = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        y = y * self.weight.float()
        y = y * torch.sigmoid(g.float())
        return y.to(x.dtype)


def chunk_kda(q, k, v, g, beta, A_log=None, dt_bias=None, scale=None,
              initial_state=None, output_final_state=False,
              use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
              use_beta_sigmoid_in_kernel=True, safe_gate=False,
              lower_bound=None, transpose_state_layout=False,
              cu_seqlens=None, **kwargs):
    """Shim of fla 0.4.2 ``chunk_kda`` for the oracle's prefill call
    (modeling_kimi_linear.py:609-627).  Flag handling:

      * use_qk_l2norm_in_kernel — honored (l2norm with fla's SUM+1e-6 form).
      * use_gate_in_kernel      — must be True here (g is the RAW decay input).
      * use_beta_sigmoid_in_kernel — accepted and IGNORED, exactly like fla
        0.4.2 (dead kwarg swallowed by **kwargs; sigmoid(beta) is
        unconditional in the fused gate kernel, gate.py:118/130).
      * safe_gate + lower_bound — selects the LOWER-BOUND gate (the K3 path).
      * transpose_state_layout  — honored on the returned state ([V,K]-major).
    """
    if cu_seqlens is not None:
        raise NotImplementedError("fla shim is prefill-only: varlen is M4")
    if initial_state is not None:
        raise NotImplementedError("fla shim is prefill-only: no initial_state")
    if not use_gate_in_kernel:
        raise NotImplementedError(
            "fla shim expects the oracle's use_gate_in_kernel=True call form "
            "(g = raw pre-decay input)")
    ref = _load_kda_reference()
    if A_log.shape[-1] != q.shape[2]:
        raise ValueError(
            "chunk_kda shim: A_log must be [num_heads] ([{}] given, {} heads)"
            .format(A_log.shape[-1], q.shape[2]))

    if safe_gate:
        if lower_bound is None:
            raise ValueError("safe_gate=True requires lower_bound")
        eff_lower_bound = float(lower_bound)
    else:
        eff_lower_bound = None

    # Single-pass composition (identical to ref.kda_reference_prefill, plus the
    # optional final state the oracle always requests).
    if use_qk_l2norm_in_kernel:
        q, k = ref.l2norm_ref(q), ref.l2norm_ref(k)
    beta_post = beta.float().sigmoid()
    if eff_lower_bound is not None:
        g_log = ref.naive_kda_lowerbound_gate(g, A_log, dt_bias, lower_bound=eff_lower_bound)
    else:
        g_log = ref.naive_kda_gate(g, A_log, dt_bias)
    o, final_state = ref.naive_recurrent_kda(
        q, k, v, g_log, beta_post, scale=scale,
        initial_state=None, output_final_state=bool(output_final_state))
    if final_state is not None and transpose_state_layout:
        final_state = final_state.mT   # naive is [K,V]-major; kernel returns [V,K]
    return o, final_state


def fused_recurrent_kda(*args, **kwargs):
    raise NotImplementedError(
        "fused_recurrent_kda is the DECODE kernel — K3 decode arrives in M3; "
        "the CPU shim deliberately does not implement it")


def prepare_lens_from_mask(mask: torch.Tensor) -> torch.Tensor:
    return mask.sum(dim=-1, dtype=torch.int32)


def prepare_cu_seqlens_from_mask(mask: torch.Tensor, dtype=torch.int32) -> torch.Tensor:
    lens = prepare_lens_from_mask(mask)
    return F.pad(lens.cumsum(0, dtype=dtype), (1, 0))


def tensor_cache(fn):
    """Identity decorator (fla's memoization is a perf detail, not semantics)."""
    return fn


def install(force: bool = True) -> None:
    """Register the shim modules in ``sys.modules``.

    ``force=True`` (default) installs even when a real fla exists, so CPU runs
    are deterministic regardless of the environment.  Never call this in
    production code — test harness only.
    """
    fla = types.ModuleType("fla")
    fla.__path__ = []          # mark as package
    fla.__version__ = "0.0.0-batchgen-cpu-shim"
    modules = types.ModuleType("fla.modules")
    modules.ShortConvolution = ShortConvolution
    modules.FusedRMSNormGated = FusedRMSNormGated
    ops = types.ModuleType("fla.ops")
    ops_kda = types.ModuleType("fla.ops.kda")
    ops_kda.chunk_kda = chunk_kda
    ops_kda.fused_recurrent_kda = fused_recurrent_kda
    ops_utils = types.ModuleType("fla.ops.utils")
    ops_utils_index = types.ModuleType("fla.ops.utils.index")
    ops_utils_index.prepare_cu_seqlens_from_mask = prepare_cu_seqlens_from_mask
    ops_utils_index.prepare_lens_from_mask = prepare_lens_from_mask
    utils = types.ModuleType("fla.utils")
    utils.tensor_cache = tensor_cache

    installed = {
        "fla": fla,
        "fla.modules": modules,
        "fla.ops": ops,
        "fla.ops.kda": ops_kda,
        "fla.ops.utils": ops_utils,
        "fla.ops.utils.index": ops_utils_index,
        "fla.utils": utils,
    }
    for name, module in installed.items():
        if not force and name in sys.modules:
            raise RuntimeError(
                "Real fla is importable but the CPU shim was asked not to "
                "override it; pass force=True (CPU tests want the shim "
                "unconditionally)")
        sys.modules[name] = module
