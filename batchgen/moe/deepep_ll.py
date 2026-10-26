"""DeepEP low-latency EP exchange for resident-MoE decode graphs.

Slice 3 of ``batchgen_design/model_support/kimi_k3/DECODE_CONCURRENCY_PLAN.md``.
One-sided IBGDA/NVLink writes replace the 16-rank NCCL all_gather (latent +
routing) and fp32 reduce_scatter of the resident-EP decode path. MEASURED on
2x8 H200 (RoCE v2, EP16, top-16, hidden 3584, graph-replayed dispatch+combine
round trip): 4 rows/rank 106 us, 8: 128, 16: 176, 32: 242 -- against ~362 us
per layer for the captured NCCL pair at the tail (r33 trace).

Needs the K3 DeepEP build (``kNumMaxTopK`` raised to 16 and hidden 3584 added
to ``SWITCH_HIDDEN``; the upstream wheel asserts ``num_topk <= 11`` and knows
2048/2560/3072/4096/5120/6144/7168/8192 only) and the NVSHMEM environment the
launcher exports (``NVSHMEM_IB_ENABLE_IBGDA=1``, ``NVSHMEM_IB_GID_INDEX``,
``NVSHMEM_HCA_LIST``). Every LL call uses the same
``num_max_dispatch_tokens_per_rank`` / hidden / num_experts so the buffer is
cleaned once at construction and no bucket switch needs a re-clean.

The GLM-5 decode MoE graph
(``batchgen/models/glm/glm5/moe_cuda_graph_segments.py``) uses the same
exchange with the stock upstream build: its shapes (hidden 6144, top-8) are in
the upstream instantiation list, so no patch is required there.

Build capability
----------------
The compile-time limits above are not introspectable from Python, so
``deepep_unavailable_reason()`` checks what the import actually provides (the
module, the ``Buffer`` methods, the keyword arguments we pass, the SM90 compile
flag, ``topk_idx_t``) and validates the requested shape against the capability
table of the build in hand -- upstream limits by default, widened when the K3
build marker is present. A caller that names no shape gets the legacy K3
contract: the patched build is required, because an upstream wheel imports
fine and only asserts inside the first dispatch.
"""
import os
from dataclasses import dataclass
from typing import FrozenSet, Optional, Tuple

import torch


K3_BUILD_MARKER = "K3_BUILD"

# Upstream (deep_ep 1.2.1) low-latency compile-time limits, read from
# csrc/kernels/internode_ll.cu (`constexpr int kNumMaxTopK = 11`) and
# csrc/kernels/launch.cuh (`SWITCH_HIDDEN`). Keep in sync with the pinned
# commit; a build whose limits differ must advertise itself with a marker.
UPSTREAM_MAX_TOPK = 11
UPSTREAM_HIDDEN_SIZES: FrozenSet[int] = frozenset(
    {2048, 2560, 3072, 4096, 5120, 6144, 7168, 8192})

# The K3 patch (DECODE_CONCURRENCY_PLAN.md) raises kNumMaxTopK to 16 and adds
# hidden 3584 to SWITCH_HIDDEN. It is a superset of upstream.
K3_MAX_TOPK = 16
K3_EXTRA_HIDDEN_SIZES: FrozenSet[int] = frozenset({3584})

# csrc/deep_ep.cpp low_latency_dispatch, FP8 path.
FP8_HIDDEN_MULTIPLE = 512          # EP_HOST_ASSERT(hidden % 512 == 0)
FP8_TOKEN_ROWS_MULTIPLE = 4        # (num_ranks * mtpr) % 4 == 0, TMA
QUANT_BLOCK = 128                  # scales are per 128 hidden channels

_REQUIRED_DISPATCH_KWARGS = (
    "use_fp8", "round_scale", "use_ue8m0", "async_finish", "return_recv_hook")
_REQUIRED_COMBINE_KWARGS = ("async_finish", "return_recv_hook", "out")


@dataclass(frozen=True)
class DeepEPCapability:
    """What the imported DeepEP build can actually run."""

    source: str                      # "k3-patched" | "upstream"
    max_topk: int
    hidden_sizes: FrozenSet[int]
    supports_fp8_dispatch: bool
    sm90_compiled: Optional[bool]    # None when the build does not report it

    def shape_reason(self, *, hidden: int, topk: int) -> Optional[str]:
        if topk > self.max_topk:
            return (f"DeepEP build ({self.source}) supports top-k <= "
                    f"{self.max_topk}, model needs top-{topk}")
        if hidden not in self.hidden_sizes:
            return (f"DeepEP build ({self.source}) has no low-latency kernel "
                    f"for hidden {hidden} (available: "
                    f"{sorted(self.hidden_sizes)})")
        return None


def _probe_capability() -> Tuple[Optional[DeepEPCapability], Optional[str]]:
    """Import DeepEP and introspect it. Returns (capability, reason)."""
    try:
        import deep_ep
    except Exception as exc:  # noqa: BLE001 - absent or unloadable extension
        return None, f"deep_ep is not importable: {type(exc).__name__}: {exc}"

    buffer_cls = getattr(deep_ep, "Buffer", None)
    if buffer_cls is None:
        return None, "deep_ep exposes no Buffer class"
    for name in ("low_latency_dispatch", "low_latency_combine",
                 "clean_low_latency_buffer",
                 "get_low_latency_rdma_size_hint"):
        if not hasattr(buffer_cls, name):
            return None, f"deep_ep.Buffer has no {name}"
    if getattr(deep_ep, "topk_idx_t", None) is None:
        return None, "deep_ep does not export topk_idx_t"

    import inspect

    for attr, required in (("low_latency_dispatch", _REQUIRED_DISPATCH_KWARGS),
                           ("low_latency_combine", _REQUIRED_COMBINE_KWARGS)):
        try:
            params = inspect.signature(getattr(buffer_cls, attr)).parameters
        except (TypeError, ValueError) as exc:
            return None, f"deep_ep.Buffer.{attr} is not introspectable: {exc}"
        missing = [k for k in required if k not in params]
        if missing:
            return None, (f"deep_ep.Buffer.{attr} does not accept {missing}; "
                          "the installed build predates the API this "
                          "integration targets")
    dispatch_params = inspect.signature(
        buffer_cls.low_latency_dispatch).parameters

    marker = os.path.join(os.path.dirname(deep_ep.__file__), K3_BUILD_MARKER)
    if os.path.exists(marker):
        capability = DeepEPCapability(
            source="k3-patched",
            max_topk=K3_MAX_TOPK,
            hidden_sizes=UPSTREAM_HIDDEN_SIZES | K3_EXTRA_HIDDEN_SIZES,
            supports_fp8_dispatch="use_fp8" in dispatch_params,
            sm90_compiled=_sm90_compiled(),
        )
    else:
        capability = DeepEPCapability(
            source="upstream",
            max_topk=UPSTREAM_MAX_TOPK,
            hidden_sizes=UPSTREAM_HIDDEN_SIZES,
            supports_fp8_dispatch="use_fp8" in dispatch_params,
            sm90_compiled=_sm90_compiled(),
        )
    return capability, None


def _sm90_compiled() -> Optional[bool]:
    try:
        import deep_ep_cpp
    except Exception:  # noqa: BLE001 - the C++ module name may differ
        return None
    probe = getattr(deep_ep_cpp, "is_sm90_compiled", None)
    if probe is None:
        return None
    try:
        return bool(probe())
    except Exception:  # noqa: BLE001 - never let a probe abort the gate
        return None


def deepep_capability() -> Optional[DeepEPCapability]:
    """The imported build's capability, or None when DeepEP is unusable."""
    capability, _reason = _probe_capability()
    return capability


def deepep_unavailable_reason(
    *,
    hidden: Optional[int] = None,
    topk: Optional[int] = None,
    num_experts: Optional[int] = None,
    num_ranks: Optional[int] = None,
    max_tokens_per_rank: Optional[int] = None,
    use_fp8: bool = False,
) -> Optional[str]:
    """None when DeepEP LL can run the given shape; else a fail-loud reason.

    Called with no shape arguments it enforces the legacy K3 contract (the
    patched build must be present), because an upstream wheel imports fine and
    only asserts inside the first dispatch -- the caller has not told us what
    it will ask for, so the widest build is required.
    """
    capability, reason = _probe_capability()
    if reason is not None:
        return reason
    if capability.sm90_compiled is False:
        return ("the installed DeepEP was not compiled for SM90; the "
                "low-latency kernels target Hopper")

    if hidden is None or topk is None:
        if capability.source != "k3-patched":
            return ("DeepEP shape was not declared by the caller and the "
                    f"installed build is '{capability.source}'; the patched "
                    f"build marker {K3_BUILD_MARKER} is required for an "
                    "undeclared shape")
        return None

    shape_reason = capability.shape_reason(hidden=int(hidden), topk=int(topk))
    if shape_reason is not None:
        return shape_reason

    if num_experts is not None and num_ranks is not None:
        if int(num_ranks) <= 0:
            return f"num_ranks must be positive, got {num_ranks}"
        if int(num_experts) % int(num_ranks) != 0:
            return (f"num_experts {num_experts} is not divisible by "
                    f"num_ranks {num_ranks}; DeepEP shards experts evenly")
    if max_tokens_per_rank is not None and int(max_tokens_per_rank) <= 0:
        return (f"max_tokens_per_rank must be positive, got "
                f"{max_tokens_per_rank}")

    if use_fp8:
        if not capability.supports_fp8_dispatch:
            return ("the installed DeepEP build's low_latency_dispatch has no "
                    "use_fp8 argument")
        if int(hidden) % FP8_HIDDEN_MULTIPLE != 0:
            return (f"DeepEP FP8 dispatch requires hidden % "
                    f"{FP8_HIDDEN_MULTIPLE} == 0, got {hidden}")
        if int(hidden) // QUANT_BLOCK * 4 > int(hidden):
            return f"DeepEP scale block does not fit hidden {hidden}"
        if num_ranks is not None and max_tokens_per_rank is not None:
            rows = int(num_ranks) * int(max_tokens_per_rank)
            if rows % FP8_TOKEN_ROWS_MULTIPLE != 0:
                return (f"DeepEP FP8 dispatch requires num_ranks * "
                        f"max_tokens_per_rank % {FP8_TOKEN_ROWS_MULTIPLE} "
                        f"== 0, got {rows}")
    return None


def deepep_available(**shape) -> bool:
    """True when DeepEP LL can run ``shape`` (see deepep_unavailable_reason).

    No arguments keeps the K3 contract: only the patched build qualifies.
    """
    return deepep_unavailable_reason(**shape) is None


def require_deepep(**shape) -> DeepEPCapability:
    """Fail loudly unless DeepEP LL can run ``shape``; no silent NCCL fallback."""
    reason = deepep_unavailable_reason(**shape)
    if reason is not None:
        raise RuntimeError(f"DeepEP low-latency exchange unavailable: {reason}")
    return deepep_capability()


class DeepEPLowLatencyExchange:
    """One low-latency ``deep_ep.Buffer`` shared by every MoE layer."""

    def __init__(self, group, *, max_tokens_per_rank: int, hidden: int,
                 num_experts: int, num_local_experts: int):
        import deep_ep

        self.max_tokens_per_rank = int(max_tokens_per_rank)
        self.hidden = int(hidden)
        self.num_experts = int(num_experts)
        self.num_local_experts = int(num_local_experts)
        self.num_ranks = int(group.size())
        self.rdma_bytes = int(deep_ep.Buffer.get_low_latency_rdma_size_hint(
            self.max_tokens_per_rank, self.hidden, self.num_ranks, self.num_experts))
        # explicitly_destroy: DeepEP's destructor otherwise runs a COLLECTIVE
        # nvshmem barrier + finalize at interpreter exit, and ranks exit at
        # different times -> workers hang in D state for 10-15 min after every
        # run (r35b/c2). The exchange lives for the process; exit reclaims it.
        self.buffer = deep_ep.Buffer(
            group, num_nvl_bytes=0, num_rdma_bytes=self.rdma_bytes,
            low_latency_mode=True, num_qps_per_rank=int(num_local_experts),
            explicitly_destroy=True)
        qp_depth = int(getattr(self.buffer, "nvshmem_qp_depth", 0) or 0)
        if qp_depth and qp_depth < (self.max_tokens_per_rank + 1) * 2:
            raise RuntimeError(
                f"DeepEP nvshmem_qp_depth {qp_depth} is below the "
                f"{(self.max_tokens_per_rank + 1) * 2} required for "
                f"{self.max_tokens_per_rank} tokens/rank")
        self.buffer.clean_low_latency_buffer(
            self.max_tokens_per_rank, self.hidden, self.num_experts)

    # ---- expert-major receive layout -------------------------------------
    @property
    def expert_stride(self) -> int:
        """Rows per local expert in ``recv_x``: ``num_ranks * tokens/rank``."""
        return self.num_ranks * self.max_tokens_per_rank

    @property
    def recv_rows(self) -> int:
        """Total rows of the expert-major receive space."""
        return self.num_local_experts * self.expert_stride

    # ---- send / receive split -------------------------------------------
    #
    # Every LL call below takes ``split_phases``. False (the default) asks
    # DeepEP for ONE kernel per exchange: it is launched on DeepEP's own comm
    # stream (fenced against the caller's stream on both sides, since
    # ``async_finish`` is False) and its CTAs send, then spin until every peer
    # has delivered. True asks for ``return_recv_hook=True``: DeepEP launches
    # the SEND phase on the caller's current stream and hands back a hook; the
    # hook launches the RECEIVE phase on that same stream (csrc/deep_ep.cpp
    # binds the stream current at the send into the hook). Both are
    # cooperative launches, like the single kernel.
    # The hook is issued here, immediately, so the call returns with the same
    # contract either way -- the outputs are complete on the caller's stream
    # once the call has been issued -- and only the kernel decomposition
    # differs: two short kernels in stream order instead of one that holds the
    # SMs it occupies across the whole round trip. Both phases run the same
    # data movement and the same reduction order, so the outputs are the same
    # bits. Both launches are ordinary stream work and are captured by a CUDA
    # graph like the single kernel is; the buffer double-buffering index is
    # advanced once per call on the host, at the send, exactly as in the
    # single-kernel mode.

    @staticmethod
    def _issue_recv(hook, what: str) -> None:
        if hook is None or not callable(hook):
            raise RuntimeError(
                f"DeepEP {what}(return_recv_hook=True) returned no callable "
                f"receive hook (got {type(hook).__name__}); the installed "
                "build does not implement the send/receive split")
        hook()

    def dispatch(self, x: torch.Tensor, topk_idx: torch.Tensor, *,
                 split_phases: bool = False
                 ) -> Tuple[torch.Tensor, torch.Tensor, tuple]:
        """x [rows, hidden] bf16, topk_idx [rows, K] int64 (global expert ids,
        -1 = none) -> recv_x [E_local, num_ranks * max_tokens_per_rank, hidden]
        bf16 (expert-major, each expert's tokens packed from row 0),
        recv_count [E_local] int32, and the handle the combine needs.
        ``split_phases``: see the send / receive split note above."""
        split = bool(split_phases)
        recv_x, recv_count, handle, _event, hook = self.buffer.low_latency_dispatch(
            x, topk_idx, self.max_tokens_per_rank, self.num_experts,
            use_fp8=False, async_finish=False, return_recv_hook=split)
        if split:
            self._issue_recv(hook, "low_latency_dispatch")
        if isinstance(recv_x, (tuple, list)):
            recv_x = recv_x[0]
        return recv_x, recv_count, handle

    def dispatch_fp8(self, x: torch.Tensor, topk_idx: torch.Tensor, *,
                     split_phases: bool = False
                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple]:
        """FP8 dispatch: the per-token 128-block cast runs inside the exchange.

        x [rows, hidden] bf16, topk_idx [rows, K] int64 ->
        ``(recv_x, recv_scales, recv_count, handle)`` with

        * ``recv_x``      [E_local, stride, hidden] ``float8_e4m3fn``
        * ``recv_scales`` [E_local, stride, hidden // 128] fp32, whose last two
          dimensions are TRANSPOSED views of a contiguous
          [E_local, hidden // 128, stride] allocation (csrc/deep_ep.cpp
          packs them column-major for TMA). ``recv_scales.transpose(1, 2)`` is
          therefore the contiguous per-expert [hidden // 128, stride] block.
        * ``recv_count``  [E_local] int32 live rows per local expert.

        Rows at or beyond ``recv_count[e]`` hold stale bytes; every consumer
        must be driven by ``recv_count``.

        ``split_phases``: see the send / receive split note above.
        """
        split = bool(split_phases)
        recv, recv_count, handle, _event, hook = self.buffer.low_latency_dispatch(
            x, topk_idx, self.max_tokens_per_rank, self.num_experts,
            use_fp8=True, round_scale=False, use_ue8m0=False,
            async_finish=False, return_recv_hook=split)
        if split:
            self._issue_recv(hook, "low_latency_dispatch")
        if not isinstance(recv, (tuple, list)) or len(recv) != 2:
            raise RuntimeError(
                "DeepEP low_latency_dispatch(use_fp8=True) did not return the "
                f"(values, scales) pair; got {type(recv).__name__}")
        recv_x, recv_scales = recv
        return recv_x, recv_scales, recv_count, handle

    def combine(self, x: torch.Tensor, topk_idx: torch.Tensor,
                topk_weights: torch.Tensor, handle: tuple,
                out: torch.Tensor, *, split_phases: bool = False
                ) -> torch.Tensor:
        """x [E_local, num_ranks * max_tokens_per_rank, hidden] bf16 expert
        outputs (same layout as recv_x) -> out [rows, hidden] bf16 = the
        fp32-accumulated top-k weighted sum on the token's own rank.
        ``split_phases``: see the send / receive split note above."""
        split = bool(split_phases)
        combined, _event, hook = self.buffer.low_latency_combine(
            x, topk_idx, topk_weights, handle,
            async_finish=False, return_recv_hook=split, out=out)
        if split:
            self._issue_recv(hook, "low_latency_combine")
        return combined


_EXCHANGES: dict = {}


def get_low_latency_exchange(group, *, max_tokens_per_rank: int, hidden: int,
                             num_experts: int, num_local_experts: int
                             ) -> "DeepEPLowLatencyExchange":
    """Process-lifetime exchange, shared by every decode-graph build with the
    same shape (a mid-run prefill wave drops and rebuilds the graph pool; the
    NVSHMEM buffer is never destroyed, see ``DeepEPLowLatencyExchange``)."""
    key = (id(group), int(max_tokens_per_rank), int(hidden), int(num_experts),
           int(num_local_experts))
    exchange = _EXCHANGES.get(key)
    if exchange is None:
        exchange = DeepEPLowLatencyExchange(
            group, max_tokens_per_rank=max_tokens_per_rank, hidden=hidden,
            num_experts=num_experts, num_local_experts=num_local_experts)
        _EXCHANGES[key] = exchange
    return exchange
