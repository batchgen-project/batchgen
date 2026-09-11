"""DeepEP low-latency EP exchange for the Kimi-K3 resident-MoE decode graph.

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
"""
import os
from typing import Tuple

import torch


K3_BUILD_MARKER = "K3_BUILD"


def deepep_available() -> bool:
    """True only for the K3 DeepEP build: the upstream wheel imports fine but
    its LL kernels assert on top-16 / hidden 3584 at the first dispatch, so a
    marker file next to ``deep_ep/__init__.py`` (written by the build
    procedure in DECODE_CONCURRENCY_PLAN.md) identifies the patched build."""
    try:
        import deep_ep
    except Exception:  # noqa: BLE001 - absent or unloadable extension
        return False
    return os.path.exists(
        os.path.join(os.path.dirname(deep_ep.__file__), K3_BUILD_MARKER))


class DeepEPLowLatencyExchange:
    """One low-latency ``deep_ep.Buffer`` shared by every MoE layer."""

    def __init__(self, group, *, max_tokens_per_rank: int, hidden: int,
                 num_experts: int, num_local_experts: int):
        import deep_ep

        self.max_tokens_per_rank = int(max_tokens_per_rank)
        self.hidden = int(hidden)
        self.num_experts = int(num_experts)
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
        self.buffer.clean_low_latency_buffer(
            self.max_tokens_per_rank, self.hidden, self.num_experts)

    def dispatch(self, x: torch.Tensor, topk_idx: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor, tuple]:
        """x [rows, hidden] bf16, topk_idx [rows, K] int64 (global expert ids,
        -1 = none) -> recv_x [E_local, num_ranks * max_tokens_per_rank, hidden]
        bf16 (expert-major, each expert's tokens packed from row 0),
        recv_count [E_local] int32, and the handle the combine needs."""
        recv_x, recv_count, handle, _event, _hook = self.buffer.low_latency_dispatch(
            x, topk_idx, self.max_tokens_per_rank, self.num_experts,
            use_fp8=False, async_finish=False, return_recv_hook=False)
        if isinstance(recv_x, (tuple, list)):
            recv_x = recv_x[0]
        return recv_x, recv_count, handle

    def combine(self, x: torch.Tensor, topk_idx: torch.Tensor,
                topk_weights: torch.Tensor, handle: tuple,
                out: torch.Tensor) -> torch.Tensor:
        """x [E_local, num_ranks * max_tokens_per_rank, hidden] bf16 expert
        outputs (same layout as recv_x) -> out [rows, hidden] bf16 = the
        fp32-accumulated top-k weighted sum on the token's own rank."""
        combined, _event, _hook = self.buffer.low_latency_combine(
            x, topk_idx, topk_weights, handle,
            async_finish=False, return_recv_hook=False, out=out)
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
