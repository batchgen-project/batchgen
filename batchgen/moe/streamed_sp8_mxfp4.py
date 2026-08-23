"""Layer-wise streamed SP8 MoE for Kimi-K3 MXFP4 prefill.

TP8 attention replicates token rows inside one node. This path takes a local
1/8 row slice, computes every routed assignment exactly once across the node,
then lets the caller gather rows inside the TP8 group.

Expert ingress is also sharded: local TP rank ``g`` copies only its contiguous
112-expert shard from host, then six node-local all-gathers assemble one full
896-expert layer on every GPU. The full-layer buffers are reused by all MoE
layers; only one layer is resident at a time.

Under the ``hierarchical_gdr`` transport the shard is fetched from host on one
source rank per local TP slot and replicated GPU-to-GPU to the other three
nodes over a dedicated cross-node group before the node-local all-gathers.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist

from batchgen.moe.fused_moe_mxfp4_resident import ResidentEPMXFP4MoELayer
from batchgen.models.moonshotai.kimi_linear.k3.mxfp4_layout import (
    MXFP4_GROUP_SIZE,
    ROUTED_EXPERT_PROJECTIONS,
    routed_expert_module_shapes,
    validate_routed_expert_slot,
)


class StreamedSP8LayerBuffer:
    """One reusable full-layer MXFP4 buffer per GPU.

    The buffer is deliberately single-buffered in HBM.  A second full
    896-expert allocation would cost another ~12 GiB/rank and defeat the
    memory objective, so prefetching layer ``L+1`` is split in two phases:
    ingress into ``local`` (which no compute kernel reads) starts as soon as
    layer ``L`` has been loaded, while the all-gather that overwrites ``full``
    is held back until the model thread confirms every layer-``L`` reader has
    been enqueued.
    """

    def __init__(
        self,
        *,
        core_engine,
        device,
        tp_group,
        tp_rank: int,
        tp_size: int,
        num_experts: int,
        intermediate_size: int,
        latent_size: int,
        acquire_batch_size: int,
        layer_indices=None,
        cross_group=None,
        cross_root=None,
        cross_source: bool = False,
    ):
        if tp_size <= 1 or num_experts % tp_size:
            raise ValueError(
                f"streamed SP8 requires num_experts divisible by TP size; "
                f"got experts={num_experts}, tp_size={tp_size}"
            )
        self.core_engine = core_engine
        self.device = device
        self.tp_group = tp_group
        self.tp_rank = int(tp_rank)
        self.tp_size = int(tp_size)
        self.cross_group = cross_group
        self.cross_root = cross_root
        self.cross_source = bool(cross_source)
        if cross_group is not None and cross_root is None:
            raise ValueError(
                "hierarchical cross-node weight transport needs the global "
                "root rank of the cross-node group"
            )
        # host_rdma: every rank pulls its own shard from its node's host store.
        # hierarchical_gdr: only the group's source rank does; the rest receive
        # the same bytes from the broadcast below.
        self._acquires_from_host = cross_group is None or self.cross_source
        self.num_experts = int(num_experts)
        self.experts_per_rank = self.num_experts // self.tp_size
        self.expert_start = self.tp_rank * self.experts_per_rank
        self.intermediate_size = int(intermediate_size)
        self.latent_size = int(latent_size)
        self.acquire_batch_size = max(1, int(acquire_batch_size))
        self.layer_indices = tuple(int(i) for i in (layer_indices or ()))
        self._next_layer = {
            current: following
            for current, following in zip(
                self.layer_indices, self.layer_indices[1:]
            )
        }
        self.shapes = routed_expert_module_shapes(
            self.intermediate_size, self.latent_size
        )
        self.local = None
        self.full = None
        self._cross_status = None
        self.scale_bf16 = {}
        self._expert_offsets = torch.arange(
            self.num_experts, dtype=torch.int64, device=self.device
        )
        self._prefetch_stream = torch.cuda.Stream(device=self.device)
        # ``self.local`` is the all-gather source and is single-buffered too.
        # The next layer's ingress may only overwrite it once the previous
        # gather has read it.  On the first boundary that gather ran on the
        # caller's compute stream — not the prefetch stream — so ordering
        # within the prefetch stream alone does not cover the hazard.
        self._local_free = torch.cuda.Event()
        self._pending = None

    def _allocate(self):
        if self.local is not None:
            return
        # The first load is normally reached from the worker's inference-mode
        # generate loop, but the same buffers are written by the background
        # prefetch thread after that loop starts.  Allocate them as ordinary
        # tensors so the asynchronous stream can update them outside
        # inference_mode on subsequent layers.
        with torch.inference_mode(False):
            self.local = {
                name: torch.empty(
                    (self.experts_per_rank, *shape),
                    dtype=torch.uint8,
                    device=self.device,
                )
                for name, shape in self.shapes.items()
            }
            self.full = {
                name: torch.empty(
                    (self.num_experts, *shape),
                    dtype=torch.uint8,
                    device=self.device,
                )
                for name, shape in self.shapes.items()
            }
            if self.cross_group is not None:
                self._cross_status = torch.empty(
                    (1,), dtype=torch.int32, device=self.device
                )

    def _acquire_local_shard(self, layer_idx: int, stream=None):
        self._allocate()
        copy_stream = stream or torch.cuda.current_stream(self.device)
        source_error = None
        if self._acquires_from_host:
            names = [
                f"routed_expert_{layer_idx}_{expert_idx}"
                for expert_idx in range(
                    self.expert_start,
                    self.expert_start + self.experts_per_rank,
                )
            ]
            # Hold every slot in this bounded batch until its D2D copy
            # completes.  The ordinary prefill phase may evict earlier expert
            # mappings while a later module is still being acquired.
            phase = "prefill_sp8"
            try:
                for begin in range(0, len(names), self.acquire_batch_size):
                    acquired = []
                    batch = names[begin:begin + self.acquire_batch_size]
                    try:
                        with torch.cuda.stream(copy_stream):
                            for local_offset, module_name in enumerate(
                                batch, start=begin
                            ):
                                weights = self.core_engine.get_weights(
                                    module_name, phase
                                )
                                acquired.append(module_name)
                                validate_routed_expert_slot(
                                    module_name, weights, self.shapes
                                )
                                for tensor_name in self.shapes:
                                    self.local[tensor_name][local_offset].copy_(
                                        weights[tensor_name]
                                    )
                    finally:
                        # The ring slot may be overwritten immediately after
                        # release. Drain all D2D slot->local copies once per
                        # bounded batch, and release every lease even when
                        # validation or a later copy fails.
                        copy_stream.synchronize()
                        for module_name in acquired:
                            self.core_engine.free_weights_buffer(module_name)
            except BaseException as exc:
                if self.cross_group is None:
                    raise
                source_error = exc

        # A source-side Python/storage failure must be announced before peers
        # enter the six large payload broadcasts. Otherwise the other three
        # nodes can wait forever after the source thread has already unwound.
        if not self._broadcast_source_status(
            copy_stream, source_error is None
        ):
            if source_error is not None:
                raise source_error
            raise RuntimeError(
                f"hierarchical_gdr source rank {self.cross_root} failed "
                f"to acquire layer {layer_idx}"
            )

        # Cross-node replication is part of ingress, not of assembly: it writes
        # ``self.local`` only, so it inherits the same WAR event and the same
        # early-ingress overlap as the host copies above, and it necessarily
        # precedes the node-local all-gather that reads ``self.local``.
        self._broadcast_local_shard(copy_stream)

    def _broadcast_source_status(self, stream, source_ok: bool) -> bool:
        """Tell peers whether the source shard is ready for payload ingress."""
        if self.cross_group is None:
            return True
        profile = StreamedSP8MXFP4MoELayer
        with torch.cuda.stream(stream):
            if self.cross_source:
                self._cross_status.fill_(1 if source_ok else 0)
            dist.broadcast(
                self._cross_status, self.cross_root, group=self.cross_group
            )
        ok = bool(self._cross_status.item())
        if profile._prefill_profile_enabled:
            profile._prefill_profile_cross_status_calls += 1
            if not ok:
                profile._prefill_profile_cross_status_failures += 1
        return ok

    def _broadcast_local_shard(self, stream):
        """Replicate the source rank's 112-expert shard to the other nodes.

        No-op for ``host_rdma``, where every rank has already pulled its own
        shard, so that path stays byte-identical.
        """
        if self.cross_group is None:
            return
        profile = StreamedSP8MXFP4MoELayer
        with torch.cuda.stream(stream):
            for tensor_name in self.shapes:
                tensor = self.local[tensor_name]
                dist.broadcast(tensor, self.cross_root, group=self.cross_group)
                if profile._prefill_profile_enabled:
                    profile._prefill_profile_cross_broadcast_calls += 1
                    profile._prefill_profile_cross_broadcast_bytes += (
                        tensor.numel() * tensor.element_size()
                    )
        if profile._prefill_profile_enabled:
            profile._prefill_profile_cross_source = self.cross_source

    def _gather_full_layer(self, stream=None):
        gather_stream = stream or torch.cuda.current_stream(self.device)
        with torch.cuda.stream(gather_stream):
            for tensor_name in self.shapes:
                dist.all_gather_into_tensor(
                    self.full[tensor_name],
                    self.local[tensor_name],
                    group=self.tp_group,
                )
        # Record on whichever stream ran the gathers: ``self.local`` is only
        # free for the next layer's ingress once they have consumed it.
        self._local_free.record(gather_stream)

    def _make_shard(self):
        packed = {}
        scales = {}
        for projection in ROUTED_EXPERT_PROJECTIONS:
            packed_name = projection + ".weight_packed"
            scale_name = projection + ".weight_scale"
            if projection in ("w1", "w3"):
                n_out, k_in = self.intermediate_size, self.latent_size
            else:
                n_out, k_in = self.latent_size, self.intermediate_size

            packed[projection] = self._offline_marlin_packed_view(
                self.full[packed_name], n_out, k_in
            )
            scale_u8 = self.full[scale_name].view(
                self.num_experts, k_in // MXFP4_GROUP_SIZE, n_out
            )
            scale = self.scale_bf16.get(projection)
            if scale is None or tuple(scale.shape) != tuple(scale_u8.shape):
                scale = torch.empty_like(scale_u8, dtype=torch.bfloat16)
                self.scale_bf16[projection] = scale
            scales[projection] = self._expand_e8m0_into(scale_u8, scale)

        self.scale_bf16 = scales
        return SimpleNamespace(
            num_local=self.num_experts,
            N=self.intermediate_size,
            K_latent=self.latent_size,
            gate_B_ptrs=self._ptrs(packed["w1"]),
            gate_scales_ptrs=self._ptrs(scales["w1"]),
            up_B_ptrs=self._ptrs(packed["w3"]),
            up_scales_ptrs=self._ptrs(scales["w3"]),
            down_B_ptrs=self._ptrs(packed["w2"]),
            down_scales_ptrs=self._ptrs(scales["w2"]),
            # Keep all storage alive for the grouped kernel pointer arrays.
            _tensors=(self.full, self.scale_bf16),
        )

    def _wait_pending(self):
        pending = self._pending
        if pending is None:
            return
        pending.thread.join()
        if pending.error is not None:
            raise RuntimeError(
                f"streamed-SP8 prefetch of layer {pending.layer_idx} failed"
            ) from pending.error
        # The prefetch stream owns the full-layer writes.  Make the caller's
        # compute stream wait without a device-wide synchronize.  A teardown
        # cancellation deliberately skips the full-layer write, so it has no
        # ready event to wait on.
        if not pending.cancelled:
            torch.cuda.current_stream(self.device).wait_event(pending.ready)
        self._pending = None

    def begin_prefetch_next(self, layer_idx: int):
        """Start host->HBM ingress for the next MoE layer.

        Phase one of the prefetch.  It is called right after the current layer
        has been loaded and before its grouped MoE is enqueued, because the
        remote ingress dominates the layer and only writes ``self.local``,
        which no compute kernel reads.  The all-gather that overwrites
        ``self.full`` is parked inside the worker thread until
        :meth:`allow_full_overwrite`.

        The call is intentionally non-blocking for the model thread.  The
        C++ ``get_weights`` binding releases the GIL while waiting for the
        daemon, and the dedicated weight process group keeps these gathers
        independent of TP attention collectives.
        """
        next_layer = self._next_layer.get(int(layer_idx))
        if next_layer is None:
            return
        if self._pending is not None:
            raise RuntimeError(
                f"streamed-SP8 prefetch already pending for "
                f"layer {self._pending.layer_idx}"
            )

        pending = SimpleNamespace(
            layer_idx=next_layer,
            error=None,
            ready=torch.cuda.Event(),
            compute_done=torch.cuda.Event(),
            overwrite_allowed=threading.Event(),
            cancelled=False,
            thread=None,
        )

        def run():
            try:
                torch.cuda.set_device(self.device)
                # WAR on ``self.local``: the previous layer's all-gather reads
                # it, and on the first boundary that gather was enqueued on the
                # compute stream.  Order this layer's ingress after it.
                self._prefetch_stream.wait_event(self._local_free)
                # Local-shard acquisition writes ``self.local`` only, which no
                # compute kernel reads, so it overlaps the current layer.
                self._acquire_local_shard(
                    next_layer, stream=self._prefetch_stream
                )
                # ``self.full`` is overwritten below.  Park here until the
                # model thread has enqueued every current-layer reader and
                # recorded the compute-stream event that covers them.
                pending.overwrite_allowed.wait()
                if pending.cancelled:
                    return
                self._prefetch_stream.wait_event(pending.compute_done)
                self._gather_full_layer(stream=self._prefetch_stream)
                pending.ready.record(self._prefetch_stream)
            except BaseException as exc:  # re-raise on the model thread
                pending.error = exc

        pending.thread = threading.Thread(
            target=run,
            name=f"k3-sp8-prefetch-{next_layer}",
            daemon=True,
        )
        self._pending = pending
        pending.thread.start()

    def allow_full_overwrite(self):
        """Phase two: let the pending prefetch overwrite ``self.full``.

        Must be called on the model thread once the current layer's grouped
        MoE, combine and up-projection kernels have all been enqueued.  The
        event is recorded here, at that late point, so it covers exactly the
        in-flight work that still reads ``self.full``.
        """
        pending = self._pending
        if pending is None or pending.overwrite_allowed.is_set():
            return
        pending.compute_done.record(torch.cuda.current_stream(self.device))
        pending.overwrite_allowed.set()

    def close(self):
        """Drain a pending prefetch before phase teardown."""
        pending = self._pending
        if pending is not None and not pending.overwrite_allowed.is_set():
            # A forward that raised between the two phases would otherwise
            # leave the worker parked on the handshake forever.  Cancel the
            # all-gather instead of granting it: the phase is ending and the
            # other ranks may never reach a matching collective.
            pending.cancelled = True
            pending.overwrite_allowed.set()
        self._wait_pending()
        self._prefetch_stream.synchronize()

    @staticmethod
    def _expand_e8m0_into(
        scale_u8: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        # Exact bit construction: E8M0 byte e -> BF16 bits uint16(e) << 7.
        bits = output.view(torch.int16)
        bits.copy_(scale_u8)
        bits.bitwise_left_shift_(7)
        return output

    def _ptrs(self, tensor: torch.Tensor) -> torch.Tensor:
        stride_bytes = tensor[0].numel() * tensor.element_size()
        return self._expert_offsets.mul(stride_bytes).add(tensor.data_ptr())

    @staticmethod
    def _offline_marlin_packed_view(
        packed_u8: torch.Tensor,
        n_out: int,
        k_in: int,
    ) -> torch.Tensor:
        # The ring slot preserves the checkpoint-shaped uint8 ABI, but its
        # linear bytes are already offline-Marlin int32 words.
        return packed_u8.view(torch.int32).reshape(
            packed_u8.shape[0], k_in // 16, n_out * 2
        )

    def load(self, layer_idx: int):
        if self._pending is not None:
            if self._pending.layer_idx != int(layer_idx):
                raise RuntimeError(
                    f"streamed-SP8 expected prefetched layer "
                    f"{self._pending.layer_idx}, requested {layer_idx}"
                )
            self._wait_pending()
        else:
            self._acquire_local_shard(layer_idx)
            self._gather_full_layer()
        return self._make_shard()

class StreamedSP8MXFP4MoELayer:
    """SP8 routed path for one K3 MoE layer."""

    _prefill_profile_enabled = False
    _prefill_profile_forward_calls = 0
    _prefill_profile_input_rows = 0
    _prefill_profile_routed_assignments = 0
    _prefill_profile_grouped_chunks = 0
    _prefill_profile_active_experts = 0
    _prefill_profile_wall_s = 0.0
    # Cross-node weight transport evidence (hierarchical_gdr). Counted by
    # StreamedSP8LayerBuffer, which owns the broadcasts; zero/False proves the
    # host_rdma path ran no cross-node collective on this rank.
    _prefill_profile_cross_broadcast_calls = 0
    _prefill_profile_cross_broadcast_bytes = 0
    _prefill_profile_cross_source = False
    _prefill_profile_cross_status_calls = 0
    _prefill_profile_cross_status_failures = 0

    @classmethod
    def reset_prefill_profile(cls, enabled: bool) -> None:
        cls._prefill_profile_enabled = bool(enabled)
        cls._prefill_profile_forward_calls = 0
        cls._prefill_profile_input_rows = 0
        cls._prefill_profile_routed_assignments = 0
        cls._prefill_profile_grouped_chunks = 0
        cls._prefill_profile_active_experts = 0
        cls._prefill_profile_wall_s = 0.0
        cls._prefill_profile_cross_broadcast_calls = 0
        cls._prefill_profile_cross_broadcast_bytes = 0
        cls._prefill_profile_cross_source = False
        cls._prefill_profile_cross_status_calls = 0
        cls._prefill_profile_cross_status_failures = 0

    @classmethod
    def prefill_profile_snapshot(cls) -> dict:
        return {
            "enabled": cls._prefill_profile_enabled,
            "forward_calls": cls._prefill_profile_forward_calls,
            "input_rows": cls._prefill_profile_input_rows,
            "routed_assignments": cls._prefill_profile_routed_assignments,
            "grouped_chunks": cls._prefill_profile_grouped_chunks,
            "active_experts": cls._prefill_profile_active_experts,
            "wall_s": cls._prefill_profile_wall_s,
            "cross_broadcast_calls": (
                cls._prefill_profile_cross_broadcast_calls
            ),
            "cross_broadcast_bytes": (
                cls._prefill_profile_cross_broadcast_bytes
            ),
            "cross_source": cls._prefill_profile_cross_source,
            "cross_status_calls": cls._prefill_profile_cross_status_calls,
            "cross_status_failures": (
                cls._prefill_profile_cross_status_failures
            ),
        }

    def __init__(
        self,
        *,
        layer_idx: int,
        buffer: StreamedSP8LayerBuffer,
        down_proj,
        norm,
        up_proj,
        chunk_rows: int = 256,
        post_chunk_rows: int = 8192,
    ):
        self.layer_idx = int(layer_idx)
        self.buffer = buffer
        self.down_proj = down_proj
        self.norm = norm
        self.up_proj = up_proj
        self.chunk_rows = int(chunk_rows)
        self.post_chunk_rows = int(post_chunk_rows)

    def forward(self, x: torch.Tensor, gate) -> torch.Tensor:
        T, H = x.shape
        cls = type(self)
        profile = cls._prefill_profile_enabled
        if profile:
            cls._prefill_profile_forward_calls += 1
            cls._prefill_profile_input_rows += T
        # Every TP rank must load the layer and enter all six weight
        # all-gathers, including a rank that owns zero rows after the
        # deterministic row split. Returning before ``buffer.load`` lets the
        # non-empty ranks enter the collective alone and deadlocks the group
        # on short or imbalanced microbatches.
        shard = self.buffer.load(self.layer_idx)
        # Start the next layer's remote ingress before this layer's kernels are
        # enqueued.  It writes ``buffer.local`` only; the all-gather that
        # overwrites ``buffer.full`` stays parked until the release below.
        self.buffer.begin_prefetch_next(self.layer_idx)
        if T == 0:
            # No kernel on this rank reads ``buffer.full``, so the overwrite is
            # already safe -- but the gather is collective and every rank must
            # still enter it.
            self.buffer.allow_full_overwrite()
            return x.new_zeros((0, H))

        profile_start = time.perf_counter() if profile else None
        helper = ResidentEPMXFP4MoELayer(
            self.layer_idx,
            shard,
            self.down_proj,
            self.norm,
            self.up_proj,
            world_size=1,
            expert_start=0,
        )
        helper.compact_dispatch = True
        gate_out = gate(x.view(T, 1, H))
        topk_idx = gate_out[0].reshape(T, -1).to(torch.int32)
        topk_weight = gate_out[1].reshape(T, -1)
        top_k = topk_idx.shape[-1]
        if profile:
            cls._prefill_profile_routed_assignments += int(topk_idx.numel())
            cls._prefill_profile_active_experts += int(
                torch.unique(topk_idx).numel()
            )
        x_latent = self.down_proj(x).contiguous()
        latent_size = shard.K_latent
        combined = x_latent.new_empty((T, latent_size))

        if profile:
            cls._prefill_profile_grouped_chunks += (
                T + self.chunk_rows - 1
            ) // self.chunk_rows
        for start in range(0, T, self.chunk_rows):
            end = min(start + self.chunk_rows, T)
            count = end - start
            expert_out, topk_pos = helper._expert_path(
                x_latent[start:end],
                topk_idx[start:end],
                count,
            )
            combined[start:end].copy_(
                helper._combine_fp32(
                    expert_out,
                    topk_pos,
                    topk_weight[start:end],
                    count,
                    latent_size,
                    top_k,
                ).to(torch.bfloat16)
            )

        output = x.new_empty((T, H))
        for start in range(0, T, self.post_chunk_rows):
            end = min(start + self.post_chunk_rows, T)
            y = combined[start:end]
            if self.norm is not None:
                y = self.norm(y)
            output[start:end].copy_(self.up_proj(y))
        # Every kernel that reads ``buffer.full`` has now been enqueued, so the
        # prefetch started above may assemble the next layer into it.
        self.buffer.allow_full_overwrite()
        if profile:
            cls._prefill_profile_wall_s += (
                time.perf_counter() - profile_start
            )
        return output
