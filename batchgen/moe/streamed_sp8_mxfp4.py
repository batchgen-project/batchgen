"""Layer-wise streamed SP8 MoE for Kimi-K3 MXFP4 prefill.

TP8 attention replicates token rows inside one node, and the eight TP slots of
a node hold DISJOINT 112-expert shards of the layer.  The MoE is therefore
expert-parallel inside the node and communicates activations nowhere else:

    local row slice -> router + down-proj on the padded slice
    -> node-local all_gather(latent, topk idx, topk weight)
    -> this rank's 112 experts over ALL node rows (non-owned -> -1)
    -> FP32 combine -> node-local reduce_scatter(SUM) -> this rank's rows
    -> norm + up-proj -> caller's all_gather_rows restores replicated hidden

Because the eight shards partition the 896 experts, every routed assignment of
the node is computed on exactly one rank, and the FP32 reduce-scatter sums the
eight disjoint partial latents.  The gathered latent is 3584-d, half the 7168
hidden, and the collectives never leave the node.

Expert ingress is sharded the same way: local TP rank ``g`` copies only its
contiguous 112-expert shard from host into ``local``, and phase two of the
prefetch copies that same shard into the ``compute`` buffer the grouped kernels
read.  Nothing all-gathers weights: a rank never holds an expert it does not
own, so one layer costs ~1.5 GiB/rank instead of ~12 GiB.

Under the ``hierarchical_gdr`` transport the shard is fetched from host on one
source rank per local TP slot and replicated GPU-to-GPU to the other three
nodes over a dedicated cross-node group before that local-to-compute copy.
"""

from __future__ import annotations

import math
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


# Match the C++ ``prefill_sp8`` ring-lease bound. Timing out the Python launch
# handoff earlier would strand peers in a payload broadcast while a cold source
# was still legitimately waiting for its host shard.
PREFETCH_HANDOFF_TIMEOUT_S = 300.0


def _profile_mark(marks):
    """Append one E0 boundary event, recorded on the current stream.

    Timing events are only ever RECORDED, never queried, in the layer and
    ingress paths: an ``elapsed_time`` or ``synchronize`` per layer would park
    the host on the very overlap this path exists to create.  The pairs are
    read exactly once, in :meth:`StreamedSP8MXFP4MoELayer.prefill_profile_snapshot`.
    """
    event = torch.cuda.Event(enable_timing=True)
    event.record()
    marks.append(event)


class StreamedSP8LayerBuffer:
    """Two reusable 112-expert MXFP4 shard buffers per GPU.

    The two buffers are roles, not a ping-pong pair: ``local`` is the ingress
    target that host copies and the cross-node broadcast write, ``compute``
    is the one the grouped marlin kernels read through the pointer arrays in
    :meth:`_make_shard`.  Both hold ``experts_per_rank`` experts, never the
    whole layer, so the pair costs about what one full layer used to.

    Prefetching layer ``L+1`` is split in two phases against that split:
    ingress into ``local`` starts as soon as layer ``L`` has been loaded, while
    the local-to-compute copy that overwrites ``compute`` is held back until the
    model thread confirms every layer-``L`` reader has been enqueued.
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
        self.compute = None
        self._cross_status = None
        self.scale_bf16 = {}
        self._expert_offsets = torch.arange(
            self.experts_per_rank, dtype=torch.int64, device=self.device
        )
        self._prefetch_stream = torch.cuda.Stream(device=self.device)
        self._status_stream = (
            torch.cuda.Stream(device=self.device)
            if self.cross_group is not None
            else None
        )
        # ``self.local`` is the assembly source and is single-buffered too.
        # The next layer's ingress may only overwrite it once the previous
        # copy has read it.  On the first boundary that copy ran on the
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
            self.compute = {
                name: torch.empty(
                    (self.experts_per_rank, *shape),
                    dtype=torch.uint8,
                    device=self.device,
                )
                for name, shape in self.shapes.items()
            }
            if self.cross_group is not None:
                self._cross_status = torch.empty(
                    (1,), dtype=torch.int32, device=self.device
                )

    def _acquire_local_shard(
        self,
        layer_idx: int,
        stream=None,
        cross_launch_gate=None,
        cross_launch_callback=None,
        profile_host=None,
    ):
        self._allocate()
        copy_stream = stream or torch.cuda.current_stream(self.device)
        profile = StreamedSP8MXFP4MoELayer
        enabled = profile._prefill_profile_enabled
        publish_profile_host = enabled and profile_host is None
        if enabled and profile_host is None:
            profile_host = SimpleNamespace(
                layer_idx=int(layer_idx),
                acquisition_start_s=None,
                acquisition_end_s=None,
                gate_open_s=None,
                order_wait_begin_s=None,
                order_wait_end_s=None,
                broadcast_enqueue_s=None,
            )
        source_error = None
        if self._acquires_from_host:
            if enabled:
                profile_host.acquisition_start_s = time.perf_counter()
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
                        # Both transports release each bounded batch as soon as
                        # its copies have landed in ``self.local``, so the ring
                        # only ever has to hold ``acquire_batch_size`` slots and
                        # the next batch can reuse the ones just freed.
                        copy_stream.synchronize()
                        for module_name in acquired:
                            self.core_engine.free_weights_buffer(module_name)
            except BaseException as exc:
                if self.cross_group is None:
                    raise
                source_error = exc
            if enabled and source_error is None:
                profile_host.acquisition_end_s = time.perf_counter()

        # A source-side Python/storage failure must be announced before peers
        # enter the six large payload broadcasts. Otherwise the other three
        # nodes can wait forever after the source thread has already unwound.
        #
        # Host/ring acquisition above is free to run as early as this thread
        # starts, but the cross-node communicator is not: the model thread
        # issues the layer's three TP8 gathers first and only then opens this
        # gate, so every GPU keeps one deterministic TP8 -> cross-node launch
        # order.  A cancelled gate means the phase is tearing down; issuing the
        # status broadcast then would park this thread against peers that will
        # never join it.
        if cross_launch_gate is not None and not cross_launch_gate():
            return
        if not self._broadcast_source_status(
            self._status_stream or copy_stream, source_error is None
        ):
            if source_error is not None:
                raise source_error
            raise RuntimeError(
                f"hierarchical_gdr source rank {self.cross_root} failed "
                f"to acquire layer {layer_idx}"
            )

        # Cross-node replication is part of ingress, not of assembly: it
        # writes ``self.local`` only, so it inherits the same WAR event and
        # the same early-ingress overlap as the host copies above, and it
        # necessarily precedes the local-to-compute copy that reads
        # ``self.local``.
        self._broadcast_local_shard(copy_stream)
        if enabled and self.cross_group is not None:
            profile_host.broadcast_enqueue_s = time.perf_counter()
        if cross_launch_callback is not None:
            cross_launch_callback()
        if publish_profile_host:
            profile._prefill_profile_host_records.append(profile_host)

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
            # ``stream`` is a non-default pool stream on the background
            # prefetch thread. Keep the scalar read in this context so its D2H
            # copy is ordered after the status broadcast rather than racing a
            # stale or uninitialized value on the thread's default stream.
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
        enabled = profile._prefill_profile_enabled
        span = []
        with torch.cuda.stream(stream):
            if enabled:
                _profile_mark(span)
            for tensor_name in self.shapes:
                tensor = self.local[tensor_name]
                dist.broadcast(tensor, self.cross_root, group=self.cross_group)
                if enabled:
                    profile._prefill_profile_cross_broadcast_calls += 1
                    profile._prefill_profile_cross_broadcast_bytes += (
                        tensor.numel() * tensor.element_size()
                    )
            if enabled:
                _profile_mark(span)
        if enabled:
            # Published only once BOTH ends are recorded, so a broadcast that
            # raised leaves no half pair behind for the snapshot to read.
            profile._prefill_profile_broadcast_spans.append(tuple(span))
            profile._prefill_profile_cross_source = self.cross_source

    def _assemble_compute_shard(self, stream=None):
        """Phase two: publish the ingressed shard to the compute buffer.

        A same-rank device-to-device copy, NOT a node-local weight all-gather.
        The eight TP slots of a node own disjoint expert ranges and the MoE is
        expert-parallel over them, so a rank never needs another rank's
        weights; gathering all 896 experts onto every GPU was the ~12 GiB and
        896-tiny-GEMM cost this path exists to avoid.
        """
        copy_stream = stream or torch.cuda.current_stream(self.device)
        with torch.cuda.stream(copy_stream):
            for tensor_name in self.shapes:
                self.compute[tensor_name].copy_(self.local[tensor_name])
        # Record on whichever stream ran the copies: ``self.local`` is only
        # free for the next layer's ingress once they have consumed it.
        self._local_free.record(copy_stream)

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
                self.compute[packed_name], n_out, k_in
            )
            scale_u8 = self.compute[scale_name].view(
                self.experts_per_rank, k_in // MXFP4_GROUP_SIZE, n_out
            )
            scale = self.scale_bf16.get(projection)
            if scale is None or tuple(scale.shape) != tuple(scale_u8.shape):
                scale = torch.empty_like(scale_u8, dtype=torch.bfloat16)
                self.scale_bf16[projection] = scale
            scales[projection] = self._expand_e8m0_into(scale_u8, scale)

        self.scale_bf16 = scales
        return SimpleNamespace(
            num_local=self.experts_per_rank,
            N=self.intermediate_size,
            K_latent=self.latent_size,
            gate_B_ptrs=self._ptrs(packed["w1"]),
            gate_scales_ptrs=self._ptrs(scales["w1"]),
            up_B_ptrs=self._ptrs(packed["w3"]),
            up_scales_ptrs=self._ptrs(scales["w3"]),
            down_B_ptrs=self._ptrs(packed["w2"]),
            down_scales_ptrs=self._ptrs(scales["w2"]),
            # Keep all storage alive for the grouped kernel pointer arrays.
            _tensors=(self.compute, self.scale_bf16),
        )

    def _wait_pending(self):
        pending = self._pending
        if pending is None:
            return
        pending.thread.join(PREFETCH_HANDOFF_TIMEOUT_S)
        if pending.thread.is_alive():
            raise RuntimeError(
                f"streamed-SP8 prefetch of layer {pending.layer_idx} did not "
                f"finish within {PREFETCH_HANDOFF_TIMEOUT_S:.0f}s"
            )
        # Once the thread has terminated this pending object can never become
        # usable again. Clear it before surfacing an error so phase teardown
        # does not re-raise the same failure and retain the GPU buffers.
        self._pending = None
        if pending.error is not None:
            raise RuntimeError(
                f"streamed-SP8 prefetch of layer {pending.layer_idx} failed"
            ) from pending.error
        # The prefetch stream owns the compute-buffer writes.  Make the caller's
        # compute stream wait without a device-wide synchronize.  A teardown
        # cancellation deliberately skips that write, so it has no ready event
        # to wait on.
        if not pending.cancelled:
            torch.cuda.current_stream(self.device).wait_event(pending.ready)

    def begin_prefetch_next(self, layer_idx: int):
        """Start host->HBM ingress for the next MoE layer.

        Phase one of the prefetch. Both transports call it right after the
        current layer is loaded, so host and ring-to-local acquisition starts
        at the earliest possible point. It precedes grouped MoE and writes only
        ``self.local``, which no compute kernel reads. Hierarchical GDR parks
        the worker thread on :meth:`allow_cross_launch` before its cross-node
        collectives so the model thread still issues the TP8 activation gathers
        first, preserving one global host launch order. The copy that
        overwrites ``self.compute`` is parked until
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
            # NCCL_LAUNCH_ORDER_IMPLICIT preserves overlap only when every
            # communicator launch has one deterministic host order. The gate
            # below proves the TP8 gathers were issued before any cross-node
            # collective; this handoff proves all cross-node broadcasts have
            # been issued before the model thread is allowed to issue the TP8
            # reduce-scatter.
            cross_launch_allowed=(
                threading.Event() if self.cross_group is not None else None
            ),
            cross_launch_enqueued=(
                threading.Event() if self.cross_group is not None else None
            ),
            cross_ingress_done=(
                torch.cuda.Event() if self.cross_group is not None else None
            ),
            cross_ingress_enqueued=(
                threading.Event() if self.cross_group is not None else None
            ),
            overwrite_allowed=threading.Event(),
            cancelled=False,
            thread=None,
            profile_host=(
                SimpleNamespace(
                    layer_idx=int(next_layer),
                    acquisition_start_s=None,
                    acquisition_end_s=None,
                    gate_open_s=None,
                    order_wait_begin_s=None,
                    order_wait_end_s=None,
                    broadcast_enqueue_s=None,
                )
                if StreamedSP8MXFP4MoELayer._prefill_profile_enabled
                else None
            ),
        )

        def open_cross_launch():
            """Hold phase one's collectives until the model thread's gate."""
            if not pending.cross_launch_allowed.wait(
                PREFETCH_HANDOFF_TIMEOUT_S
            ):
                raise RuntimeError(
                    f"streamed-SP8 prefetch of layer {next_layer} was not "
                    "released for cross-node launch within "
                    f"{PREFETCH_HANDOFF_TIMEOUT_S:.0f}s"
                )
            return not pending.cancelled

        def run():
            try:
                torch.cuda.set_device(self.device)
                # WAR on ``self.local``: the previous layer's assembly copy
                # reads it, and on the first boundary that copy was enqueued on
                # the compute stream.  Order this layer's ingress after it.
                self._prefetch_stream.wait_event(self._local_free)
                # Local-shard acquisition writes ``self.local`` only, which no
                # compute kernel reads, so it overlaps the current layer.  Its
                # host phase starts here; its cross-node phase waits on the
                # gate.
                self._acquire_local_shard(
                    next_layer,
                    stream=self._prefetch_stream,
                    cross_launch_gate=(
                        open_cross_launch
                        if pending.cross_launch_allowed is not None
                        else None
                    ),
                    cross_launch_callback=(
                        pending.cross_launch_enqueued.set
                        if pending.cross_launch_enqueued is not None
                        else None
                    ),
                    profile_host=pending.profile_host,
                )
                if pending.cross_ingress_done is not None:
                    # Record after the six payload broadcasts on the same
                    # stream. The model thread waits on this event before it
                    # re-enters TP8, keeping the orthogonal communicator
                    # families from executing concurrently while retaining
                    # early host acquisition.
                    pending.cross_ingress_done.record(self._prefetch_stream)
                    pending.cross_ingress_enqueued.set()
                # ``self.compute`` is overwritten below.  Park here until the
                # model thread has enqueued every current-layer reader and
                # recorded the compute-stream event that covers them.
                pending.overwrite_allowed.wait()
                if pending.cancelled:
                    return
                self._prefetch_stream.wait_event(pending.compute_done)
                self._assemble_compute_shard(stream=self._prefetch_stream)
                if pending.profile_host is not None:
                    StreamedSP8MXFP4MoELayer._prefill_profile_host_records.append(
                        pending.profile_host
                    )
                pending.ready.record(self._prefetch_stream)
            except BaseException as exc:  # re-raise on the model thread
                pending.error = exc
            finally:
                # A thread that has terminated will never issue another cross
                # collective, whether it failed, was cancelled or completed
                # normally.  Release the handoff unconditionally so the model
                # thread can never be parked by a dead worker.
                if pending.cross_launch_enqueued is not None:
                    pending.cross_launch_enqueued.set()
                if pending.cross_ingress_enqueued is not None:
                    pending.cross_ingress_enqueued.set()

        pending.thread = threading.Thread(
            target=run,
            name=f"k3-sp8-prefetch-{next_layer}",
            daemon=True,
        )
        self._pending = pending
        pending.thread.start()

    def allow_cross_launch(self):
        """Let the pending prefetch issue its cross-node collectives.

        Must be called on the model thread once this layer's three TP8
        activation gathers have been issued.  Until then the worker thread has
        already pulled its shard from the host store or the source ring, but
        has issued neither the status broadcast nor the six payload
        broadcasts, so ``NCCL_LAUNCH_ORDER_IMPLICIT=1`` still sees one
        deterministic TP8 -> cross-node order on every GPU.

        Host-RDMA has no cross-node communicator and is a no-op here.
        """
        pending = self._pending
        if pending is None or pending.cross_launch_allowed is None:
            return
        if (
            pending.profile_host is not None
            and pending.profile_host.gate_open_s is None
        ):
            pending.profile_host.gate_open_s = time.perf_counter()
        pending.cross_launch_allowed.set()

    def order_tp_collective_after_cross_launch(self):
        """Issue the next TP8 collective after cross-node ingress.

        The host gate still proves every cross-node call was issued after the
        preceding TP8 gathers. The CUDA event additionally keeps the payload
        broadcasts from executing concurrently with the following TP8
        reduce-scatter.

        Host-RDMA has no cross-node GPU collective and is a no-op here.
        """
        pending = self._pending
        if self.cross_group is None or pending is None:
            return
        if pending.profile_host is not None:
            pending.profile_host.order_wait_begin_s = time.perf_counter()
        if not pending.cross_launch_enqueued.wait(PREFETCH_HANDOFF_TIMEOUT_S):
            raise RuntimeError(
                f"streamed-SP8 prefetch of layer {pending.layer_idx} did not "
                "issue its cross-node collectives within "
                f"{PREFETCH_HANDOFF_TIMEOUT_S:.0f}s"
            )
        if pending.error is not None:
            # Join and clear the failed object before surfacing it. Otherwise
            # phase teardown would observe the same poisoned pending state and
            # raise a second time before releasing the layer buffers.
            self._wait_pending()
        if not pending.cross_ingress_enqueued.wait(
            PREFETCH_HANDOFF_TIMEOUT_S
        ):
            raise RuntimeError(
                f"streamed-SP8 prefetch of layer {pending.layer_idx} did not "
                "enqueue its cross-node completion event within "
                f"{PREFETCH_HANDOFF_TIMEOUT_S:.0f}s"
            )
        if pending.error is not None:
            self._wait_pending()
        torch.cuda.current_stream(self.device).wait_event(
            pending.cross_ingress_done
        )
        if pending.profile_host is not None:
            pending.profile_host.order_wait_end_s = time.perf_counter()

    def allow_full_overwrite(self):
        """Phase two: let the pending prefetch overwrite ``self.compute``.

        Must be called on the model thread once the current layer's grouped
        MoE, its FP32 combine and the node-local reduce-scatter that consumes
        them have all been enqueued.  The event is recorded here, at that late
        point, so it covers exactly the in-flight work that still reads
        ``self.compute``.
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
            # A forward that raised between the phases would otherwise leave
            # the worker parked on a handshake forever.  ``allow_cross_launch``
            # always precedes ``allow_full_overwrite`` in a forward, so an
            # ungranted overwrite covers an ungranted cross launch too.  Cancel
            # both instead of granting them: the phase is ending, and neither
            # the peers' broadcasts nor the ingress the assembly follows may
            # ever complete on the other ranks.
            pending.cancelled = True
            if pending.cross_launch_allowed is not None:
                pending.cross_launch_allowed.set()
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
            self._assemble_compute_shard()
        return self._make_shard()

class StreamedSP8MXFP4MoELayer:
    """SP8 routed path for one K3 MoE layer."""

    _prefill_profile_enabled = False
    _prefill_profile_forward_calls = 0
    # Real rows this rank owns after the node row split -- NOT multiplied by
    # top_k, because expert ownership is skewed and a rank computes the
    # assignments of its 112 experts over ALL node rows, not of its own rows.
    _prefill_profile_input_rows = 0
    # Assignments this rank actually computed, i.e. routed to one of its own
    # ``expert_shard_size`` experts, padding excluded.  The eight shards of a
    # node partition the 896 experts, so summing this over the node's ranks
    # must equal ``node_routed_assignments`` exactly -- no duplicates, no gaps.
    _prefill_profile_routed_assignments = 0
    # num_rows * top_k for the node, replicated identically on all eight ranks.
    _prefill_profile_node_routed_assignments = 0
    _prefill_profile_expert_shard_size = 0
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
    # E0 span evidence.  Every entry is a COMPLETE record: a layer or a
    # broadcast that returned early, was cancelled or raised publishes nothing,
    # so the snapshot never has to interpret a half pair.
    #
    # ``_prefill_profile_layer_marks`` holds one nine-event boundary sequence
    # per non-empty layer, recorded on the compute stream:
    #   0 layer start       1/2 activation gathers start/end
    #   3/4 grouped + FP32 combine start/end
    #   5 post-ingress-wait 6/7 reduce-scatter start/end
    #   8 post-MoE layer end
    _prefill_profile_layer_marks = []
    # One (start, end) pair per completed six-tensor payload broadcast, on the
    # ingress stream.  Empty on host_rdma, which runs no cross-node collective.
    _prefill_profile_broadcast_spans = []
    # Host ``perf_counter`` timestamps for one ingress. The async record is
    # published only after assembly, by which point the model thread has also
    # completed the launch-order handoff. The first synchronous layer has no
    # gate/order-wait timestamps but still records acquisition and enqueue.
    _prefill_profile_host_records = []

    @classmethod
    def reset_prefill_profile(cls, enabled: bool) -> None:
        cls._prefill_profile_enabled = bool(enabled)
        cls._prefill_profile_forward_calls = 0
        cls._prefill_profile_input_rows = 0
        cls._prefill_profile_routed_assignments = 0
        cls._prefill_profile_node_routed_assignments = 0
        cls._prefill_profile_expert_shard_size = 0
        cls._prefill_profile_grouped_chunks = 0
        cls._prefill_profile_active_experts = 0
        cls._prefill_profile_wall_s = 0.0
        cls._prefill_profile_cross_broadcast_calls = 0
        cls._prefill_profile_cross_broadcast_bytes = 0
        cls._prefill_profile_cross_source = False
        cls._prefill_profile_cross_status_calls = 0
        cls._prefill_profile_cross_status_failures = 0
        # Replaced, not cleared in place: a snapshot already handed to the
        # caller keeps the records it aggregated from.
        cls._prefill_profile_layer_marks = []
        cls._prefill_profile_broadcast_spans = []
        cls._prefill_profile_host_records = []

    @staticmethod
    def _profile_span_stats(samples_ms) -> dict:
        """count / sum / p50 / p95 of one span family, in milliseconds."""
        samples = sorted(samples_ms)
        count = len(samples)
        if count == 0:
            return {"count": 0, "sum_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}

        def percentile(fraction):
            # Nearest-rank: the reported value is always an observed sample and
            # is reproducible across ranks without an interpolation rule.
            return samples[min(count - 1, math.ceil(fraction * count) - 1)]

        return {
            "count": count,
            "sum_ms": float(sum(samples)),
            "p50_ms": float(percentile(0.50)),
            "p95_ms": float(percentile(0.95)),
        }

    @classmethod
    def prefill_profile_snapshot(cls) -> dict:
        def scalar(value):
            # Routed-assignment evidence is accumulated on the GPU so the
            # profiler does not synchronize once per layer. Prefill has already
            # completed when this snapshot is emitted, so one final scalar read
            # preserves exact accounting without perturbing the measured path.
            return (
                int(value.item())
                if isinstance(value, torch.Tensor)
                else value
            )

        marks = cls._prefill_profile_layer_marks
        broadcasts = cls._prefill_profile_broadcast_spans
        def boundary_ms(first, second):
            return [
                record[first].elapsed_time(record[second])
                for record in marks
            ]

        def host_span_ms(start_name, end_name):
            return [
                (getattr(record, end_name) - getattr(record, start_name))
                * 1000.0
                for record in cls._prefill_profile_host_records
                if getattr(record, start_name) is not None
                and getattr(record, end_name) is not None
            ]

        return {
            "enabled": cls._prefill_profile_enabled,
            "forward_calls": cls._prefill_profile_forward_calls,
            "input_rows": cls._prefill_profile_input_rows,
            "routed_assignments": scalar(
                cls._prefill_profile_routed_assignments
            ),
            "node_routed_assignments": (
                cls._prefill_profile_node_routed_assignments
            ),
            "expert_shard_size": cls._prefill_profile_expert_shard_size,
            "grouped_chunks": cls._prefill_profile_grouped_chunks,
            "active_experts": scalar(cls._prefill_profile_active_experts),
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
            "broadcast_execution": cls._profile_span_stats(
                [start.elapsed_time(end) for start, end in broadcasts]
            ),
            "fence_stall": cls._profile_span_stats(boundary_ms(4, 5)),
            "grouped_execution": cls._profile_span_stats(boundary_ms(3, 4)),
            "reduce_scatter": cls._profile_span_stats(boundary_ms(6, 7)),
            "post_moe": cls._profile_span_stats(boundary_ms(7, 8)),
            "host_acquisition": cls._profile_span_stats(
                host_span_ms("acquisition_start_s", "acquisition_end_s")
            ),
            "host_launch_handoff": cls._profile_span_stats(
                host_span_ms("order_wait_begin_s", "order_wait_end_s")
            ),
            "host_timestamp_counts": {
                "ingress_records": len(cls._prefill_profile_host_records),
                "gate_open": sum(
                    record.gate_open_s is not None
                    for record in cls._prefill_profile_host_records
                ),
                "broadcast_enqueue": sum(
                    record.broadcast_enqueue_s is not None
                    for record in cls._prefill_profile_host_records
                ),
            },
        }

    def __init__(
        self,
        *,
        layer_idx: int,
        buffer: StreamedSP8LayerBuffer,
        down_proj,
        norm,
        up_proj,
        chunk_rows: int = 2048,
        post_chunk_rows: int = 8192,
    ):
        self.layer_idx = int(layer_idx)
        self.buffer = buffer
        self.down_proj = down_proj
        self.norm = norm
        self.up_proj = up_proj
        self.chunk_rows = int(chunk_rows)
        self.post_chunk_rows = int(post_chunk_rows)

    def forward(self, x: torch.Tensor, gate, num_rows: int) -> torch.Tensor:
        """Run this rank's expert shard over the whole node's rows.

        ``x`` is this rank's contiguous slice of the node's replicated rows and
        ``num_rows`` is the node's row count before that split -- the caller
        knows it, and it is identical on all eight ranks, so it decides the
        padded ``ntp`` stride without an extra collective.
        """
        T, H = x.shape
        cls = type(self)
        profile = cls._prefill_profile_enabled
        buffer = self.buffer
        num_rows = int(num_rows)
        if profile:
            cls._prefill_profile_forward_calls += 1
            cls._prefill_profile_input_rows += T
        # Every TP rank must load the layer and enter the ingress collectives,
        # including a rank that owns zero rows after the deterministic row
        # split. Returning before ``buffer.load`` lets the non-empty ranks enter
        # the cross-node broadcast alone and deadlocks the group on short or
        # imbalanced microbatches.
        shard = buffer.load(self.layer_idx)
        # Both transports start ingress at the earliest point, so the host and
        # ring-to-local acquisition of the next layer overlaps this one from
        # here. Hierarchical GDR holds only its cross-node collectives, until
        # the gate released after the TP8 activation gathers below, so every
        # GPU still orders TP8 -> cross-node.
        buffer.begin_prefetch_next(self.layer_idx)
        # ``num_rows`` is the node-wide count, so this branch is taken by all
        # eight ranks together or by none.  Keying it on the per-rank ``T``
        # instead would desynchronize the node-local collectives below.
        if num_rows == 0:
            # No activation collective exists to order against, but the gate
            # must still open or the worker thread would never issue -- and
            # never hand off -- the cross-node broadcasts its peers enter.
            buffer.allow_cross_launch()
            buffer.order_tp_collective_after_cross_launch()
            buffer.allow_full_overwrite()
            return x.new_zeros((0, H))

        profile_start = time.perf_counter() if profile else None
        # E0 boundaries for THIS layer, recorded on the compute stream and
        # never waited on here.  Published as one record at the end, so a layer
        # that raises leaves no partial sequence behind.  A zero-row node
        # returned above and contributes no record at all.
        marks = [] if profile else None
        if profile:
            _profile_mark(marks)
        tp_size = buffer.tp_size
        tp_group = buffer.tp_group
        expert_start = buffer.expert_start
        num_local = buffer.experts_per_rank
        helper = ResidentEPMXFP4MoELayer(
            self.layer_idx,
            shard,
            self.down_proj,
            self.norm,
            self.up_proj,
            world_size=1,
            expert_start=expert_start,
        )
        helper.compact_dispatch = True

        # --- router + down-proj on this rank's OWN rows, padded to ntp ---
        # The router is row-local and the down-proj is per token, so running
        # them before the gather costs 1/8 of the work and makes the gathered
        # buffers line up with the caller's balanced row split.
        ntp = (num_rows + tp_size - 1) // tp_size
        padded = x.new_zeros((ntp, H))
        if T > 0:
            padded[:T].copy_(x)
        gate_out = gate(padded.view(ntp, 1, H))
        topk_idx = gate_out[0].reshape(ntp, -1).to(torch.int32)
        topk_weight = gate_out[1].reshape(ntp, -1).contiguous()
        top_k = topk_idx.shape[-1]
        if T < ntp:
            # The router returns real expert ids for a zero hidden. Mark the pad
            # rows unroutable: -1 is below every rank's ``expert_start``, so no
            # shard dispatches them, and the owned-assignment accounting below
            # stays exact instead of counting phantom work.
            topk_idx[T:].fill_(-1)
            topk_weight[T:].zero_()
        x_latent = self.down_proj(padded).contiguous()
        latent_size = shard.K_latent

        # --- node-local gather of the LATENT + routing (never the hidden) ---
        num_node_rows = tp_size * ntp
        all_latent = x_latent.new_empty((num_node_rows, latent_size))
        all_idx = topk_idx.new_empty((num_node_rows, top_k))
        all_weight = topk_weight.new_empty((num_node_rows, top_k))
        if profile:
            _profile_mark(marks)
        dist.all_gather_into_tensor(all_latent, x_latent, group=tp_group)
        dist.all_gather_into_tensor(all_idx, topk_idx, group=tp_group)
        dist.all_gather_into_tensor(all_weight, topk_weight, group=tp_group)
        if profile:
            _profile_mark(marks)
        # All three TP8 gather calls have now been issued from the model
        # thread. The prefetch thread, whose host ingress has been running
        # since ``begin_prefetch_next`` above, may issue its cross-node
        # communicator next; implicit launch ordering permits GPU overlap but
        # preserves this deterministic host order.
        buffer.allow_cross_launch()

        if profile:
            cls._prefill_profile_grouped_chunks += (
                num_node_rows + self.chunk_rows - 1
            ) // self.chunk_rows
            owned = (all_idx >= expert_start) & (
                all_idx < expert_start + num_local
            )
            cls._prefill_profile_routed_assignments += owned.sum()
            # Keep the profile fixed-shape and asynchronous. Boolean indexing
            # plus ``unique`` materializes a data-dependent CUDA shape and can
            # synchronize every layer. Map non-owned assignments to one
            # sentinel and scatter the owned IDs into a 113-entry bitmap.
            local_ids = all_idx - expert_start
            sentinel = torch.full_like(local_ids, num_local)
            safe_ids = torch.where(owned, local_ids, sentinel)
            active = torch.zeros(
                (num_local + 1,), dtype=torch.uint8, device=all_idx.device
            )
            active.scatter_(0, safe_ids.reshape(-1).to(torch.int64), 1)
            cls._prefill_profile_active_experts += active[:num_local].sum()
            cls._prefill_profile_node_routed_assignments += num_rows * top_k
            cls._prefill_profile_expert_shard_size = num_local
            # Start grouped timing after the asynchronous accounting kernels;
            # they are observability overhead, not grouped-MoE execution.
            _profile_mark(marks)

        # --- this rank's 112 experts over ALL node rows (non-owned -> -1) ---
        # Chunked on the independent token dimension only. 2048 rows keeps the
        # per-expert GEMMs wide enough to matter while bounding the padded
        # dispatch/intermediate/expert_out buffers; the 256-row bound in
        # ``compact_prefill_chunk_rows`` answers the resident-EP path's HBM
        # budget, where 896 experts stay materialized on every rank.
        combined = all_latent.new_empty(
            (num_node_rows, latent_size), dtype=torch.float32
        )
        for start in range(0, num_node_rows, self.chunk_rows):
            end = min(start + self.chunk_rows, num_node_rows)
            count = end - start
            expert_out, topk_pos = helper._expert_path(
                all_latent[start:end],
                all_idx[start:end],
                count,
            )
            combined[start:end].copy_(
                helper._combine_fp32(
                    expert_out,
                    topk_pos,
                    all_weight[start:end],
                    count,
                    latent_size,
                    top_k,
                )
            )
        if profile:
            _profile_mark(marks)

        # --- node-local SUM of the eight disjoint FP32 partials, scattered ---
        # back to each rank's own padded row block. Summing in FP32 before the
        # single bf16 downcast matches the resident-EP reduction exactly.
        # Cross-node ingress completes before TP8 is entered again. Host/ring
        # acquisition still started at the top of this layer and overlapped its
        # router, gathers and grouped expert work; only the two orthogonal NCCL
        # communicator families are kept from contending with one another.
        buffer.order_tp_collective_after_cross_launch()
        if profile:
            # Bounds the compute stream's stall on the cross-node ingress
            # fence. host_rdma has no such fence, so this span is ~0 there.
            _profile_mark(marks)
        local_latent = combined.new_empty((ntp, latent_size))
        if profile:
            _profile_mark(marks)
        dist.reduce_scatter_tensor(
            local_latent, combined, op=dist.ReduceOp.SUM, group=tp_group
        )
        if profile:
            _profile_mark(marks)
        # Every kernel and node-local collective that reads ``buffer.compute``
        # has now been enqueued, so the prefetch started above may assemble the
        # next layer into it.  Nothing below touches expert weights.
        buffer.allow_full_overwrite()

        output = x.new_empty((T, H))
        for start in range(0, T, self.post_chunk_rows):
            end = min(start + self.post_chunk_rows, T)
            y = local_latent[start:end].to(torch.bfloat16)
            if self.norm is not None:
                y = self.norm(y)
            output[start:end].copy_(self.up_proj(y))
        if profile:
            _profile_mark(marks)
            cls._prefill_profile_layer_marks.append(tuple(marks))
            cls._prefill_profile_wall_s += (
                time.perf_counter() - profile_start
            )
        return output
