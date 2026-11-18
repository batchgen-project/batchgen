"""Node replica weight transport install behind a worker seam (repatriation R3).

``ReplicaTransportEngine`` owns the install of the node replica weight
transport for a prefill phase: forming the replica group from per-rank
topology records, deriving the frame layout from the installed task's module
names, and handing the core engine the plan, the cohort control block and
every member's ring segments.

The engine is intentionally stateless, the S5/S8/S7/S9 shape: each operation
receives the live worker as its collaborator and rebinds ``self`` to it, so
the moved statements keep their original state access and call spellings. The
worker keeps a same-name delegator and remains the sole state owner
(``_replica_plan_installed``, ``_replica_block_cache``,
``weight_copy_task`` and the cross-node broadcast plan all live on the
worker). The uninstall path, the cross-node broadcast plan and the comm
thread stay on the worker: they are shared with the prefill weight-copy
pipeline's lifecycle.
"""

from __future__ import annotations

import logging
import os
import socket
import time

import torch
import torch.distributed as dist

from batchgen.native_loader import core_engine


class ReplicaTransportEngine:
    """Stateless install seam for the node replica weight transport."""

    def _install_replica_transport(self, worker) -> None:
        """Install the node replica weight transport for this prefill phase.

        Forms the replica group from per-rank topology records, derives the
        frame layout from the installed task's module names, and hands the
        engine the plan, the cohort control block and every member's ring
        segments. A rank that cannot participate (no frame grammar, daemon
        or hierarchical source, no peer access, group of one) runs the
        degenerate single-member group: the uninstalled engine, which is the
        same record state machine with every record HostPrivate.
        """
        # Moved VERBATIM from batchgen_worker.py (repatriation R3); rebinding self
        # to the worker keeps every original self. spelling. The engine is stateless.
        self = worker
        from batchgen.batchgen_worker import allocate_node_shared_tensor
        from batchgen.weight_replica_group import (
            RankRecord,
            compute_replica_groups,
            derive_frame_plan,
            derive_owner_map,
            install_record_kinds,
            stagger_task_for_member,
        )

        transport = getattr(
            self.parallel_manager, "_distributed_weight_transport", None
        )
        seam_requested = (
            getattr(self.parallel_manager, "cross_seam_enabled", None)
            is not None
            and self.parallel_manager.cross_seam_enabled()
        )
        cross_seam = (
            seam_requested
            and transport == "hierarchical_gdr"
            and getattr(self, "_cross_node_broadcast_plan", None) is not None
        )

        def _decline(reason: str) -> None:
            if seam_requested:
                # Under the seam every member already holds the full
                # schedule, so a member without the engine binding would
                # fetch records it does not own: never run that silently.
                raise RuntimeError(
                    f"Rank {self.rank}: cross-node replica seam requested "
                    f"but the transport cannot install: {reason}"
                )
            if self.rank == 0:
                logging.info(
                    "[REPLICA] transport not installed: %s", reason
                )

        if seam_requested and not cross_seam:
            _decline("no cross-node plan was declared for this member")
        if not dist.is_initialized():
            _decline("torch.distributed is not initialized")
            return
        if (
            getattr(self.args, "distributed_weight_config", None)
            and not cross_seam
        ):
            _decline("daemon-backed weight source (spec v4, 4.1)")
            return
        if transport == "hierarchical_gdr" and not cross_seam:
            _decline("hierarchical_gdr weight source (spec v4, 4.1)")
            return
        plan = derive_frame_plan(self.weight_copy_task)
        if plan is None:
            _decline("installed task does not fit the layer grammar")
            return
        # Install-step timing: the install runs once per prefill phase and
        # spans collectives, shared mappings and peer imports, so each rank
        # logs where its install time and device memory went.
        _steps: list = []
        _last = time.perf_counter()

        def _mark(step: str) -> None:
            nonlocal _last
            now = time.perf_counter()
            free_gib = torch.cuda.mem_get_info(self.local_rank)[0] / (1 << 30)
            _steps.append((step, now - _last, free_gib))
            _last = now

        with open("/proc/sys/kernel/random/boot_id") as handle:
            boot_id = handle.read().strip()
        record = RankRecord(
            node_identity=f"{boot_id}:{socket.gethostname()}",
            device_index=self.local_rank,
            peer_access_row=[
                bool(torch.cuda.can_device_access_peer(self.local_rank, j))
                if j != self.local_rank else True
                for j in range(self.local_world_size)
            ],
            # Canonical (order-insensitive) identity: members compare the
            # SET of modules, because a prior install staggered this
            # member's walk order and the stagger differs per member.
            list_fingerprint=str(
                self._weight_copy_task_fingerprint(
                    {
                        module_type: sorted(names)
                        for module_type, names in (
                            self.weight_copy_task or {}
                        ).items()
                    }
                )
            ),
            weight_source_kind="local",
            rank=self.rank,
        )
        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, record)
        _mark("gather")
        if cross_seam:
            # Under streamed SP8 every rank of a node streams a different
            # expert shard, so no intra-node peer shares this member's
            # records; the members that do share them live on other nodes
            # and cannot share memory. The engine therefore runs a
            # single-member group: routed experts take their cross-node
            # kind (source root: owned; others: CrossReplica, landed by the
            # comm thread), every other record is this rank's own host copy.
            group = [self.rank]
            owners = {
                name: self.rank
                for names in self.weight_copy_task.values()
                for name in names
            }
            kinds = {name: "HostPrivate" for name in owners}
            for name, kind in self._cross_node_broadcast_plan.kinds.items():
                if name not in kinds:
                    raise RuntimeError(
                        f"Rank {self.rank}: cross-node plan names {name}, "
                        "which is not in the installed task"
                    )
                # The source root owns every record of its slot; with no
                # intra peers a shared record has no reader but itself.
                kinds[name] = (
                    "CrossReplica" if kind == "CrossReplica" else "HostShared"
                )
        else:
            groups = compute_replica_groups({r.rank: r for r in gathered})
            group = sorted(groups[self.rank])
            if len(group) < 2:
                _decline("replica group has one member")
                return
            owners = derive_owner_map(group, self.weight_copy_task, plan)
            kinds = install_record_kinds(self.rank, group, owners)
        member_index = group.index(self.rank)
        # Stagger this member's within-layer walk (own block first, then
        # the other owners round-robin) and rebuild the engine queue to
        # match; the producer has not started. Record identity stays
        # canonical by name, so barriers are unaffected. A single-member
        # group has no other owner to stagger against.
        if len(group) > 1:
            staggered = stagger_task_for_member(
                self.weight_copy_task, plan, owners, group, self.rank
            )
            if staggered != self.weight_copy_task:
                self.weight_copy_task = staggered
                self.core_engine.set_weight_copy_queue(staggered)
        layers_per_pass = len(plan.layer_sequence)
        # Record indices: one frame's records in (sorted type, in-frame
        # position) order, identical on every member.
        type_offsets = {}
        offset = 0
        for module_type in sorted(plan.frame_size):
            type_offsets[module_type] = offset
            offset += plan.frame_size[module_type]
        records_per_frame = offset
        num_types = len(type_offsets)
        from batchgen.weight_replica_transport import in_frame_positions

        # Record index = the record's position among its type's records in
        # its frame (canonical order), never the raw expert id: a streamed
        # SP8 slot holds a block of global expert ids, while the per-frame
        # record table is sized by the frame's record count.
        positions = in_frame_positions(self.weight_copy_task, plan)
        modules = []
        for module_type in sorted(self.weight_copy_task):
            names = self.weight_copy_task[module_type]
            if not names:
                continue
            ring_slots = int(
                self.engine_config.GPU_Buffer_Config
                .num_prefill_module_buffer[module_type]
            )
            # Only records that peers read need the two-frame rendezvous;
            # under the seam that is the routed experts alone (every other
            # type is each rank's own host copy).
            if (
                not cross_seam or module_type == "routed_expert"
            ) and ring_slots < 2 * plan.frame_size[module_type]:
                # A ring that cannot hold the two-frame rendezvous cannot
                # replicate; the model keeps its single-member behaviour.
                _decline(
                    f"ring capacity for {module_type} is {ring_slots} "
                    f"slots, the two-frame rendezvous needs "
                    f"{2 * plan.frame_size[module_type]}"
                )
                return
            for name in names:
                layer_position, _expert = plan.position[name]
                record_index = type_offsets[module_type] + positions[name]
                owner_member = group.index(owners[name])
                kind = {
                    "HostShared": 0,
                    "HostPrivate": 1,
                    "Replica": 2,
                    "CrossReplica": 3,
                }[kinds[name]]
                modules.append(
                    (
                        module_type,
                        name,
                        kind,
                        owner_member,
                        layer_position,
                        record_index,
                    )
                )
        _mark("plan")
        # The control block: one shared node mapping per geometry, reused
        # across installs of the same shape. Barrier words are per
        # (member, type): frame completion is monotone only within a type.
        key = (len(group), records_per_frame, num_types)
        cached = self._replica_block_cache.get(key)
        if cached is None and cross_seam:
            # A single-member group shares the block with nobody: a
            # process-private mapping replaces the node-shared one, whose
            # creator handshake assumes every node rank joins one group.
            nbytes = int(
                core_engine.replica_control_block_bytes(
                    len(group), records_per_frame, num_types
                )
            )
            # The block's header is cache-line aligned (alignas(64)) and the
            # CPU allocator does not promise that alignment, so carve an
            # aligned window out of a slightly larger buffer; the slice keeps
            # the backing storage alive.
            backing = torch.zeros(nbytes + 64, dtype=torch.uint8)
            start = (-backing.data_ptr()) % 64
            cached = (backing[start:start + nbytes], None)
            self._replica_block_cache[key] = cached
        if cached is None:
            nbytes = int(
                core_engine.replica_control_block_bytes(
                    len(group), records_per_frame, num_types
                )
            )
            storage, memfd = allocate_node_shared_tensor(
                label=(
                    f"{self._query_book_shm_prefix}replica_block_"
                    f"{len(group)}x{records_per_frame}x{num_types}"
                ),
                rows=nbytes,
                width=1,
                dtype=torch.uint8,
                is_creator=self.local_rank == 0,
                exchange=self._exchange_node_memfd_identity,
                barrier=dist.barrier,
            )
            cached = (storage.view(-1), memfd)
            self._replica_block_cache[key] = cached
        block_tensor = cached[0]
        if member_index == 0:
            core_engine.replica_control_block_initialize(
                block_tensor, len(group), records_per_frame, num_types
            )
        dist.barrier()
        _mark("block")
        # Ring segments travel as exported VMM fds over a node-local unix
        # socket (SCM_RIGHTS): the peer maps only the segment, so the import
        # is milliseconds and leaves no foreign context or whole-device peer
        # mapping behind (the torch CUDA-IPC path cost minutes and GiBs on a
        # serving worker; slice 2c). The /proc/<pid>/fd reopen seam does not
        # work for driver fds (VMM P1, finding C1), so the fds are sent.
        import socket as _socket

        types = sorted(plan.frame_size)
        own_fds = {}
        own_bytes = {}
        field_layouts = {}
        for module_type in types:
            fd, mapped = self.core_engine.ring_segment_export_fd(module_type)
            own_fds[module_type] = int(fd)
            own_bytes[module_type] = int(mapped)
            field_layouts[module_type] = [
                (int(offset), int(stride), int(nbytes))
                for offset, stride, nbytes in
                self.core_engine.ring_field_layout(module_type)
            ]
        _mark("export")
        peer_fds = {t: [-1] * len(group) for t in types}
        peer_bytes = {t: [0] * len(group) for t in types}
        for module_type in types:
            peer_bytes[module_type][member_index] = own_bytes[module_type]
        listener = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        sock_name = (
            f"\0{self._query_book_shm_prefix}replica-fd-{self.rank}"
        )
        try:
            listener.bind(sock_name)
            listener.listen(len(group))
            dist.barrier()
            payload = repr(
                (self.rank, types, [own_bytes[t] for t in types])
            ).encode()
            senders = []
            for member_rank in group:
                if member_rank == self.rank:
                    continue
                peer = _socket.socket(
                    _socket.AF_UNIX, _socket.SOCK_STREAM
                )
                peer.connect(
                    f"\0{self._query_book_shm_prefix}replica-fd-"
                    f"{member_rank}"
                )
                _socket.send_fds(
                    peer, [payload], [own_fds[t] for t in types]
                )
                senders.append(peer)
            for _ in range(len(group) - 1):
                conn, _addr = listener.accept()
                with conn:
                    data, fds, _flags, _addr2 = _socket.recv_fds(
                        conn, 4096, len(types)
                    )
                    import ast as _ast

                    sender_rank, sender_types, sender_bytes = (
                        _ast.literal_eval(data.decode())
                    )
                    if sender_types != types or len(fds) != len(types):
                        raise RuntimeError(
                            f"Rank {self.rank}: replica fd exchange from "
                            f"rank {sender_rank} carried types "
                            f"{sender_types} ({len(fds)} fds), expected "
                            f"{types}"
                        )
                    sender_member = group.index(sender_rank)
                    for module_type, fd, nbytes in zip(
                        types, fds, sender_bytes
                    ):
                        peer_fds[module_type][sender_member] = int(fd)
                        peer_bytes[module_type][sender_member] = int(nbytes)
            # The senders may close only after every receiver imported.
            dist.barrier()
            for peer in senders:
                peer.close()
        finally:
            listener.close()
        _mark("handles")
        try:
            self.core_engine.install_replica_plan(
                member_index,
                len(group),
                layers_per_pass,
                records_per_frame,
                modules,
                block_tensor,
                peer_fds,
                peer_bytes,
                field_layouts,
            )
        finally:
            for module_type in types:
                os.close(own_fds[module_type])
                for fd in peer_fds[module_type]:
                    if fd >= 0:
                        os.close(fd)
        self._replica_plan_installed = True
        _mark("engine")
        logging.info(
            "[REPLICA] install steps rank=%d: %s",
            self.rank,
            " ".join(
                f"{n}={d:.2f}s/free={f:.1f}G" for n, d, f in _steps
            ),
        )
        if self.rank == group[0]:
            logging.info(
                "[REPLICA] installed: group=%s member=%d records/frame=%d "
                "layers/pass=%d fingerprint=%s",
                group,
                member_index,
                records_per_frame,
                layers_per_pass,
                record.list_fingerprint[:16],
            )
