"""Coordinated worker-pool restart across the nodes of one server.

A worker cohort is a single ``torch.distributed`` world spanning every node, so
no node can replace its own ranks alone: the surviving peers stay blocked in
the old world's collectives while the fresh ranks wait for a rendezvous nobody
else joins.  Every node therefore runs the same phase protocol against a
server-level store hosted by node 0, and leaves a phase only once every node
has acknowledged it.

Node 0 is the only entry point.  It owns the epoch counter, publishes the
request, and runs the phases like any other node; the peers learn about the
epoch from a follower thread and run the identical phase function.

A single-node server creates no store at all and every method here returns
immediately, so its restart keeps exactly the phases and timings it had.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

#: The phases of one restart, in the order every node runs them.  A node
#: publishes its ack for a phase and may only proceed once every peer has.
PHASE_ARMED = "P0"
PHASE_TORN_DOWN = "P1"
PHASE_READY = "P2"
RESTART_PHASES = (PHASE_ARMED, PHASE_TORN_DOWN, PHASE_READY)

# A peer's control store exists only once its server process is up, and a
# server spends minutes loading weights before that.  The connect budget is
# therefore the one the daemon readiness wait already uses, not a
# network-level timeout.
_STORE_CONNECT_TIMEOUT_S = 600.0
# Node 0 bumps the epoch and then writes the request describing it, so a
# follower can observe the new epoch a moment before that description exists.
# Only this publication gap is waited out; a request still missing afterwards
# means node 0 died between the two writes, which fails this node's restart.
_REQUEST_PUBLISH_WAIT_S = 30.0
# A store blip must not fail an otherwise healthy restart, so reads and writes
# are retried rather than fatal.  The follower backs off by this much between
# attempts; a restart in progress fails at its shared phase deadline instead.
_CONTROL_ERROR_BACKOFF_S = 2.0
# Reasons are published into the store and read back through
# /v1/workers/status.  A whole traceback helps nobody there and would make the
# value unbounded.
_REASON_MAX_CHARS = 400
# Node 0 publishes the run identity right after it opens the store, so a peer
# that connected has it almost at once; the wait only covers a peer that
# connected before node 0 finished its own start-up up to that point.
_RUN_IDENTITY_WAIT_S = 60.0

_EPOCH_KEY = "epoch"
# One key holding the whole payload, so a reader never sees half of it.
_RUN_IDENTITY_KEY = "run/identity"
_ACK_OK = "ok"
_ACK_FAILED_PREFIX = "failed:"


class PeerRestartError(RuntimeError):
    """A peer node failed, or never reached, the phase this node waited for."""

    def __init__(self, node: int, phase: str, reason: str) -> None:
        super().__init__(
            f"peer node {node} did not complete restart phase {phase}: {reason}"
        )
        self.node = node
        self.phase = phase
        self.reason = reason


@dataclass(frozen=True)
class RestartRequest:
    """One restart, as every node sees it.

    ``deadline`` is wall-clock (``time.time``) rather than monotonic on
    purpose: it is derived from the requester's clock and has to mean the same
    instant on every node, so the phase barriers all give up together instead
    of one node waiting out its own full timeout after the others have gone.
    """

    epoch: int
    timeout_s: float
    requested_at: float

    @property
    def deadline(self) -> float:
        return self.requested_at + self.timeout_s


def _request_key(epoch: int) -> str:
    return f"restart/{epoch}/request"


def _ack_key(epoch: int, phase: str, node: int) -> str:
    return f"restart/{epoch}/{phase}/node{node}"


def _result_key(epoch: int) -> str:
    return f"restart/{epoch}/result"


def _down_key(generation: int, node: int) -> str:
    return f"down/{generation}/node{node}"


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return "" if value is None else str(value)


def _short_reason(reason: str) -> str:
    """One bounded line: the store value is also a status-endpoint field."""
    collapsed = " ".join(str(reason).split())
    if len(collapsed) > _REASON_MAX_CHARS:
        collapsed = collapsed[: _REASON_MAX_CHARS - 3] + "..."
    return collapsed or "unspecified"


def create_tcp_store(
    host: str,
    port: int,
    nnodes: int,
    node_rank: int,
    connect_timeout_s: float,
):
    """The default control store: node 0 hosts it, every peer connects.

    ``wait_for_workers`` is off because node 0 must be able to serve while a
    peer is still loading weights; the peers' own constructors carry the
    connect budget.
    """
    import torch.distributed as dist

    return dist.TCPStore(
        host,
        port,
        world_size=nnodes,
        is_master=node_rank == 0,
        timeout=timedelta(seconds=connect_timeout_s),
        wait_for_workers=False,
    )


class RestartCoordinator:
    """The node-local half of the coordinated restart protocol."""

    def __init__(
        self,
        nnodes: int,
        node_rank: int,
        host: str,
        port: int,
        *,
        store_factory: Optional[Callable[..., Any]] = None,
        poll_interval_s: float = 0.5,
        connect_timeout_s: float = _STORE_CONNECT_TIMEOUT_S,
        request_wait_s: float = _REQUEST_PUBLISH_WAIT_S,
        error_backoff_s: float = _CONTROL_ERROR_BACKOFF_S,
        run_identity_wait_s: float = _RUN_IDENTITY_WAIT_S,
    ) -> None:
        self._nnodes = int(nnodes)
        self._node_rank = int(node_rank)
        self._host = host
        self._port = int(port)
        self._store_factory = store_factory or create_tcp_store
        self._poll_interval_s = float(poll_interval_s)
        self._connect_timeout_s = float(connect_timeout_s)
        self._request_wait_s = float(request_wait_s)
        self._error_backoff_s = float(error_backoff_s)
        self._run_identity_wait_s = float(run_identity_wait_s)
        self._store: Optional[Any] = None
        self._epoch = 0
        # Phases this node published an ``ok`` ack for.  An ack is a promise
        # the peers act on, so it is never overwritten with a failure.
        self._acked: set = set()
        self._control_error: Optional[str] = None
        self._reported_error: Optional[str] = None
        self._reported_down: set = set()
        self._follower: Optional[threading.Thread] = None
        self._follower_stop = threading.Event()

    # ---------------------- Lifecycle ----------------------
    @property
    def enabled(self) -> bool:
        return self._nnodes > 1

    @property
    def node_rank(self) -> int:
        return self._node_rank

    @property
    def nnodes(self) -> int:
        return self._nnodes

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def control_error(self) -> Optional[str]:
        return self._control_error

    def start(self) -> None:
        """Open the control store; node 0 hosts it for the server's whole life."""
        if not self.enabled or self._store is not None:
            return
        self._store = self._store_factory(
            self._host,
            self._port,
            self._nnodes,
            self._node_rank,
            self._connect_timeout_s,
        )
        logger.info(
            "Restart control store %s at %s:%d (node %d of %d)",
            "hosted" if self._node_rank == 0 else "connected",
            self._host,
            self._port,
            self._node_rank,
            self._nnodes,
        )

    def close(self) -> None:
        self._follower_stop.set()
        follower = self._follower
        self._follower = None
        if follower is not None and follower is not threading.current_thread():
            follower.join(timeout=5)
        store = self._store
        self._store = None
        if store is None:
            return
        # Dropping the last reference tears the listener down; a store that
        # grows an explicit close is honoured too.
        closer = getattr(store, "close", None)
        if closer is not None:
            try:
                closer()
            except Exception:
                logger.warning("Closing the restart control store failed",
                               exc_info=True)

    def _require_store(self) -> Any:
        if self._store is None:
            raise RuntimeError("the restart control store is not open")
        return self._store

    # ---------------------- Run identity ----------------------
    def publish_run_identity(self, payload: str) -> None:
        """Node 0: publish the run identity every node names its logs by.

        The payload is opaque here; a single-node server has no peer to tell.
        """
        if not self.enabled:
            return
        self._require_store().set(_RUN_IDENTITY_KEY, payload)

    def wait_run_identity(self) -> str:
        """Node k > 0: the payload node 0 published, waiting for it briefly.

        Raises ``RuntimeError`` when node 0 has not published it in time.
        """
        store = self._require_store()
        if not store.check([_RUN_IDENTITY_KEY]):
            try:
                store.wait(
                    [_RUN_IDENTITY_KEY],
                    timedelta(seconds=self._run_identity_wait_s),
                )
            except Exception as exc:
                raise RuntimeError(
                    "node 0 did not publish the run identity within "
                    f"{self._run_identity_wait_s:.0f}s: {exc}"
                ) from exc
        return _text(store.get(_RUN_IDENTITY_KEY))

    # ---------------------- Node 0: requesting ----------------------
    def begin_restart(self, timeout_s: float) -> RestartRequest:
        """Claim the next epoch and publish what every node must run."""
        requested_at = time.time()
        request = RestartRequest(
            epoch=0, timeout_s=float(timeout_s), requested_at=requested_at
        )
        if not self.enabled:
            return request
        store = self._require_store()
        epoch = int(store.add(_EPOCH_KEY, 1))
        request = RestartRequest(
            epoch=epoch, timeout_s=float(timeout_s), requested_at=requested_at
        )
        store.set(
            _request_key(epoch),
            json.dumps(
                {"timeout_s": request.timeout_s, "requested_at": requested_at}
            ),
        )
        self._epoch = epoch
        logger.info("Restart epoch %d requested (timeout %.1fs)",
                    epoch, request.timeout_s)
        return request

    def publish_result(self, epoch: int, result: Dict[str, Any]) -> None:
        if not self.enabled or epoch is None:
            return
        try:
            self._require_store().set(
                _result_key(epoch), json.dumps(result, sort_keys=True, default=str)
            )
        except Exception as exc:
            self._record_control_error(exc)

    # ---------------------- Phase protocol ----------------------
    def barrier(self, epoch: int, phase: str, deadline: float) -> None:
        """Ack this node's phase, then wait for every peer's ack.

        Raises :class:`PeerRestartError` when a peer published a failure or
        never acked before the shared deadline, so the caller's own failure
        path runs and the pool ends up retryable rather than half-restarted.
        """
        if not self.enabled:
            return
        pending = {node for node in range(self._nnodes) if node != self._node_rank}
        acked = False
        while True:
            try:
                if not acked:
                    self._write_ack(epoch, phase, _ACK_OK)
                    acked = True
                for node in sorted(pending):
                    value = self._read_ack(epoch, phase, node)
                    if value is None:
                        continue
                    if value.startswith(_ACK_FAILED_PREFIX):
                        raise PeerRestartError(
                            node, phase, value[len(_ACK_FAILED_PREFIX):]
                        )
                    pending.discard(node)
            except PeerRestartError:
                raise
            except Exception as exc:
                raise self._store_failure(phase, exc) from exc
            if not pending and acked:
                return
            remaining = deadline - time.time()
            if remaining <= 0:
                raise PeerRestartError(
                    min(pending), phase, "no phase ack before the restart deadline"
                )
            time.sleep(min(self._poll_interval_s, remaining))

    def peer_failed(self, epoch: int, phase: str) -> Optional[PeerRestartError]:
        """Poll for a peer failure without waiting for the missing acks."""
        if not self.enabled or epoch is None:
            return None
        try:
            for node in range(self._nnodes):
                if node == self._node_rank:
                    continue
                value = self._read_ack(epoch, phase, node)
                if value is not None and value.startswith(_ACK_FAILED_PREFIX):
                    return PeerRestartError(
                        node, phase, value[len(_ACK_FAILED_PREFIX):]
                    )
        except Exception as exc:
            return self._store_failure(phase, exc)
        return None

    def _store_failure(self, phase: str, exc: BaseException) -> RuntimeError:
        """The error a restart phase raises when the control store fails.

        Never retried: the store lives in node 0's server and a TCPStore
        client does not reconnect, so a failed call means node 0 stopped or
        died (or, on node 0, that this server is stopping). Waiting out the
        deadline would only hold every peer in ``restarting`` for the whole
        restart timeout.
        """
        self._record_control_error(exc)
        if self._store is None:
            return RuntimeError(
                f"restart phase {phase} aborted: the restart control store is "
                "closed"
            )
        if self._node_rank != 0:
            return PeerRestartError(
                0, phase, f"restart coordinator unreachable: {exc}"
            )
        return RuntimeError(
            f"restart phase {phase} could not be coordinated: {exc}"
        )

    def raise_if_peer_failed(self, epoch: int, phase: str) -> None:
        failure = self.peer_failed(epoch, phase)
        if failure is not None:
            raise failure

    def fail_phase(self, epoch: int, phase: str, reason: str) -> None:
        """Publish this node's failure so the peers abort at their barrier."""
        if not self.enabled or epoch is None:
            return
        if (epoch, phase) in self._acked:
            # The peers already acted on this node's ``ok``; the failure
            # belongs to whatever phase it has not acked yet, and the caller
            # names that phase.  Overwriting the promise would only hide it.
            logger.error(
                "Restart epoch %d phase %s already acked by this node; "
                "not publishing the failure: %s",
                epoch, phase, reason,
            )
            return
        try:
            self._write_ack(epoch, phase, _ACK_FAILED_PREFIX + _short_reason(reason))
        except Exception as exc:
            self._record_control_error(exc)

    def peer_acks(self) -> Dict[str, Any]:
        """The newest phase each node acked in the current epoch, for status."""
        if not self.enabled:
            return {}
        acks: Dict[str, Any] = {}
        try:
            for node in range(self._nnodes):
                for phase in reversed(RESTART_PHASES):
                    value = self._read_ack(self._epoch, phase, node)
                    if value is None:
                        continue
                    acks[f"node{node}"] = {"phase": phase, "ack": value}
                    break
        except Exception as exc:
            self._record_control_error(exc)
        return acks

    def _write_ack(self, epoch: int, phase: str, value: str) -> None:
        self._require_store().set(_ack_key(epoch, phase, self._node_rank), value)
        if value == _ACK_OK:
            self._acked.add((epoch, phase))

    def _read_ack(self, epoch: int, phase: str, node: int) -> Optional[str]:
        store = self._require_store()
        key = _ack_key(epoch, phase, node)
        # check() answers from what the store already has; get() would block
        # on a missing key until the store's own timeout.
        if not store.check([key]):
            return None
        return _text(store.get(key))

    # ---------------------- Cohort failure publication ----------------------
    def publish_down(self, generation: int, reason: str) -> None:
        """Announce that this node's cohort of ``generation`` is down.

        First writer wins: a cohort dies on every node at once, so the node
        that noticed first names the reason and the others' echoes do not
        overwrite it.
        """
        if not self.enabled:
            return
        try:
            store = self._require_store()
            winner = _text(
                store.compare_set(
                    _down_key(generation, self._node_rank),
                    "",
                    _short_reason(reason),
                )
            )
            logger.info(
                "Published cohort generation %d as down on node %d: %s",
                generation, self._node_rank, winner,
            )
        except Exception as exc:
            self._record_control_error(exc)

    def _poll_peer_down(self, generation: int) -> Optional[Tuple[int, str]]:
        store = self._require_store()
        for node in range(self._nnodes):
            if node == self._node_rank:
                continue
            key = _down_key(generation, node)
            if key in self._reported_down or not store.check([key]):
                continue
            reason = _text(store.get(key))
            # Only once the read succeeded: a key marked reported before a
            # failed get would never be reported at all.
            self._reported_down.add(key)
            return node, reason
        return None

    # ---------------------- Follower ----------------------
    def start_follower(
        self,
        *,
        state: Callable[[], str],
        generation: Callable[[], int],
        on_peer_restart: Callable[[RestartRequest], Any],
        on_peer_down: Callable[[int, str], Any],
    ) -> None:
        """Watch the store on EVERY node, on a thread of its own.

        A coordinated restart runs to completion inline on this thread and can
        take minutes, so it must never be the asyncio loop: node 0 reaches the
        same code from an ``asyncio.to_thread`` worker, and the peers reach it
        from here.
        """
        if not self.enabled or self._follower is not None:
            return
        self._follower_stop.clear()
        try:
            self._epoch = self._read_epoch()
        except Exception as exc:
            # A baseline that cannot be read now is re-read by the loop; the
            # only cost is that the first poll may see an epoch as new.
            self._record_control_error(exc)
        self._follower = threading.Thread(
            target=self._follow,
            args=(state, generation, on_peer_restart, on_peer_down),
            name="restart-coordinator",
            daemon=True,
        )
        self._follower.start()

    def _follow(
        self,
        state: Callable[[], str],
        generation: Callable[[], int],
        on_peer_restart: Callable[[RestartRequest], Any],
        on_peer_down: Callable[[int, str], Any],
    ) -> None:
        while not self._follower_stop.is_set():
            try:
                epoch = self._read_epoch()
                if self._node_rank != 0 and epoch > self._epoch:
                    self._epoch = epoch
                    try:
                        request = self._read_request(epoch)
                    except Exception as exc:
                        # The epoch exists but what to run with it does not, so
                        # this node cannot join it. Node 0 is told, rather than
                        # left waiting out the deadline it never published.
                        self._record_control_error(exc)
                        self.fail_phase(
                            epoch, PHASE_ARMED, f"restart request unreadable: {exc}"
                        )
                        continue
                    self._run_callback(on_peer_restart, request)
                    continue
                self._epoch = max(self._epoch, epoch)
                if state() == "serving":
                    down = self._poll_peer_down(generation())
                    if down is not None:
                        self._run_callback(on_peer_down, *down)
            except Exception as exc:
                self._record_control_error(exc)
                self._follower_stop.wait(self._error_backoff_s)
                continue
            self._follower_stop.wait(self._poll_interval_s)

    @staticmethod
    def _run_callback(callback: Callable[..., Any], *args: Any) -> None:
        """A failed restart or failure handover must not end the follower.

        Both callbacks already leave the pool in the retryable
        ``workers_down`` state; this thread has to survive to see the next
        epoch, which is how that state is recovered from.
        """
        try:
            callback(*args)
        except BaseException:
            logger.exception("Coordinated restart handler failed")

    def _read_epoch(self) -> int:
        # add(key, 0) reads the counter and creates it as 0 when absent, so no
        # node has to initialize it and no read blocks on a missing key.
        return int(self._require_store().add(_EPOCH_KEY, 0))

    def _read_request(self, epoch: int) -> RestartRequest:
        store = self._require_store()
        key = _request_key(epoch)
        if not store.check([key]):
            store.wait([key], timedelta(seconds=self._request_wait_s))
        payload = json.loads(_text(store.get(key)))
        return RestartRequest(
            epoch=epoch,
            timeout_s=float(payload["timeout_s"]),
            requested_at=float(payload["requested_at"]),
        )

    # ---------------------- Error reporting ----------------------
    def _record_control_error(self, exc: BaseException) -> None:
        message = f"coordinator unreachable: {exc}"
        self._control_error = message
        if message != self._reported_error:
            self._reported_error = message
            logger.error("Restart control store error: %s", exc, exc_info=True)
