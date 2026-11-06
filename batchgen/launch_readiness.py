"""Launch readiness probe: the node-level gates a cold start depends on.

A multi-node cold start commits to its environment long before it uses it.
Mount flags, kernel watermarks, port ownership and the reachability of every
path the launch names are all decided at launch time, but each one is first
*exercised* minutes in, in the middle of a staging copy or a rendezvous --
so a wrong flag or an unreadable path is reported late, from deep inside a
phase, after the whole group has already committed its GPUs to the start.

This module moves those gates to the front. Every check completes in seconds,
every filesystem touch is bounded by a hard deadline, and the result is a
named list of ok/warn/fail verdicts with the evidence behind each one. It is
run two ways, from one implementation:

* ``python -m batchgen.launch_readiness --model <id> ...`` before launching,
  to judge a node (and with ``--json``, to judge a fleet from a script);
* inside the server's own startup, before any worker, port, SHM segment or
  weight byte exists, where a failing check refuses the launch.

It is deliberately dependency-light: nothing but the standard library is
imported at module load, and every ``batchgen`` import is local to the check
that needs it, so the probe runs in an environment where the heavy runtime
cannot even be imported yet -- which is exactly the environment it judges.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import socket
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

STATUS_OK = "ok"
STATUS_WARN = "warn"
STATUS_FAIL = "fail"

#: Bytes one probe read asks for. The store's first MiB is enough to prove
#: the mapping is readable; a larger read would measure bandwidth instead.
PROBE_READ_BYTES = 1 << 20
#: How long a probe read may take before it is declared stalled. Generous
#: against a cold cache on a network filesystem, far under the time a stalled
#: read would otherwise hold a launch.
DEFAULT_READ_DEADLINE_S = 15.0
#: Free bytes the staging tmpfs must have beyond the store itself, for the
#: metadata and manifest copied beside it and for the tmpfs's own slack.
STAGING_HEADROOM_BYTES = 1 << 30
#: Upper bound on ``vm.watermark_scale_factor`` (units of 1/10000 of memory
#: per zone). The kernel keeps that fraction free above the low watermark, and
#: it reclaims to reach it. A large value therefore holds a large slice of RAM
#: permanently unavailable and makes the kernel reclaim rather than hand out
#: the huge and pinned pages a cold start allocates in bulk -- which shows up
#: as a staging copy or a host-KV registration that crawls, not as an error.
#: 500 leaves 5% of memory held free, which is already high for a serving node.
MAX_WATERMARK_SCALE_FACTOR = 500
_WATERMARK_SCALE_FACTOR_PATH = "/proc/sys/vm/watermark_scale_factor"
_THP_SHMEM_ENABLED_PATH = "/sys/kernel/mm/transparent_hugepage/shmem_enabled"
_IP_LOCAL_PORT_RANGE_PATH = "/proc/sys/net/ipv4/ip_local_port_range"
_IP_LOCAL_RESERVED_PORTS_PATH = "/proc/sys/net/ipv4/ip_local_reserved_ports"
#: THP shmem settings under which shared memory is backed by huge pages.
THP_SHMEM_VALUES = ("always", "within_size")
#: Bound on the ``nvidia-smi`` probe, so a wedged driver cannot hold the gate.
NVIDIA_SMI_TIMEOUT_S = 10.0


class LaunchReadinessError(RuntimeError):
    """The node is not ready to launch; the message names every failure."""


class StalledReadError(OSError):
    """A bounded probe read did not return before its deadline."""


# --------------------------------------------------------------------------- #
#  Results                                                                    #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CheckResult:
    """One named check's verdict and the evidence behind it."""

    name: str
    status: str
    evidence: str

    @property
    def failed(self) -> bool:
        return self.status == STATUS_FAIL

    def to_json(self) -> Dict[str, str]:
        return {"name": self.name, "status": self.status, "evidence": self.evidence}

    def __str__(self) -> str:
        return f"[{self.status}] {self.name}: {self.evidence}"


@dataclass(frozen=True)
class LaunchReadinessReport:
    checks: Tuple[CheckResult, ...]

    @property
    def failures(self) -> Tuple[CheckResult, ...]:
        return tuple(check for check in self.checks if check.failed)

    @property
    def warnings(self) -> Tuple[CheckResult, ...]:
        return tuple(check for check in self.checks if check.status == STATUS_WARN)

    @property
    def ok(self) -> bool:
        return not self.failures

    def to_json(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "checks": [check.to_json() for check in self.checks],
        }

    def message(self) -> str:
        """One line naming every failure, for the exception and the log."""
        return "launch readiness failed: " + "; ".join(
            f"{check.name}: {check.evidence}" for check in self.failures
        )


# --------------------------------------------------------------------------- #
#  Bounded reads                                                              #
# --------------------------------------------------------------------------- #


def read_with_deadline(
    path: Path | str,
    *,
    size: int = PROBE_READ_BYTES,
    deadline_s: float = DEFAULT_READ_DEADLINE_S,
    opener: Callable[..., Any] = open,
) -> int:
    """Read up to ``size`` bytes of ``path``; raise if it does not return.

    The failure this exists for is a read that hangs inside the filesystem:
    the calling thread is in an uninterruptible syscall, so it cannot be
    cancelled, signalled or timed out from inside. The deadline is therefore
    enforced by NOT waiting -- the read runs on a daemon thread that this
    function abandons once the deadline passes, and the probe reports the
    stall instead of joining the stuck thread forever. A daemon thread does
    not hold process exit, so the abandoned read cannot keep the probe alive.

    Returns the number of bytes read. Raises :class:`StalledReadError` on the
    deadline, or whatever the read itself raised.
    """
    outcome: Dict[str, Any] = {}

    def _read() -> None:
        try:
            with opener(path, "rb") as handle:
                outcome["read"] = len(handle.read(size))
        except BaseException as error:  # noqa: BLE001 - reported verbatim below
            outcome["error"] = error

    thread = threading.Thread(
        target=_read, name="launch-readiness-read", daemon=True
    )
    thread.start()
    thread.join(deadline_s)
    if thread.is_alive():
        raise StalledReadError(
            f"read stalled: reading {size} B of {path} did not return within "
            f"{deadline_s:.0f}s. The path is on a filesystem that is not "
            "answering; a launch would hang here with nothing in its log"
        )
    error = outcome.get("error")
    if error is not None:
        raise error
    return int(outcome.get("read", 0))


# --------------------------------------------------------------------------- #
#  What a launch names                                                        #
# --------------------------------------------------------------------------- #


@dataclass
class LaunchTarget:
    """The subset of a launch's arguments the node-level gates depend on."""

    model: str
    distributed_weight_config: Optional[Path] = None
    listen_port: Optional[int] = None
    cache_dir: Optional[Path] = None
    converted_ckpt_dir: Optional[Path] = None
    fast_init: bool = False
    dist_init_addr: Optional[str] = None
    restart_control_addr: Optional[str] = None
    pynccl_port_base: Optional[int] = None
    nnodes: int = 1
    node_rank: int = 0
    world_size: int = 1

    @classmethod
    def from_server_args(cls, server_args: Any) -> "LaunchTarget":
        """The same target the server itself is about to start with."""

        def _path(value: Any) -> Optional[Path]:
            return None if value is None else Path(str(value))

        return cls(
            model=str(getattr(server_args, "model", "")),
            distributed_weight_config=_path(
                getattr(server_args, "distributed_weight_config", None)
            ),
            listen_port=getattr(server_args, "listen_port", None),
            cache_dir=_path(getattr(server_args, "cache_dir", None)),
            converted_ckpt_dir=_path(
                getattr(server_args, "converted_ckpt_dir", None)
            ),
            fast_init=bool(getattr(server_args, "fast_init", False)),
            dist_init_addr=getattr(server_args, "dist_init_addr", None),
            restart_control_addr=getattr(server_args, "restart_control_addr", None),
            pynccl_port_base=getattr(server_args, "pynccl_port_base", None),
            nnodes=int(getattr(server_args, "nnodes", 1) or 1),
            node_rank=int(getattr(server_args, "node_rank", 0) or 0),
            world_size=int(getattr(server_args, "world_size", 1) or 1),
        )

    def effective_converted_ckpt_dir(self) -> Optional[Path]:
        """Where the launch resolves its converted checkpoint, as the worker
        does: the explicit flag, else ``<cache-dir>/converted_ckpt``."""
        if self.converted_ckpt_dir is not None:
            return self.converted_ckpt_dir
        if self.cache_dir is not None:
            return self.cache_dir / "converted_ckpt"
        return None


# --------------------------------------------------------------------------- #
#  Probe                                                                      #
# --------------------------------------------------------------------------- #


def _read_sysfs(path: str) -> str:
    return Path(path).read_text()


def is_port_free(port: int) -> bool:
    """Can this node bind ``port``?

    Byte for byte what ``server_args.is_port_available`` does, and it must
    stay that way (``tests/test_launch_readiness.py`` pins the two against
    each other). It is restated here rather than imported because importing
    ``batchgen.server.server_args`` executes ``batchgen/server/__init__.py``,
    which pulls in the HTTP server and its ASGI stack -- the opposite of what
    a probe that must run before the heavy runtime can be imported needs.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("", port))
            sock.listen(1)
            return True
        except (socket.error, OverflowError):
            return False


def _parse_host_port(addr: str) -> Tuple[str, int]:
    """``host:port`` as ``server_args.parse_host_port`` parses it."""
    if ":" not in addr:
        raise ValueError(f"Address must be in host:port format, got '{addr}'")
    host, _, port_text = addr.rpartition(":")
    return host, int(port_text)


def _run_nvidia_smi(argv: Sequence[str], timeout_s: float) -> str:
    return subprocess.run(
        list(argv),
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=True,
    ).stdout


def _nearest_existing(path: Path) -> Optional[Path]:
    """The deepest existing directory at or above ``path``.

    The staging directory is created by the staging step, not by this probe,
    so its free space is measured on the mount that will hold it.
    """
    for candidate in (path, *path.parents):
        if candidate.is_dir():
            return candidate
    return None


class LaunchReadinessProbe:
    """Runs every check against one :class:`LaunchTarget`.

    Every filesystem and subprocess touch goes through an injected callable,
    so each check can be driven against a stalled read, an unreadable sysfs
    knob or a busy GPU without needing such a node.
    """

    def __init__(
        self,
        target: LaunchTarget,
        *,
        read_text: Callable[[str], str] = _read_sysfs,
        read_mounts: Optional[Callable[[], str]] = None,
        free_bytes_of: Optional[Callable[[Path], int]] = None,
        port_probe: Optional[Callable[[int], bool]] = None,
        run_nvidia_smi: Callable[[Sequence[str], float], str] = _run_nvidia_smi,
        opener: Callable[..., Any] = open,
        read_deadline_s: float = DEFAULT_READ_DEADLINE_S,
        staging: Any = None,
    ) -> None:
        self.target = target
        self._read_text = read_text
        self._read_mounts = read_mounts
        self._free_bytes_of = free_bytes_of
        self._port_probe = port_probe or is_port_free
        self._run_nvidia_smi = run_nvidia_smi
        self._staging_module = staging
        self._opener = opener
        self._read_deadline_s = read_deadline_s
        # Read once and shared by the staging and store-config checks, so a
        # malformed config is reported by one check rather than by every one.
        self._weight_config: Optional[Dict[str, Any]] = None
        self._weight_config_error: Optional[str] = None
        self._weight_config_loaded = False

    # -- the staging module owns the mount rules; this only calls them -------

    def _staging(self):
        """The module that owns the staging mount rules and the store loader.

        Imported here, not at module load, so this probe can be imported in an
        environment where the runtime cannot: the staging module reaches the
        model package's store loader, which reaches the heavy runtime.
        """
        if self._staging_module is None:
            from batchgen.server import weight_store_staging

            self._staging_module = weight_store_staging
        return self._staging_module

    def _weight_store_config(self) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """The node config this launch names, parsed once."""
        if self._weight_config_loaded:
            return self._weight_config, self._weight_config_error
        self._weight_config_loaded = True
        path = self.target.distributed_weight_config
        if path is None:
            return None, None
        try:
            # The schema half of the loader: it normalizes store_residency,
            # store_staging_dir and the topology without judging the store
            # bytes, which the bounded read probe and the manifest gate do.
            self._weight_config = self._staging().read_distributed_weight_config(path)
        except Exception as error:  # noqa: BLE001 - the reason is the evidence
            self._weight_config_error = f"{path} is unusable: {error}"
        return self._weight_config, self._weight_config_error

    # -- 1. the staging tmpfs ------------------------------------------------

    def check_staging_tmpfs(self) -> CheckResult:
        name = "staging-tmpfs"
        config, error = self._weight_store_config()
        if error is not None:
            return CheckResult(
                name,
                STATUS_FAIL,
                f"cannot tell which staging directory this launch needs: {error}",
            )
        if config is None:
            if self.target.fast_init:
                return CheckResult(
                    name,
                    STATUS_OK,
                    "--fast-init without a distributed weight config names no "
                    "staging directory; the huge-page gate this launch does "
                    "depend on is the THP shmem check",
                )
            return CheckResult(
                name, STATUS_OK, "no distributed weight config: nothing is staged"
            )
        if config.get("store_residency") is None:
            return CheckResult(
                name,
                STATUS_OK,
                "the config sets no store_residency, so the store is mapped "
                "from store_path directly and nothing is staged",
            )
        staging = self._staging()
        staging_dir = Path(str(config["store_staging_dir"]))
        read_mounts = self._read_mounts or staging.read_proc_mounts
        try:
            # The server's own gate, called rather than restated: a mount that
            # it would refuse must be refused here identically.
            entry = staging.check_staging_mount(
                staging_dir,
                fast_init=self.target.fast_init,
                read_mounts=read_mounts,
            )
        except staging.WeightStoreStagingError as refusal:
            return CheckResult(name, STATUS_FAIL, str(refusal))
        except OSError as error:
            return CheckResult(
                name,
                STATUS_FAIL,
                f"cannot check the mount holding {staging_dir}: {error}",
            )
        needed = int(config["store_bytes"]) + STAGING_HEADROOM_BYTES
        measured = _nearest_existing(staging_dir)
        if measured is None:
            return CheckResult(
                name,
                STATUS_FAIL,
                f"neither {staging_dir} nor any directory above it exists, so "
                "the staging tmpfs cannot be sized",
            )
        free_bytes_of = self._free_bytes_of or staging.free_bytes
        try:
            available = int(free_bytes_of(measured))
        except OSError as error:
            return CheckResult(
                name, STATUS_FAIL, f"cannot read the free space of {measured}: {error}"
            )
        if available < needed:
            return CheckResult(
                name,
                STATUS_FAIL,
                f"the tmpfs holding {staging_dir} has {available} B free but "
                f"staging this store needs {needed} B (store {config['store_bytes']} B "
                f"plus {STAGING_HEADROOM_BYTES} B headroom); free it or remount "
                f"it larger before launching",
            )
        return CheckResult(
            name,
            STATUS_OK,
            f"{staging_dir} is on {entry.fstype} mount {entry.mountpoint} "
            f"(huge={entry.option('huge') or 'never'}, size option "
            f"{entry.option('size') or 'unset'}) with {available} B free for "
            f"the {needed} B this launch stages",
        )

    # -- 2. the kernel's free-memory watermark -------------------------------

    def check_watermark_scale_factor(self) -> CheckResult:
        name = "watermark-scale-factor"
        try:
            raw = self._read_text(_WATERMARK_SCALE_FACTOR_PATH).strip()
            value = int(raw)
        except (OSError, ValueError) as error:
            return CheckResult(
                name,
                STATUS_WARN,
                f"cannot read {_WATERMARK_SCALE_FACTOR_PATH} ({error}); the "
                "launch's huge-page and pinned allocations are not pre-judged",
            )
        if value > MAX_WATERMARK_SCALE_FACTOR:
            return CheckResult(
                name,
                STATUS_FAIL,
                f"vm.watermark_scale_factor is {value}, above {MAX_WATERMARK_SCALE_FACTOR}: "
                f"the kernel holds {value / 100:.1f}% of each zone free and "
                "reclaims to keep it there, which starves the bulk huge-page "
                "and pinned allocations a cold start makes. Lower it with: "
                f"sysctl -w vm.watermark_scale_factor={MAX_WATERMARK_SCALE_FACTOR}",
            )
        return CheckResult(
            name,
            STATUS_OK,
            f"vm.watermark_scale_factor={value} (bound {MAX_WATERMARK_SCALE_FACTOR})",
        )

    # -- 3. transparent huge pages for shared memory -------------------------

    def check_thp_shmem(self) -> CheckResult:
        name = "thp-shmem"
        # --fast-init is refused outright by the server when THP shmem is off,
        # so the probe must refuse it too: a warning here would pass a node
        # the launch is about to reject.
        failing_status = STATUS_FAIL if self.target.fast_init else STATUS_WARN
        try:
            line = self._read_text(_THP_SHMEM_ENABLED_PATH).strip()
        except OSError as error:
            return CheckResult(
                name,
                failing_status,
                f"cannot read {_THP_SHMEM_ENABLED_PATH} ({error}); this kernel "
                "may have no THP support at all",
            )
        match = re.search(r"\[(\w+)\]", line)
        active = match.group(1) if match else line
        if active not in THP_SHMEM_VALUES:
            return CheckResult(
                name,
                failing_status,
                f"THP shmem_enabled is {active!r}, not one of {THP_SHMEM_VALUES}: "
                "shared memory is backed by 4 KiB pages. Fix it with: "
                f"echo always > {_THP_SHMEM_ENABLED_PATH}",
            )
        return CheckResult(name, STATUS_OK, f"THP shmem_enabled={active}")

    # -- 4. the ports this launch binds --------------------------------------

    def _ephemeral_port_range(self) -> Optional[Tuple[int, int]]:
        try:
            low, high = self._read_text(_IP_LOCAL_PORT_RANGE_PATH).split()[:2]
            return int(low), int(high)
        except (OSError, ValueError, IndexError):
            return None

    def _reserved_ports(self) -> frozenset:
        """Ports the kernel is told never to hand out as ephemeral."""
        reserved: set = set()
        try:
            text = self._read_text(_IP_LOCAL_RESERVED_PORTS_PATH).strip()
        except OSError:
            return frozenset()
        for part in filter(None, (piece.strip() for piece in text.split(","))):
            low, _, high = part.partition("-")
            try:
                reserved.update(range(int(low), int(high or low) + 1))
            except ValueError:
                continue
        return frozenset(reserved)

    def _launch_ports(self) -> List[Tuple[str, int]]:
        """Every port this node binds, as the launch's own validation derives
        them: the listen port always, and the rendezvous ports only on the
        node that hosts them."""
        target = self.target
        ports: List[Tuple[str, int]] = []
        if target.listen_port is not None:
            ports.append(("listen", int(target.listen_port)))
        if target.node_rank != 0:
            return ports
        if target.nnodes > 1 and target.dist_init_addr:
            try:
                _, dist_port = _parse_host_port(str(target.dist_init_addr))
            except ValueError:
                # A malformed address is the launch validator's refusal to
                # make, not this probe's: it reports the ports it could derive.
                dist_port = None
            if dist_port is not None:
                ports.append(("dist_init_addr", dist_port))
                control = target.restart_control_addr
                if control:
                    try:
                        _, control_port = _parse_host_port(str(control))
                    except ValueError:
                        control_port = None
                    if control_port is not None:
                        ports.append(("restart_control_addr", control_port))
                else:
                    ports.append(("restart_control_addr", dist_port + 1))
        if target.world_size > 1 and target.pynccl_port_base is not None:
            ports.append(("pynccl_port_base", int(target.pynccl_port_base)))
        return ports

    def check_ports(self) -> CheckResult:
        name = "ports"
        ports = self._launch_ports()
        if not ports:
            return CheckResult(name, STATUS_OK, "this launch names no local port")
        ephemeral = self._ephemeral_port_range()
        reserved = self._reserved_ports()
        taken: List[str] = []
        ephemeral_risk: List[str] = []
        evidence: List[str] = []
        for label, port in ports:
            free = self._port_probe(port)
            if not free:
                taken.append(f"{label} port {port} is already in use")
            in_ephemeral = (
                ephemeral is not None
                and ephemeral[0] <= port <= ephemeral[1]
                and port not in reserved
            )
            if in_ephemeral:
                ephemeral_risk.append(
                    f"{label} port {port} falls inside ip_local_port_range "
                    f"{ephemeral[0]}-{ephemeral[1]} and is not in "
                    "ip_local_reserved_ports, so an outbound connection of "
                    "any process on this node can take it before the launch "
                    f"binds it; reserve it with: sysctl -w "
                    f"net.ipv4.ip_local_reserved_ports={port}"
                )
            evidence.append(f"{label}={port} ({'free' if free else 'in use'})")
        if taken:
            return CheckResult(name, STATUS_FAIL, "; ".join(taken))
        if ephemeral_risk:
            return CheckResult(name, STATUS_WARN, "; ".join(ephemeral_risk))
        return CheckResult(name, STATUS_OK, ", ".join(evidence))

    # -- 5. every path the launch names, read with a hard deadline -----------

    def _probe_paths(self) -> List[Tuple[str, Path, int]]:
        """``(label, path, bytes to read)`` for every path a launch reads."""
        target = self.target
        probes: List[Tuple[str, Path, int]] = []
        if target.cache_dir is not None:
            probes.append(("model config", target.cache_dir / "config.json", 1 << 16))
        converted = target.effective_converted_ckpt_dir()
        if converted is not None:
            sidecars: Sequence[Path] = ()
            try:
                sidecars = sorted(converted.glob("*.json"))
            except OSError:
                sidecars = ()
            if sidecars:
                probes.append(
                    ("converted checkpoint sidecar", sidecars[0], 1 << 16)
                )
            else:
                # Reported as an unreadable path rather than silently skipped:
                # a converted-checkpoint directory with no sidecar is not a
                # directory the launch can load from.
                probes.append(
                    ("converted checkpoint sidecar", converted / "*.json", 1 << 16)
                )
        config, error = self._weight_store_config()
        if error is None and config is not None:
            probes.append(
                ("weight store", Path(str(config["store_path"])), PROBE_READ_BYTES)
            )
        return probes

    def check_path_reads(self) -> CheckResult:
        name = "path-reads"
        probes = self._probe_paths()
        if not probes:
            return CheckResult(
                name, STATUS_OK, "this launch names no path to read ahead of it"
            )
        failures: List[str] = []
        evidence: List[str] = []
        for label, path, size in probes:
            try:
                read = read_with_deadline(
                    path,
                    size=size,
                    deadline_s=self._read_deadline_s,
                    opener=self._opener,
                )
            except StalledReadError as stalled:
                failures.append(f"{label}: {stalled}")
            except OSError as error:
                failures.append(f"{label} {path} cannot be read: {error}")
            else:
                evidence.append(f"{label} {path} ({read} B read)")
        if failures:
            return CheckResult(name, STATUS_FAIL, "; ".join(failures))
        return CheckResult(name, STATUS_OK, "; ".join(evidence))

    # -- 6. the GPUs this launch takes --------------------------------------

    def check_gpu_idle(self) -> CheckResult:
        name = "gpu-idle"
        argv = (
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader",
        )
        try:
            output = self._run_nvidia_smi(argv, NVIDIA_SMI_TIMEOUT_S)
        except FileNotFoundError:
            return CheckResult(
                name,
                STATUS_WARN,
                "nvidia-smi is not on PATH, so resident GPU processes are not "
                "pre-judged",
            )
        except subprocess.TimeoutExpired:
            return CheckResult(
                name,
                STATUS_FAIL,
                f"nvidia-smi did not answer within {NVIDIA_SMI_TIMEOUT_S:.0f}s; "
                "the driver is not responding, so no launch can take these GPUs",
            )
        except subprocess.CalledProcessError as error:
            return CheckResult(
                name,
                STATUS_WARN,
                f"nvidia-smi exited {error.returncode}, so resident GPU "
                f"processes are not pre-judged: {(error.stderr or '').strip()}",
            )
        processes = [line.strip() for line in output.splitlines() if line.strip()]
        if processes:
            return CheckResult(
                name,
                STATUS_FAIL,
                f"{len(processes)} compute process(es) still hold these GPUs "
                f"({'; '.join(processes)}); a launch would contend with them "
                "for memory and SMs",
            )
        return CheckResult(name, STATUS_OK, "no compute process holds these GPUs")

    # -- 7. the store this launch maps --------------------------------------

    def check_weight_store_config(self) -> CheckResult:
        name = "weight-store-config"
        config, error = self._weight_store_config()
        if error is not None:
            return CheckResult(name, STATUS_FAIL, error)
        path = self.target.distributed_weight_config
        if config is None or path is None:
            return CheckResult(
                name, STATUS_OK, "this launch names no distributed weight config"
            )
        staging = self._staging()
        try:
            # The full loader: the schema AND the completion manifest every
            # consumer of the store goes through, so a build that never
            # published is refused here instead of mid-staging.
            staging.load_distributed_weight_config(path)
        except Exception as refusal:  # noqa: BLE001 - the refusal is the evidence
            return CheckResult(name, STATUS_FAIL, str(refusal))
        return CheckResult(
            name,
            STATUS_OK,
            f"{path} is node {config['node_rank']} of {config['num_nodes']} "
            f"({config['store_bytes']} B store, residency "
            f"{config['store_residency'] or 'none'}) and passes the manifest gate",
        )

    # -- the probe ----------------------------------------------------------

    def run(self) -> LaunchReadinessReport:
        return LaunchReadinessReport(
            tuple(
                check()
                for check in (
                    self.check_weight_store_config,
                    self.check_staging_tmpfs,
                    self.check_watermark_scale_factor,
                    self.check_thp_shmem,
                    self.check_ports,
                    self.check_path_reads,
                    self.check_gpu_idle,
                )
            )
        )


def probe_launch_readiness(target: LaunchTarget, **kwargs: Any) -> LaunchReadinessReport:
    """Every check against ``target``; see :class:`LaunchReadinessProbe`."""
    return LaunchReadinessProbe(target, **kwargs).run()


def run_launch_readiness(server_args: Any, **kwargs: Any) -> LaunchReadinessReport:
    """The startup gate: probe this launch, log every verdict, refuse a failure.

    Called before worker processes, ports, SHM segments or weight bytes exist,
    so a node that cannot serve this launch is refused while nothing has been
    committed yet.
    """
    report = probe_launch_readiness(LaunchTarget.from_server_args(server_args), **kwargs)
    for check in report.checks:
        level = {
            STATUS_OK: logging.INFO,
            STATUS_WARN: logging.WARNING,
            STATUS_FAIL: logging.ERROR,
        }[check.status]
        logger.log(level, "[launch-readiness] %s", check)
    if not report.ok:
        raise LaunchReadinessError(report.message())
    return report


# --------------------------------------------------------------------------- #
#  CLI                                                                        #
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m batchgen.launch_readiness",
        description=(
            "Judge whether this node is ready for the launch described by "
            "these arguments. Exits 0 when every check passes, 1 otherwise."
        ),
    )
    parser.add_argument("--model", required=True, help="Model identifier to launch")
    parser.add_argument(
        "--distributed-weight-config",
        type=Path,
        default=None,
        help="This node's distributed weight store config (node<R>.json)",
    )
    parser.add_argument(
        "--listen-port", type=int, default=None, help="Server listen port"
    )
    parser.add_argument("--cache-dir", type=Path, default=None, help="Model cache dir")
    parser.add_argument(
        "--converted-ckpt-dir",
        type=Path,
        default=None,
        help="Converted checkpoint dir (default: <cache-dir>/converted_ckpt)",
    )
    parser.add_argument(
        "--dist-init-addr", default=None, help="host:port of the rendezvous store"
    )
    parser.add_argument(
        "--restart-control-addr",
        default=None,
        help="host:port of the restart control store (default: dist port + 1)",
    )
    parser.add_argument("--pynccl-port-base", type=int, default=None)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument(
        "--fast-init",
        action="store_true",
        help="Judge the huge-page gates --fast-init depends on",
    )
    parser.add_argument(
        "--read-deadline-s",
        type=float,
        default=DEFAULT_READ_DEADLINE_S,
        help="Hard deadline on each bounded probe read",
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit the results as JSON"
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    target = LaunchTarget(
        model=args.model,
        distributed_weight_config=args.distributed_weight_config,
        listen_port=args.listen_port,
        cache_dir=args.cache_dir,
        converted_ckpt_dir=args.converted_ckpt_dir,
        fast_init=args.fast_init,
        dist_init_addr=args.dist_init_addr,
        restart_control_addr=args.restart_control_addr,
        pynccl_port_base=args.pynccl_port_base,
        nnodes=args.nnodes,
        node_rank=args.node_rank,
        world_size=args.world_size,
    )
    report = probe_launch_readiness(target, read_deadline_s=args.read_deadline_s)
    if args.json:
        print(json.dumps(report.to_json(), indent=2))  # noqa: hygiene - CLI output
    else:
        for check in report.checks:
            print(str(check))  # noqa: hygiene - CLI output
        if not report.ok:
            print(report.message(), file=sys.stderr)  # noqa: hygiene - CLI output
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CheckResult",
    "LaunchReadinessError",
    "LaunchReadinessProbe",
    "LaunchReadinessReport",
    "LaunchTarget",
    "StalledReadError",
    "is_port_free",
    "main",
    "probe_launch_readiness",
    "read_with_deadline",
    "run_launch_readiness",
]
