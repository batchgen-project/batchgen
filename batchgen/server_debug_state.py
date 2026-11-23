"""Server-level debug state: the one registry of debug keys and the state object.

A debug switch belongs to the SERVER, not to a request. The server holds one
immutable :class:`DebugState`; an operator changes it through the admin
endpoint, every rank receives the same generation, and a worker adopts it only
at a phase entry (the start of a prefill wave or of a decode entry), never
inside a wave or a decode step.

Every key is declared exactly once, by its owner (core keys below, a model
family's keys in that family's package), with:

* ``owner``  - ``core`` or the model family that reads it;
* ``phase``  - which phase reads it (``prefill``, ``decode`` or ``both``);
* ``tier``   - what a change costs:
    - :data:`TIER_PHASE_ENTRY`: takes effect at the next phase entry;
    - :data:`TIER_REWARM`: the value is baked into captured decode graphs or
      into a phase instance, so the worker pool must be re-warmed (host
      weights and host KV stay untouched);
* ``parse``  - validation and normalisation of the value. An unknown key or a
  value the parser rejects is refused by name; nothing is silently ignored.

No key changes the size of host weights or host KV, so no key needs a cold
start.

A serving window that saw a non-empty state is not an official result: each
adoption logs one ``[DEBUG_STATE]`` line (:func:`format_log_line`) that the
benchmark recorder reads.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Optional

TIER_PHASE_ENTRY = "phase_entry"
TIER_REWARM = "rewarm"
TIERS = (TIER_PHASE_ENTRY, TIER_REWARM)
PHASES = ("prefill", "decode", "both")

LOG_PREFIX = "[DEBUG_STATE]"
ADMIN_ENDPOINT = "/v1/admin/debug-state"


class DebugStateError(ValueError):
    """An unknown key or a refused value; the message names the key."""


# ---------------------------------------------------------------- value parsers

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off", ""})


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value in (0, 1):
            return bool(value)
    elif isinstance(value, str):
        text = value.strip().lower()
        if text in _TRUE:
            return True
        if text in _FALSE:
            return False
    raise ValueError("must be true or false")


def parse_choice(*allowed: str) -> Callable[[Any], str]:
    def parse(value: Any) -> str:
        if isinstance(value, str) and value.strip().lower() in allowed:
            return value.strip().lower()
        raise ValueError("must be one of " + ", ".join(allowed))
    return parse


def parse_int(*, minimum: int) -> Callable[[Any], int]:
    def parse(value: Any) -> int:
        if isinstance(value, bool):
            raise ValueError(f"must be an integer >= {minimum}")
        try:
            number = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"must be an integer >= {minimum}") from None
        if number < minimum or (isinstance(value, float) and value != number):
            raise ValueError(f"must be an integer >= {minimum}")
        return number
    return parse


def parse_number(*, minimum: float) -> Callable[[Any], float]:
    def parse(value: Any) -> float:
        if isinstance(value, bool):
            raise ValueError(f"must be a finite number >= {minimum}")
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"must be a finite number >= {minimum}") from None
        if number != number or number in (float("inf"), float("-inf")) or number < minimum:
            raise ValueError(f"must be a finite number >= {minimum}")
        return number
    return parse


def parse_path(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value
    raise ValueError("must be a non-empty path string")


def _dump_kernel_args(value: Any) -> str:
    if isinstance(value, str) and value.strip().lower() in ("sync", "nosync"):
        return value.strip().lower()
    if parse_bool(value):
        return "sync"
    raise ValueError("must be sync or nosync")


def parse_rank_list(value: Any) -> tuple:
    if isinstance(value, (list, tuple)) and value and all(
        isinstance(rank, int) and not isinstance(rank, bool) and rank >= 0 for rank in value
    ):
        return tuple(value)
    raise ValueError("must be a non-empty list of rank numbers")


def _weight_pass_end_delay(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"rank", "seconds"}:
        raise ValueError("must be an object with exactly rank and seconds")
    rank, seconds = value["rank"], value["seconds"]
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
        raise ValueError("rank must be a rank number")
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not 0 < seconds <= 60:
        raise ValueError("seconds must be a number in (0, 60]")
    return MappingProxyType({"rank": rank, "seconds": float(seconds)})


def parse_layer_list(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip().lower()
        if text == "all":
            return "all"
        parts = [part.strip() for part in text.split(",") if part.strip()]
        if parts and all(part.isdigit() for part in parts):
            return tuple(int(part) for part in parts)
    elif isinstance(value, (list, tuple)) and value and all(
        isinstance(layer, int) and not isinstance(layer, bool) and layer >= 0 for layer in value
    ):
        return tuple(value)
    raise ValueError('must be "all", a comma-separated list of layer numbers, or a list of them')


def parse_object(value: Any) -> Mapping[str, Any]:
    # The owning module validates the inner structure when it builds the probe.
    if isinstance(value, Mapping) and value:
        return MappingProxyType(dict(value))
    raise ValueError("must be a non-empty object")


# ---------------------------------------------------------------- key declarations

@dataclass(frozen=True)
class DebugKey:
    name: str
    owner: str
    phase: str
    tier: str
    parse: Callable[[Any], Any]
    summary: str


def key(name, owner, phase, parse, summary, tier=TIER_PHASE_ENTRY) -> DebugKey:
    return DebugKey(name=name, owner=owner, phase=phase, tier=tier, parse=parse, summary=summary)


# Keys read by core code. A model family declares its own keys in its own
# package; ``batchgen/models/debug_keys.py`` assembles the server's registry.
CORE_KEYS = (
    key("prefix_pool_epilogue_trace", "core", "prefill", parse_bool,
         "trace the pooled-prefill epilogue"),
    key("prefix_pool_epilogue_probe_token", "core", "prefill", parse_int(minimum=0),
         "token id whose logit the epilogue trace reports"),
    key("prefix_pool_dump_kernel_args", "core", "prefill", _dump_kernel_args,
         "dump pooled-prefill kernel arguments (sync or nosync)"),
    key("prefix_pool_dump_dir", "core", "prefill", parse_path,
         "directory of the kernel-argument dump"),
    key("prefix_pool_pad_poison", "core", "prefill", parse_bool,
         "poison page-table padding to expose out-of-range reads"),
    key("prefix_pool_offload_check", "core", "prefill", parse_bool,
         "byte-check every prefix-pool offload"),
    key("weight_pass_end_delay", "core", "prefill", _weight_pass_end_delay,
         "sleep one rank at the end of a weight pass to inject rank skew"),
    key("decode_profile_steps", "core", "decode", parse_int(minimum=1),
         "number of decode steps the profiler window covers"),
    key("decode_profile_ranks", "core", "decode", parse_rank_list,
         "ranks the decode profiler runs on"),
    key("decode_profile_trace", "core", "decode", parse_path,
         "file the decode profiler exports its trace to"),
    key("triton_strict", "core", "both", parse_bool,
         "raise on a kernel compile after readiness"),
    key("prefix_trajectory_commit", "core", "both", parse_choice("on", "off"),
         "commit decode trajectories into the prefix pool"),
)

CORE_RETIRED = {
    "prefix_rank_assignment_mode":
        "the prefix rank assignment is the launch parameter --prefix-rank-assignment-mode",
    "prefix_rank_cost_profile":
        "the prefix rank cost profile is the launch parameter --prefix-rank-cost-profile",
}


# ---------------------------------------------------------------- registry and state

class Registry:
    """The keys one server accepts, and the retired names it refuses with a reason."""

    def __init__(self, keys: Iterable[DebugKey], retired: Optional[Mapping[str, str]] = None):
        table = {}
        for item in keys:
            if item.name in table:
                raise ValueError(f"debug key {item.name} is declared twice")
            if item.phase not in PHASES or item.tier not in TIERS or not item.summary:
                raise ValueError(f"debug key {item.name} has an incomplete declaration")
            table[item.name] = item
        self.keys: Mapping[str, DebugKey] = MappingProxyType(table)
        self.retired: Mapping[str, str] = MappingProxyType(dict(retired or {}))
        overlap = set(self.keys) & set(self.retired)
        if overlap:
            raise ValueError(f"debug keys both declared and retired: {sorted(overlap)}")
        self.empty = DebugState(self, 0, MappingProxyType({}))

    def check_name(self, name: str) -> DebugKey:
        if name in self.retired:
            raise DebugStateError(f"debug key {name} is retired: {self.retired[name]}")
        item = self.keys.get(name)
        if item is None:
            raise DebugStateError(f"unknown debug key {name!r}")
        return item

    def validate(self, name: str, value: Any) -> Any:
        """The normalised value of ``name``; raises :class:`DebugStateError` naming the key."""
        item = self.check_name(name)
        try:
            return item.parse(value)
        except ValueError as error:
            raise DebugStateError(f"debug key {name} {error}; got {value!r}") from None

    def from_wire(self, wire: Optional[Mapping[str, Any]]) -> "DebugState":
        if not wire:
            return self.empty
        values = {name: self.validate(name, value)
                  for name, value in (wire.get("values") or {}).items()}
        return DebugState(self, int(wire.get("generation", 0)), MappingProxyType(values))


@dataclass(frozen=True)
class DebugState:
    """One immutable generation of the server's debug switches."""

    registry: Registry
    generation: int
    values: Mapping[str, Any]

    def get(self, name: str, default: Any = None) -> Any:
        """The value of a REGISTERED key (``default`` when it is not set)."""
        self.registry.check_name(name)
        return self.values.get(name, default)

    def __bool__(self) -> bool:
        return bool(self.values)

    def changed(self, *, set: Optional[Mapping[str, Any]] = None,
                clear: Iterable[str] = ()) -> "DebugState":
        """The next generation after setting and clearing keys (validated)."""
        values = dict(self.values)
        for name in clear:
            self.registry.check_name(name)
            values.pop(name, None)
        for name, value in (set or {}).items():
            values[name] = self.registry.validate(name, value)
        return DebugState(self.registry, self.generation + 1, MappingProxyType(values))

    def rewarm_keys_changed(self, other: "DebugState") -> tuple:
        """Names of re-warm keys whose value differs between the two states."""
        return tuple(sorted(
            name for name, item in self.registry.keys.items()
            if item.tier == TIER_REWARM and self.values.get(name) != other.values.get(name)
        ))

    def to_wire(self) -> dict:
        return {"generation": self.generation, "values": _plain(self.values)}


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {name: _plain(item) for name, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def format_log_line(state: DebugState, *, node: int) -> str:
    """``[DEBUG_STATE] generation=G node=N keys=<k=v,...|none>`` (read by the recorder)."""
    if state.values:
        keys = ",".join(f"{name}={_plain(state.values[name])!r}".replace(" ", "")
                        for name in sorted(state.values))
    else:
        keys = "none"
    return f"{LOG_PREFIX} generation={state.generation} node={node} keys={keys}"


def request_field_refusal() -> str:
    """Why a request carrying ``batchgen_debug`` is refused."""
    return ("batchgen_debug is no longer accepted on a request: debug switches belong to the "
            f"server. Set them with POST {ADMIN_ENDPOINT} before submitting.")
