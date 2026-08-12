# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                          #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
# ---------------------------------------------------------------------------- #

"""Reconcile a HuggingFace checkpoint against BatchGen's weight plumbing.

Offline (CPU, no GPU, no weight bytes read) proof that the three declarations a
model must get right actually agree with the artifact on disk:

  * ``state_dict_name_map``   ckpt tensor name -> (module_key, tensor_key), consumed
                              by ``Parameter_Server::Init``.
  * ``module_shapes``         the GPU ring-slot geometry, consumed by
                              ``GPU_Weight_Buffer::Init``.
  * the skeleton / ignore partition of everything the name map does not cover.

Why this exists.  Every disagreement between those three is SILENT today:

  * a checkpoint tensor with no name-map entry is promoted to
    ``skeleton_state_dict_`` without a word (``Parameter_Server.cpp:357-397``);
  * a ``module_shapes`` key that no checkpoint tensor writes leaves the GPU slot
    at its ``torch::zeros`` init, or holding the previous module's bytes
    (``GPU_Weight_Buffer.cpp:121-122`` + slot reuse) — this is the GLM-5 class,
    in-code postmortem at ``models/glm/glm5/glm5_initializer.py:141-146``;
  * a host tensor with no matching slot takes the ``dst[tensor_name]`` undefined
    branch in ``HtoD_Engine.cu:446-451``, which logs and continues;
  * a wrong shape/dtype is copied by ``blocking_copy_`` (``HtoD_Engine.cu:232-238``)
    with NO destination bound check.

Model-agnostic on purpose: it knows nothing about any model.  A model supplies a
:class:`ReconcileSpec` of plain data and gets a :class:`ReconcileReport` back.

Typical use, from a model's parameter server at startup::

    from batchgen.models.weight_reconciler import read_index, reconcile
    ckpt = read_index(os.path.join(cache_dir, "model.safetensors.index.json"))
    reconcile(ckpt, my_spec).raise_for_status()

or, stronger, on a machine that has the shards mounted::

    ckpt = read_safetensors_headers(cache_dir)   # per-tensor shape + dtype
    reconcile(ckpt, my_spec).raise_for_status()
"""

import json
import os
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch


# --------------------------------------------------------------------------- #
#  Checkpoint side                                                             #
# --------------------------------------------------------------------------- #

_SAFETENSORS_DTYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}
for _st_name, _torch_attr in (("F8_E4M3", "float8_e4m3fn"),
                              ("F8_E5M2", "float8_e5m2")):
    if hasattr(torch, _torch_attr):
        _SAFETENSORS_DTYPES[_st_name] = getattr(torch, _torch_attr)

_ITEMSIZE: Dict[torch.dtype, int] = {}


def itemsize(dtype: torch.dtype) -> int:
    """Bytes per element, cached (``torch.empty`` per call is not free at 500k)."""
    size = _ITEMSIZE.get(dtype)
    if size is None:
        size = torch.empty((), dtype=dtype).element_size()
        _ITEMSIZE[dtype] = size
    return size


def nbytes(shape: Sequence[int], dtype: torch.dtype) -> int:
    n = 1
    for dim in shape:
        n *= int(dim)
    return n * itemsize(dtype)


@dataclass(frozen=True)
class CkptTensor:
    shape: Tuple[int, ...]
    dtype: torch.dtype
    nbytes: int


@dataclass
class CheckpointIndex:
    """What is actually in the checkpoint.

    ``tensors`` is populated only by :func:`read_safetensors_headers`; from the
    HF index alone we get names and the aggregate ``total_bytes`` but no shapes.
    """

    tensor_names: Set[str]
    total_bytes: int
    num_shards: int = 0
    tensors: Optional[Dict[str, CkptTensor]] = None


def read_index(index_path: str) -> CheckpointIndex:
    """Read ``model.safetensors.index.json``. Cheap: one JSON, no weight bytes."""
    with open(index_path, "r") as handle:
        index = json.load(handle)
    weight_map = index["weight_map"]
    total = int(index["metadata"]["total_size"])
    return CheckpointIndex(
        tensor_names=set(weight_map),
        total_bytes=total,
        num_shards=len(set(weight_map.values())),
    )


def read_safetensors_headers(ckpt_dir: str) -> CheckpointIndex:
    """Read every shard's safetensors header: name -> (shape, dtype, nbytes).

    Reads only the JSON header of each shard (a few hundred KB), never a weight
    byte, so it is fast even on a 1.5 TB checkpoint — but it must run where the
    shards are mounted.
    """
    files = sorted(
        f for f in os.listdir(ckpt_dir) if f.endswith(".safetensors")
    )
    if not files:
        raise FileNotFoundError(f"No .safetensors shards under {ckpt_dir}")
    tensors: Dict[str, CkptTensor] = {}
    for file_name in files:
        with open(os.path.join(ckpt_dir, file_name), "rb") as handle:
            header_len = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_len))
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            dtype = _SAFETENSORS_DTYPES.get(meta["dtype"])
            if dtype is None:
                raise ValueError(
                    f"{file_name}: tensor {name} has safetensors dtype "
                    f"{meta['dtype']!r}, which this torch build cannot express."
                )
            start, end = meta["data_offsets"]
            tensors[name] = CkptTensor(
                shape=tuple(int(s) for s in meta["shape"]),
                dtype=dtype,
                nbytes=int(end) - int(start),
            )
    return CheckpointIndex(
        tensor_names=set(tensors),
        total_bytes=sum(t.nbytes for t in tensors.values()),
        num_shards=len(files),
        tensors=tensors,
    )


# --------------------------------------------------------------------------- #
#  Engine side                                                                 #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class IgnoreRule:
    """A checkpoint prefix that is deliberately not loaded, with the reason.

    Required, not optional: without it an unloaded tensor is indistinguishable
    from a forgotten one, and the C++ parameter server swallows both.
    """

    prefix: str
    reason: str


@dataclass
class ReconcileSpec:
    """Everything BatchGen declares about a checkpoint, as plain data."""

    #: ckpt name -> {"module_key": ..., "tensor_key": ...}
    name_map: Dict[str, Dict[str, str]]
    #: module_type -> {tensor_key -> shape}
    module_shapes: Dict[str, Dict[str, Sequence[int]]]
    #: module_type -> default slot dtype
    weight_dtypes: Dict[str, torch.dtype]
    #: module_type -> {tensor_key -> dtype}, overriding weight_dtypes
    tensor_dtypes: Dict[str, Dict[str, torch.dtype]] = field(default_factory=dict)
    #: ckpt name -> (shape, dtype) for resident/skeleton tensors (not in name_map)
    skeleton: Dict[str, Tuple[Sequence[int], torch.dtype]] = field(default_factory=dict)
    ignore_rules: Sequence[IgnoreRule] = ()
    #: pinned expectations for the ignored set; a change must fail loudly
    ignored_count: Optional[int] = None
    ignored_bytes: Optional[int] = None
    #: suffix pair identifying a quantized (packed, scale) couple
    quant_pair_suffixes: Tuple[str, str] = (".weight_packed", ".weight_scale")


def _module_type(module_key: str, module_shapes: Dict[str, Dict]) -> Optional[str]:
    """``routed_expert_1_5`` -> ``routed_expert``; ``kda_attn_0`` -> ``kda_attn``."""
    key = module_key
    while True:
        if key in module_shapes:
            return key
        cut = key.rfind("_")
        if cut < 0:
            return None
        key = key[:cut]


def _match_ignore(name: str, rules: Sequence[IgnoreRule]) -> Optional[IgnoreRule]:
    for rule in rules:
        if name.startswith(rule.prefix):
            return rule
    return None


# --------------------------------------------------------------------------- #
#  Report                                                                      #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Finding:
    bucket: str
    key: str
    detail: str


@dataclass
class ReconcileReport:
    n_checkpoint: int = 0
    n_mapped: int = 0
    n_skeleton: int = 0
    n_ignored: int = 0
    checkpoint_bytes: int = 0
    declared_bytes_by_type: Dict[str, int] = field(default_factory=dict)
    declared_skeleton_bytes: int = 0
    declared_ignored_bytes: Optional[int] = None
    findings: List[Finding] = field(default_factory=list)
    #: bucket -> total number of findings (the ``findings`` list is capped)
    counts: Dict[str, int] = field(default_factory=dict)
    #: checks that could not run, with the reason. Never a silent pass.
    skipped: List[str] = field(default_factory=list)
    max_findings: int = 50

    # -- construction helpers ------------------------------------------- #
    def add(self, bucket: str, key: str, detail: str) -> None:
        self.counts[bucket] = self.counts.get(bucket, 0) + 1
        if self.counts[bucket] <= self.max_findings:
            self.findings.append(Finding(bucket, key, detail))

    def skip(self, reason: str) -> None:
        self.skipped.append(reason)

    # -- verdict --------------------------------------------------------- #
    @property
    def n_unaccounted(self) -> int:
        return self.counts.get("unaccounted", 0)

    @property
    def declared_bytes(self) -> int:
        return (
            sum(self.declared_bytes_by_type.values())
            + self.declared_skeleton_bytes
            + (self.declared_ignored_bytes or 0)
        )

    @property
    def ok(self) -> bool:
        return not self.counts and not self.skipped

    def raise_for_status(self, allow_skipped: bool = False) -> "ReconcileReport":
        if self.counts or (self.skipped and not allow_skipped):
            raise WeightReconcileError(self.render())
        return self

    def render(self) -> str:
        lines = [
            "checkpoint reconciliation",
            f"  checkpoint tensors   : {self.n_checkpoint:,}",
            f"  -> module rings      : {self.n_mapped:,}",
            f"  -> skeleton          : {self.n_skeleton:,}",
            f"  -> explicitly ignored: {self.n_ignored:,}",
            f"  -> UNACCOUNTED       : {self.n_unaccounted:,}",
            "  declared bytes:",
        ]
        for module_type in sorted(self.declared_bytes_by_type):
            lines.append(
                f"    {module_type:<18}: "
                f"{self.declared_bytes_by_type[module_type]:>18,}"
            )
        lines += [
            f"    {'skeleton':<18}: {self.declared_skeleton_bytes:>18,}",
            f"    {'ignored':<18}: {(self.declared_ignored_bytes or 0):>18,}",
            f"    {'declared total':<18}: {self.declared_bytes:>18,}",
            f"    {'checkpoint total':<18}: {self.checkpoint_bytes:>18,}",
            f"    {'delta':<18}: "
            f"{self.declared_bytes - self.checkpoint_bytes:>+18,}",
        ]
        if self.skipped:
            lines.append("  SKIPPED CHECKS (not a pass):")
            lines += [f"    - {reason}" for reason in self.skipped]
        if self.counts:
            lines.append("  FINDINGS:")
            for bucket in sorted(self.counts):
                lines.append(f"    [{bucket}] {self.counts[bucket]:,}")
            for finding in self.findings:
                lines.append(
                    f"      [{finding.bucket}] {finding.key}: {finding.detail}"
                )
            shown = len(self.findings)
            total = sum(self.counts.values())
            if total > shown:
                lines.append(f"      ... {total - shown:,} further finding(s) suppressed")
        else:
            lines.append("  FINDINGS: none")
        return "\n".join(lines)


class WeightReconcileError(RuntimeError):
    """Raised by :meth:`ReconcileReport.raise_for_status`. Never downgraded."""


# --------------------------------------------------------------------------- #
#  The reconciliation                                                          #
# --------------------------------------------------------------------------- #

def reconcile(checkpoint: CheckpointIndex, spec: ReconcileSpec,
              max_findings: int = 50) -> ReconcileReport:
    """Cross-check a checkpoint against a BatchGen weight declaration.

    Checks, each of which is silent in production today:

    ``unaccounted``          checkpoint tensor with no destination at all.
    ``double_declared``      tensor claimed by both the name map and the skeleton.
    ``dangling``             declaration naming a tensor the checkpoint lacks.
    ``slot_never_written``   ``module_shapes`` key no checkpoint tensor writes.
    ``source_without_slot``  mapped tensor with no ``module_shapes`` slot.
    ``unknown_module_type``  ``module_key`` that resolves to no declared type.
    ``module_type_unused``   declared type nothing is mapped to.
    ``quant_pair``           packed/scale couple broken or half-mapped.
    ``tensor_mismatch``      declared shape/dtype/bytes != the checkpoint's
                             (only when per-tensor headers are available).
    ``byte_total``           sum of declared bytes != the checkpoint total.
    """
    report = ReconcileReport(max_findings=max_findings)
    report.n_checkpoint = len(checkpoint.tensor_names)
    report.checkpoint_bytes = checkpoint.total_bytes
    report.declared_bytes_by_type = {t: 0 for t in spec.module_shapes}

    names = checkpoint.tensor_names
    per_tensor = checkpoint.tensors

    # -- 0. a tensor may not be claimed twice -------------------------- #
    for name in sorted(set(spec.name_map) & set(spec.skeleton)):
        report.add(
            "double_declared", name,
            "claimed by BOTH the name map and the skeleton; the parameter "
            "server files it into the module ring and the model ALSO loads it "
            "from the skeleton, so which value survives depends on call order",
        )

    # -- 1. classify every checkpoint tensor --------------------------- #
    written: Dict[str, Set[str]] = {}   # module_key -> tensor_keys actually sourced
    bytes_incomplete = False
    for name in names:
        entry = spec.name_map.get(name)
        if entry is not None:
            report.n_mapped += 1
            module_key = entry["module_key"]
            tensor_key = entry["tensor_key"]
            written.setdefault(module_key, set()).add(tensor_key)
            module_type = _module_type(module_key, spec.module_shapes)
            shape = None
            if module_type is not None:
                shape = spec.module_shapes[module_type].get(tensor_key)
            if shape is None:
                bytes_incomplete = True
                continue
            dtype = spec.tensor_dtypes.get(module_type, {}).get(
                tensor_key, spec.weight_dtypes[module_type]
            )
            declared = nbytes(shape, dtype)
            report.declared_bytes_by_type[module_type] += declared
            _check_tensor(report, per_tensor, name, shape, dtype, declared)
            continue

        skeleton = spec.skeleton.get(name)
        if skeleton is not None:
            report.n_skeleton += 1
            shape, dtype = skeleton
            declared = nbytes(shape, dtype)
            report.declared_skeleton_bytes += declared
            _check_tensor(report, per_tensor, name, shape, dtype, declared)
            continue

        if _match_ignore(name, spec.ignore_rules) is not None:
            report.n_ignored += 1
            continue

        report.add(
            "unaccounted", name,
            "no name-map entry, no skeleton declaration and no ignore rule — "
            "Parameter_Server.cpp:368 would silently promote it to "
            "skeleton_state_dict_",
        )

    # -- 2. declarations naming tensors the checkpoint does not have ---- #
    for name in spec.name_map:
        if name not in names:
            report.add(
                "dangling", name,
                "name-map entry names a tensor absent from the checkpoint; "
                "nothing is ever copied into its GPU slot",
            )
    for name in spec.skeleton:
        if name not in names:
            report.add(
                "dangling", name,
                "skeleton declaration names a tensor absent from the checkpoint",
            )

    # -- 3. slot coverage, aggregated per (module_type, tensor_key) ----- #
    missing: Dict[Tuple[str, str], List[int]] = {}   # -> [count, example_key]
    extra: Dict[Tuple[str, str], List] = {}
    seen_types: Set[str] = set()
    for module_key, tensor_keys in written.items():
        module_type = _module_type(module_key, spec.module_shapes)
        if module_type is None:
            report.add(
                "unknown_module_type", module_key,
                "module_key resolves to no module_shapes entry; "
                f"declared types are {sorted(spec.module_shapes)}",
            )
            continue
        seen_types.add(module_type)
        declared_keys = set(spec.module_shapes[module_type])
        for tensor_key in declared_keys - tensor_keys:
            slot = missing.setdefault((module_type, tensor_key), [0, module_key])
            slot[0] += 1
        for tensor_key in tensor_keys - declared_keys:
            slot = extra.setdefault((module_type, tensor_key), [0, module_key])
            slot[0] += 1

    for (module_type, tensor_key), (count, example) in sorted(missing.items()):
        report.add(
            "slot_never_written", f"{module_type}.{tensor_key}",
            f"{count:,} module(s) allocate this GPU slot but no checkpoint "
            f"tensor is mapped to it (e.g. {example}); the buffer keeps its "
            "torch::zeros init, or the previous module's bytes on slot reuse",
        )
    for (module_type, tensor_key), (count, example) in sorted(extra.items()):
        report.add(
            "source_without_slot", f"{module_type}.{tensor_key}",
            f"{count:,} module(s) source this tensor but module_shapes declares "
            f"no slot for it (e.g. {example}); HtoD_Engine.cu:446 takes the "
            "'no valid storage' branch and the bytes are dropped",
        )
    for module_type in sorted(set(spec.module_shapes) - seen_types):
        report.add(
            "module_type_unused", module_type,
            "module_shapes declares this type but the name map maps nothing "
            "to it — every slot of the ring stays at its zeros init",
        )

    # -- 4. quantized (packed, scale) couples --------------------------- #
    packed_suffix, scale_suffix = spec.quant_pair_suffixes
    for name in names:
        if name.endswith(packed_suffix):
            sibling = name[: -len(packed_suffix)] + scale_suffix
            if sibling not in names:
                report.add("quant_pair", name,
                           "packed tensor with no matching scale in the checkpoint")
            elif (name in spec.name_map) != (sibling in spec.name_map):
                report.add("quant_pair", name,
                           "only one member of the packed/scale couple is mapped; "
                           "the pair is the unit of meaning and a lone member "
                           "decodes to garbage")
        elif name.endswith(scale_suffix):
            sibling = name[: -len(scale_suffix)] + packed_suffix
            if sibling not in names:
                report.add("quant_pair", name,
                           "scale tensor with no matching packed tensor")

    # -- 5. the ignored set is pinned ----------------------------------- #
    if spec.ignored_count is not None and report.n_ignored != spec.ignored_count:
        report.add(
            "ignored_count", "ignore_rules",
            f"matched {report.n_ignored} tensors, pinned at {spec.ignored_count}; "
            "the ignored region of the checkpoint changed — re-derive its byte "
            "size before trusting any reservation computed from it",
        )
    if per_tensor is not None:
        report.declared_ignored_bytes = sum(
            per_tensor[n].nbytes for n in names
            if n not in spec.name_map and n not in spec.skeleton
            and _match_ignore(n, spec.ignore_rules) is not None
        )
    else:
        report.declared_ignored_bytes = spec.ignored_bytes

    # -- 6. byte reconciliation ----------------------------------------- #
    if bytes_incomplete:
        report.skip(
            "byte-total check: at least one mapped tensor has no module_shapes "
            "slot, so the declared total cannot be computed (see the "
            "source_without_slot / unknown_module_type findings)"
        )
    elif report.declared_ignored_bytes is None and report.n_ignored:
        report.skip(
            "byte-total check: the ignored set is non-empty and its size is "
            "unknown (pass ReconcileSpec.ignored_bytes, or use "
            "read_safetensors_headers so it can be measured)"
        )
    elif report.declared_bytes != report.checkpoint_bytes:
        report.add(
            "byte_total", "sum(declared)",
            f"declared {report.declared_bytes:,} B but the checkpoint holds "
            f"{report.checkpoint_bytes:,} B "
            f"(delta {report.declared_bytes - report.checkpoint_bytes:+,}); at "
            "least one shape or dtype is wrong, and blocking_copy_ "
            "(HtoD_Engine.cu:232-238) copies the source byte size into the "
            "destination with no bound check",
        )
    return report


def _check_tensor(report: ReconcileReport,
                  per_tensor: Optional[Dict[str, CkptTensor]],
                  name: str, shape: Sequence[int], dtype: torch.dtype,
                  declared: int) -> None:
    """Per-tensor shape/dtype/byte equality (only with real shard headers)."""
    if per_tensor is None:
        return
    actual = per_tensor.get(name)
    if actual is None:
        return   # reported as `dangling`
    if tuple(int(s) for s in shape) != actual.shape or dtype != actual.dtype \
            or declared != actual.nbytes:
        report.add(
            "tensor_mismatch", name,
            f"declared {list(shape)} {dtype} = {declared:,} B, checkpoint has "
            f"{list(actual.shape)} {actual.dtype} = {actual.nbytes:,} B",
        )
