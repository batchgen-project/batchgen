"""GLM-5 PSM: the two copies the decode configure used to pay for twice.

Each prefill -> decode switch moved this rank's routed-expert shard to the GPU
and then copied the same bytes again device-to-device, because
``Glm5MoE._init_fp8_blockwise_weights`` allocated the 3D stacks the grouped GEMM
reads instead of adopting the ones the loader had just filled. Separately, every
per-expert FP8 dequant scale was uploaded with its own ``.to(device)`` call —
num_layers x num_experts x 3 of them, for a very small total payload, re-paid on
every switch in both directions although the scales never change.

These tests pin the replacements:

* ``_bind_local_routed_expert_stacks`` makes every expert alias a row of its
  layer's stack and publishes the stacks as ``_w3d_prealloc``, so the second
  copy is skipped;
* ``_upload_skeleton_scales`` packs the host-resident scales per dtype into one
  buffer, copies once, and the views it hands out are bitwise equal to the
  sources;
* the default decode path selects the preallocated stacks, the persistent path
  keeps its fixed-address arena, and a partially resident rank keeps the old
  per-expert loader.

The PSM module cannot be imported on a CPU-only machine (its model module needs
Triton), so methods are exec'd in isolation — the idiom in
``test_glm5_persistent_phase.py``.
"""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
PSM = ROOT / "batchgen" / "models" / "glm" / "glm5" / "Parallel_Strategy_Manager.py"
MODEL = ROOT / "batchgen" / "models" / "glm" / "glm5" / "model.py"
PSM_SRC = PSM.read_text()


def _psm_class():
    tree = ast.parse(PSM_SRC)
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "GLM5ParallelStrategyManager"
    )


def _isolated(names, **globals_override):
    """Build a bare class carrying the named PSM methods."""
    psm = _psm_class()
    wanted = []
    for name in names:
        wanted.append(
            copy.deepcopy(
                next(
                    node
                    for node in psm.body
                    if isinstance(node, ast.FunctionDef) and node.name == name
                )
            )
        )
    module = ast.Module(
        body=[
            ast.ClassDef(
                name="Psm",
                bases=[],
                keywords=[],
                body=wanted,
                decorator_list=[],
            )
        ],
        type_ignores=[],
    )
    namespace = {}
    exec(compile(ast.fix_missing_locations(module), str(PSM), "exec"), namespace)
    cls = namespace["Psm"]
    for name in names:
        member = getattr(cls, name)
        # A property's code object lives on its getter.
        target = member.fget if isinstance(member, property) else member
        target.__globals__.update(globals_override)
    return cls


class _TorchProxy:
    """Real torch, with the CUDA stream call stubbed out."""

    cuda = SimpleNamespace(
        current_stream=lambda: SimpleNamespace(synchronize=lambda: None),
        is_available=lambda: False,
    )

    def __getattr__(self, name):
        return getattr(torch, name)


_QUIET_LOGGING = SimpleNamespace(
    info=lambda *a, **k: None, debug=lambda *a, **k: None
)


# ---------------------------------------------------------------------------
# Scale packing
# ---------------------------------------------------------------------------


def _scale_psm(skeleton, rank=0):
    cls = _isolated(
        ["_upload_skeleton_scales", "_device_scale", "_pack_scales"],
        torch=_TorchProxy(),
        logging=_QUIET_LOGGING,
    )
    psm = cls()
    psm.skeleton_state_dict = skeleton
    psm.rank = rank
    psm._device_scales = None
    psm.engine_config = SimpleNamespace(
        Basic_Config=SimpleNamespace(device_torch="cpu")
    )
    return psm


def test_packed_scales_unpack_to_the_per_expert_values_bitwise():
    torch.manual_seed(0)
    skeleton = {}
    expected = {}
    for layer in range(3):
        for expert in range(4):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                key = (
                    f"model.layers.{layer}.mlp.experts.{expert}."
                    f"{proj}.weight_scale_inv"
                )
                tensor = torch.randn(2, 3, dtype=torch.float32)
                skeleton[key] = tensor
                expected[key] = tensor.clone()
    # A non-scale weight must be ignored entirely.
    skeleton["model.layers.0.mlp.experts.0.gate_proj.weight"] = torch.randn(2, 3)

    psm = _scale_psm(skeleton)
    scales = psm._upload_skeleton_scales()

    assert set(scales) == set(expected)
    for key, want in expected.items():
        got = scales[key]
        assert got.shape == want.shape
        assert got.dtype == want.dtype
        assert torch.equal(got, want), key


def _packer():
    cls = _isolated(["_pack_scales"], torch=_TorchProxy(), logging=_QUIET_LOGGING)
    return cls()._pack_scales


def test_packed_buffer_unpacks_to_each_tensor_bitwise():
    """The layout contract: one buffer, one reshaped slice per key.

    This is the half of the upload that does not need a device, so it is
    checked directly: a wrong offset or a wrong shape would silently hand one
    expert another expert's scale.
    """
    torch.manual_seed(7)
    entries = []
    for layer in range(2):
        for expert in range(3):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                key = (
                    f"model.layers.{layer}.mlp.experts.{expert}."
                    f"{proj}.weight_scale_inv"
                )
                entries.append((key, torch.randn(2, 3, dtype=torch.float32)))
    sources = dict(entries)

    staging, layout = _packer()(entries, torch.float32)

    assert staging.numel() == sum(t.numel() for t in sources.values())
    assert staging.dtype is torch.float32
    assert [key for key, _, _ in layout] == [key for key, _ in entries]
    seen = []
    for key, offset, shape in layout:
        count = 1
        for dim in shape:
            count *= dim
        view = staging[offset:offset + count].view(shape)
        assert torch.equal(view, sources[key]), key
        seen.append((offset, offset + count))
    # Contiguous, non-overlapping, covering the whole buffer.
    assert seen == sorted(seen)
    assert seen[0][0] == 0
    assert seen[-1][1] == staging.numel()
    for (_, prev_end), (next_start, _) in zip(seen, seen[1:]):
        assert prev_end == next_start


def test_packing_preserves_ragged_shapes():
    entries = [
        ("a.weight_scale_inv", torch.arange(6, dtype=torch.float32).reshape(2, 3)),
        ("b.weight_scale_inv", torch.arange(4, dtype=torch.float32).reshape(4)),
        ("c.weight_scale_inv", torch.arange(2, dtype=torch.float32).reshape(1, 2)),
    ]
    sources = dict(entries)

    staging, layout = _packer()(entries, torch.float32)

    for key, offset, shape in layout:
        count = 1
        for dim in shape:
            count *= dim
        assert tuple(sources[key].shape) == shape
        assert torch.equal(
            staging[offset:offset + count].view(shape), sources[key]
        ), key


def test_upload_groups_by_dtype():
    skeleton = {
        "a.weight_scale_inv": torch.tensor([1.5, 2.5], dtype=torch.float32),
        "b.weight_scale_inv": torch.tensor([3.0, 4.0], dtype=torch.float64),
        "c.weight_scale_inv": torch.tensor([5.5], dtype=torch.float32),
        "d.weight": torch.tensor([9.0]),
    }
    psm = _scale_psm(skeleton)
    packed = []
    real = psm._pack_scales

    def recording(entries, dtype):
        packed.append((dtype, [key for key, _ in entries]))
        return real(entries, dtype)

    psm._pack_scales = recording
    # Force the upload path: pretend nothing is resident yet.
    psm.engine_config = SimpleNamespace(
        Basic_Config=SimpleNamespace(device_torch="meta")
    )

    psm._upload_skeleton_scales()

    assert len(packed) == 2, "one buffer per dtype"
    by_dtype = {dtype: keys for dtype, keys in packed}
    assert by_dtype[torch.float32] == ["a.weight_scale_inv", "c.weight_scale_inv"]
    assert by_dtype[torch.float64] == ["b.weight_scale_inv"]
    # The non-scale entry is never packed.
    assert all("d.weight" not in keys for _, keys in packed)


def test_scales_already_on_the_target_device_are_handed_back_unchanged():
    """share_skeleton_on_device() pre-moves the skeleton; do not repack it.

    The persistent phase path uploads the skeleton once and both instances
    alias that copy. Repacking would hand out different addresses for the same
    values, so already-resident tensors are returned as themselves.
    """
    resident = torch.randn(3, dtype=torch.float32)
    skeleton = {"x.weight_scale_inv": resident}
    psm = _scale_psm(skeleton)

    scales = psm._upload_skeleton_scales()

    assert scales["x.weight_scale_inv"] is resident


def test_device_scale_uploads_once_and_caches():
    skeleton = {
        "a.weight_scale_inv": torch.randn(2),
        "b.weight_scale_inv": torch.randn(2),
    }
    psm = _scale_psm(skeleton)
    uploads = []
    real = psm._upload_skeleton_scales

    def counting():
        uploads.append(1)
        return real()

    psm._upload_skeleton_scales = counting

    first = psm._device_scale("a.weight_scale_inv")
    second = psm._device_scale("b.weight_scale_inv")
    again = psm._device_scale("a.weight_scale_inv")

    assert len(uploads) == 1
    assert again is first
    assert second is not None
    # A key the skeleton does not carry is reported as absent, not invented.
    assert psm._device_scale("missing.weight_scale_inv") is None


# ---------------------------------------------------------------------------
# Expert stacks
# ---------------------------------------------------------------------------


_NAMES = ("gate_proj.weight", "up_proj.weight", "down_proj.weight")


def _stack_psm(*, experts_per_rank=4, world_size=2, global_rank=1, layers=3,
               first_dense=1, resident=None, shapes=None):
    cls = _isolated(
        [
            "_bind_local_routed_expert_stacks",
            "_local_routed_experts_fully_resident",
            "_expert_stack_layer_bytes",
        ],
        torch=_TorchProxy(),
        logging=_QUIET_LOGGING,
    )
    psm = cls()
    psm.NUM_TOTAL_EXPERTS = experts_per_rank * world_size
    psm.world_size = world_size
    psm.global_rank = global_rank
    psm.FIRST_K_DENSE = first_dense
    psm.num_local_expert_per_layer = (
        experts_per_rank if resident is None else resident
    )
    psm.model_config = SimpleNamespace(num_hidden_layers=layers)

    shapes = shapes or {name: (2, 3) for name in _NAMES}
    store = {}
    start = global_rank * experts_per_rank
    for layer in range(first_dense, layers):
        for expert in range(start, start + experts_per_rank):
            store[f"routed_expert_{layer}_{expert}"] = {
                name: torch.full(shapes[name], float(layer * 100 + expert))
                for name in _NAMES
            }
    psm.core_engine = SimpleNamespace(get_tensor=lambda key: store[key])
    psm.local_routed_experts = [
        f"routed_expert_{first_dense}_{start + i}" for i in range(experts_per_rank)
    ]

    experts = {}
    layer_objs = []
    for layer in range(layers):
        mlp = SimpleNamespace(
            experts={
                idx: SimpleNamespace(fp8_gate=None, fp8_up=None, fp8_down=None)
                for idx in range(psm.NUM_TOTAL_EXPERTS)
            }
        )
        experts[layer] = mlp
        layer_objs.append(SimpleNamespace(mlp=mlp))
    psm.model = SimpleNamespace(model=SimpleNamespace(layers=layer_objs))
    return psm, experts, store, start


def _carve(shape, dtype):
    return torch.empty(shape, dtype=dtype)


def test_every_expert_aliases_a_row_of_its_layer_stack():
    psm, layers, store, start = _stack_psm()

    pairs = psm._bind_local_routed_expert_stacks(_carve)

    moe_layers = [1, 2]
    assert len(pairs) == len(moe_layers) * 4 * len(_NAMES)
    for layer in moe_layers:
        mlp = layers[layer]
        stacks = mlp._w3d_prealloc
        assert len(stacks) == len(_NAMES)
        for stack in stacks:
            assert stack.shape[0] == 4
        for i in range(4):
            expert = mlp.experts[start + i]
            for attr, stack in zip(
                ("fp8_gate", "fp8_up", "fp8_down"), stacks
            ):
                row = getattr(expert, attr)
                assert row.data_ptr() == stack[i].data_ptr(), (layer, i, attr)
    # Dense layers below FIRST_K_DENSE are untouched.
    assert not hasattr(layers[0], "_w3d_prealloc")


def test_bound_pairs_copy_the_right_host_tensor_into_the_right_row():
    psm, layers, store, start = _stack_psm()

    pairs = psm._bind_local_routed_expert_stacks(_carve)
    for dst, src in pairs:
        dst.copy_(src)

    for layer in (1, 2):
        stacks = layers[layer]._w3d_prealloc
        for i in range(4):
            host = store[f"routed_expert_{layer}_{start + i}"]
            for name, stack in zip(_NAMES, stacks):
                assert torch.equal(stack[i], host[name]), (layer, i, name)


def test_binding_refuses_a_partially_resident_rank():
    psm, _, _, _ = _stack_psm(resident=2)

    assert psm._local_routed_experts_fully_resident() is False
    with pytest.raises(RuntimeError, match="every local routed expert resident"):
        psm._bind_local_routed_expert_stacks(_carve)


def test_binding_fails_loud_on_a_host_shape_mismatch():
    psm, _, store, start = _stack_psm()
    # One expert's host tensor has the wrong shape: the stack row it would be
    # copied into cannot hold it, and a silent reshape would scramble weights.
    store[f"routed_expert_2_{start + 1}"]["up_proj.weight"] = torch.zeros(5, 7)

    with pytest.raises(RuntimeError, match="does not match stack row"):
        psm._bind_local_routed_expert_stacks(_carve)


def test_binding_fails_loud_on_a_host_dtype_mismatch():
    psm, _, store, start = _stack_psm()
    store[f"routed_expert_1_{start + 2}"]["down_proj.weight"] = torch.zeros(
        2, 3, dtype=torch.float64
    )

    with pytest.raises(RuntimeError, match="does not match stack row"):
        psm._bind_local_routed_expert_stacks(_carve)


# ---------------------------------------------------------------------------
# Path selection and the removed device-to-device copy
# ---------------------------------------------------------------------------


def _method(name):
    return next(
        node
        for node in _psm_class().body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_decode_configure_selects_prealloc_by_default():
    src = ast.unparse(_method("configure_decoding"))
    assert "self._load_local_routed_experts_into_arena()" in src
    assert "self._load_local_routed_experts_prealloc()" in src
    assert "self._local_routed_experts_fully_resident()" in src
    # The legacy per-expert loader survives for partially resident ranks.
    assert "self._load_local_routed_experts()" in src
    assert src.index("_load_local_routed_experts_prealloc") < src.index(
        "self._load_local_routed_experts()"
    )


def test_both_stack_paths_share_one_binder():
    arena = ast.unparse(_method("_load_local_routed_experts_into_arena"))
    prealloc = ast.unparse(_method("_load_local_routed_experts_prealloc"))
    for src in (arena, prealloc):
        assert "_bind_local_routed_expert_stacks" in src
    # The persistent path still owns a fixed-address arena and still keeps the
    # pairs, because it refills them on every activation.
    assert "VmmArena" in arena
    assert "self._expert_copy_pairs =" in arena
    # The default path must not keep a manager-level reference, or this rank's
    # whole routed-expert shard would stay pinned across the prefill phase.
    assert "self._expert_copy_pairs" not in prealloc
    assert "VmmArena" not in prealloc


def test_blockwise_init_adopts_preallocated_stacks_without_recopying():
    model_src = MODEL.read_text()
    tree = ast.parse(model_src)
    init = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_init_fp8_blockwise_weights"
    )
    branch = next(
        node
        for node in ast.walk(init)
        if isinstance(node, ast.If) and "prealloc is not None" in ast.unparse(node.test)
    )
    adopt = ast.unparse(branch.body)
    # The adopting branch must not allocate a stack or copy experts into one.
    assert "torch.empty" not in adopt
    assert ".copy_(" not in adopt
    assert "self.fp8_gate_w3d, self.fp8_up_w3d, self.fp8_down_w3d) = prealloc" in adopt
    # The allocate-and-copy code remains, for the partially resident path.
    assert "torch.empty" in ast.unparse(branch.orelse)


# ---------------------------------------------------------------------------
# Reuse capability bit
# ---------------------------------------------------------------------------


def _reuse_psm(*, flag=False, reuse=False, resident=4, experts_per_rank=4,
               world_size=2):
    cls = _isolated(
        ["_instances_are_reused", "_local_routed_experts_fully_resident"],
        torch=_TorchProxy(),
        logging=_QUIET_LOGGING,
    )
    psm = cls()
    psm.persistent_phase_instances = flag
    psm.reuse_phase_instances = reuse
    psm.NUM_TOTAL_EXPERTS = experts_per_rank * world_size
    psm.world_size = world_size
    if resident is not None:
        psm.num_local_expert_per_layer = resident
    return psm


def test_reuse_is_off_by_default():
    assert _reuse_psm()._instances_are_reused is False


def test_cli_flag_still_turns_reuse_on_regardless_of_residency():
    """--persistent-phase-instances keeps its own behaviour exactly.

    It asks for reuse unconditionally; `configure_decoding` then raises from the
    arena loader if the rank is not fully resident, which is what it did before.
    """
    assert _reuse_psm(flag=True, resident=2)._instances_are_reused is True
    assert _reuse_psm(flag=True, resident=4)._instances_are_reused is True


def test_default_lifecycle_reuses_only_a_fully_resident_rank():
    assert _reuse_psm(reuse=True, resident=4)._instances_are_reused is True
    # Streaming rank: no reusable instance, so the worker rebuilds as before
    # instead of the arena loader raising at it.
    assert _reuse_psm(reuse=True, resident=2)._instances_are_reused is False


def test_residency_is_unknown_before_the_first_decode_configure():
    """`num_local_expert_per_layer` is set inside configure_decoding.

    Reported as not resident rather than guessed, so a prefill-first process
    cannot cache an instance it has not proven it can reuse.
    """
    psm = _reuse_psm(reuse=True, resident=None)
    assert psm._local_routed_experts_fully_resident() is False
    assert psm._instances_are_reused is False


def test_expert_loader_selection_keeps_the_flag_path_separate():
    src = ast.unparse(_method("configure_decoding"))
    # The flag reaches the arena loader with no residency condition in front of
    # it, so its existing failure mode is untouched.
    assert "if self.persistent_phase_instances:" in src
    assert "self.reuse_phase_instances and self._local_routed_experts_fully_resident()" in src
    # Three distinct destinations: arena, preallocated stacks, legacy loader.
    assert src.count("_load_local_routed_experts_into_arena()") == 2
    assert "self._load_local_routed_experts_prealloc()" in src
    assert "self._load_local_routed_experts()" in src


def test_instance_caching_follows_the_same_predicate():
    for name in ("configure_prefill", "configure_decoding"):
        src = ast.unparse(_method(name))
        assert "if self._instances_are_reused:" in src, name
