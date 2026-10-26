"""CPU contract for folding the GLM-5 decode Q NoPE/RoPE split away.

The q_b_proj output is one packed ``[batch, num_heads * q_head_dim]`` bf16
tensor. Both DSA attention segments used to turn it into two separate static
buffers per layer:

    q_view = q_flat.view(B, 1, H, D).transpose(1, 2)
    q_nope.copy_(q_view[..., :nope].squeeze(2).contiguous())
    q_rope_4d.copy_(q_view[..., nope:].contiguous())

i.e. two strided gathers plus the device copies into the static buffers, on
every one of the 78 decoder layers, inside the captured graph.

Neither consumer needs a gather:

  * the absorb GEMV is already handed its activation's batch and head strides
    and only requires a contiguous K axis, so it reads the NoPE half in place;
  * the fused rmsnorm+RoPE kernel takes the RoPE half as ``q_pe_src`` and
    writes the rotated pair into the SAME contiguous ``q_rope_4d`` that FA3
    consumes, so the gather disappears into a launch that already ran.

What this file pins is that the fold stays folded. A numeric test cannot see
the difference -- the values are identical by construction, which is the whole
point -- so the quantity being optimized (launches inside the replay) is only
observable structurally. The two attention-segment modules do not import on a
CPU box (FA3 / Triton / deep_gemm), so they are asserted on with ``ast``, the
same approach as ``test_glm5_fold_elementwise_into_neighbours.py``.

It also pins what must NOT move: the RoPE destination is still
``buffers.q_rope_4d`` (FA3's query), and the row gate folded in by the
preceding slice is still a SELECT driven by the boolean ``valid_mask``.
"""

import ast
from pathlib import Path

import pytest


_GLM5 = Path(__file__).parents[1] / "batchgen" / "models" / "glm" / "glm5"
SEGMENTS = (
    ("cuda_graph_segments.py", "Glm5FullDsaAttnSegment"),
    ("reuse_topk_segment.py", "Glm5ReuseTopkAttnSegment"),
)


def _tree(name):
    return ast.parse((_GLM5 / name).read_text())


def _class(tree, class_name):
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise AssertionError(f"no class `{class_name}`")


def _method(tree, class_name, method_name):
    for stmt in _class(tree, class_name).body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == method_name:
            return stmt
    raise AssertionError(f"no `{class_name}.{method_name}`")


def _calls(node, func_source):
    return [
        sub
        for sub in ast.walk(node)
        if isinstance(sub, ast.Call) and ast.unparse(sub.func) == func_source
    ]


def _attr_calls(node, attr):
    return [
        (ast.unparse(sub.func.value), sub)
        for sub in ast.walk(node)
        if isinstance(sub, ast.Call)
        and isinstance(sub.func, ast.Attribute)
        and sub.func.attr == attr
    ]


def _kwargs(call):
    return {kw.arg: ast.unparse(kw.value) for kw in call.keywords if kw.arg}


def _assigned(node, name):
    for sub in ast.walk(node):
        if isinstance(sub, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in sub.targets
        ):
            return ast.unparse(sub.value)
    raise AssertionError(f"no assignment to `{name}`")


def _dataclass_fields(tree, name):
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return [
                stmt.target.id
                for stmt in node.body
                if isinstance(stmt, ast.AnnAssign)
                and isinstance(stmt.target, ast.Name)
            ]
    raise AssertionError(f"no dataclass `{name}`")


def _name_set(tree, name):
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
            and isinstance(node.value, ast.Call)
        ):
            (arg,) = node.value.args
            return {
                elt.value
                for elt in arg.elts
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
            }
    raise AssertionError(f"no set binding `{name}`")


# --------------------------------------------------------------------------- #
# The gathers are gone
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("segment,class_name", SEGMENTS)
def test_neither_q_half_is_materialized_by_a_copy(segment, class_name):
    fwd = _method(_tree(segment), class_name, "forward")
    targets = [target for target, _ in _attr_calls(fwd, "copy_")]
    offenders = [t for t in targets if "q_nope" in t or "q_rope" in t]
    assert not offenders, (
        f"{segment}: both q halves are read out of the packed q_b_proj output "
        f"with strides; a copy_ into a static q buffer is exactly the launch "
        f"this change removes, got {offenders}"
    )


@pytest.mark.parametrize("segment,class_name", SEGMENTS)
def test_no_q_slice_is_made_contiguous(segment, class_name):
    """``.contiguous()`` on a strided q slice IS the gather kernel."""
    fwd = _method(_tree(segment), class_name, "forward")
    offenders = [
        target
        for target, _ in _attr_calls(fwd, "contiguous")
        if "q_flat" in target or "q_heads" in target or "q_view" in target
    ]
    assert not offenders, (
        f"{segment}: a strided .contiguous() on the q projection output is the "
        f"gather launch, not a free reshape, got {offenders}"
    )
    # The transposed intermediate the old form needed must be gone with it.
    assert "q_view" not in ast.unparse(fwd), f"{segment}: q_view is back"


@pytest.mark.parametrize("segment,class_name", SEGMENTS)
def test_both_halves_are_strided_views_of_the_single_projection_output(
    segment, class_name
):
    fwd = _method(_tree(segment), class_name, "forward")
    assert _assigned(fwd, "q_heads") == (
        "buffers.q_flat.view(batch_size, attn.num_heads, attn.q_head_dim)"
    ), (
        f"{segment}: the head view must come straight off the contiguous "
        f"q_b_proj output, so each head's slice is one (row, head) stride pair"
    )
    assert _assigned(fwd, "q_nope_strided") == "q_heads[..., :attn.qk_nope_head_dim]"
    assert _assigned(fwd, "q_rope_strided") == (
        "q_heads[..., attn.qk_nope_head_dim:].unsqueeze(2)"
    )


# --------------------------------------------------------------------------- #
# The consumers read the views
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("segment,class_name", SEGMENTS)
def test_rope_kernel_reads_the_strided_half_and_writes_the_static_buffer(
    segment, class_name
):
    fwd = _method(_tree(segment), class_name, "forward")
    (call,) = _calls(fwd, "_fused_rmsnorm_rope")
    args = [ast.unparse(a) for a in call.args]
    kw = _kwargs(call)
    # Destination unchanged: this is FA3's query tensor and must stay the
    # contiguous static buffer.
    assert args[1] == "buffers.q_rope_4d", (
        f"{segment}: the rotated q must still land in the contiguous static "
        f"buffer FA3 reads, got {args[1]!r}"
    )
    assert kw.get("q_pe_src") == "q_rope_strided", (
        f"{segment}: the rope half must be READ from the packed projection "
        f"output; dropping q_pe_src silently reinstates the gather, got "
        f"{kw.get('q_pe_src')!r}"
    )
    # The preceding slice's fold must survive this one: still a SELECT on the
    # boolean row mask, still writing primary_k_tensor directly.
    assert kw.get("row_valid_mask") == "buffers.valid_mask"
    assert "primary_k_tensor" in kw.get("out", "")


@pytest.mark.parametrize("segment,class_name", SEGMENTS)
def test_absorb_gemv_reads_the_strided_nope_half(segment, class_name):
    fwd = _method(_tree(segment), class_name, "forward")
    # Slice 2c-2 renamed the entry point: production now calls the
    # container-dispatching `absorb_*`. The invariant pinned here is
    # unchanged; only the symbol moved.
    (call,) = _calls(fwd, "absorb_q_out")
    first = ast.unparse(call.args[0])
    assert first == "q_nope_strided", (
        f"{segment}: the absorb GEMV takes the NoPE half in place -- it is "
        f"handed the batch and head strides and needs only K contiguous -- so "
        f"a separate buffer here means the gather came back, got {first!r}"
    )


@pytest.mark.parametrize("segment,class_name", SEGMENTS)
def test_the_fold_adds_no_launch_of_its_own(segment, class_name):
    """One rope launch and one absorb launch per layer, as before."""
    fwd = _method(_tree(segment), class_name, "forward")
    assert len(_calls(fwd, "_fused_rmsnorm_rope")) == 1
    assert len(_calls(fwd, "absorb_q_out")) == 1


# --------------------------------------------------------------------------- #
# The dead buffer is gone, not merely unused
# --------------------------------------------------------------------------- #


def test_the_nope_buffer_field_is_removed():
    fields = _dataclass_fields(_tree("cuda_graph_segments.py"), "_Glm5FullDsaSegmentBuffers")
    assert "q_nope" not in fields, (
        "a static buffer nothing writes is how a removed copy creeps back in; "
        "the NoPE half has no buffer any more"
    )
    # The RoPE destination stays: it is the kernel's out and FA3's query.
    assert "q_rope_4d" in fields
    assert "q_flat" in fields


def test_the_nope_buffer_is_absent_from_every_field_classification_set():
    segments = _tree("cuda_graph_segments.py")
    reuse = _tree("reuse_topk_segment.py")
    sets = {
        "_GLM5_DSA_VIEW_FIELDS": _name_set(segments, "_GLM5_DSA_VIEW_FIELDS"),
        "_GLM5_DSA_REBUILT_FIELDS": _name_set(segments, "_GLM5_DSA_REBUILT_FIELDS"),
        "_GLM5_REUSE_VIEW_FIELDS": _name_set(reuse, "_GLM5_REUSE_VIEW_FIELDS"),
        "_GLM5_REUSE_PLACEHOLDER_FIELDS": _name_set(
            reuse, "_GLM5_REUSE_PLACEHOLDER_FIELDS"
        ),
        "_GLM5_REUSE_SPECIAL_FIELDS": _name_set(reuse, "_GLM5_REUSE_SPECIAL_FIELDS"),
    }
    for name, members in sets.items():
        assert "q_nope" not in members, f"{name} still classifies q_nope"
    # Both modules assert the xor of fields against these sets at IMPORT time;
    # the existing coverage tests check the same thing from source, so a stale
    # entry here would be a hard import failure on the node.
    assert "q_rope_4d" in sets["_GLM5_DSA_VIEW_FIELDS"]
    assert "q_rope_4d" in sets["_GLM5_REUSE_VIEW_FIELDS"]


@pytest.mark.parametrize("segment,class_name", SEGMENTS)
def test_the_nope_buffer_is_no_longer_allocated(segment, class_name):
    alloc = ast.unparse(_method(_tree(segment), class_name, "setup_static_buffers"))
    assert "q_nope" not in alloc, (
        f"{segment}: setup_static_buffers still allocates the NoPE half"
    )
    assert "q_rope_4d=torch.empty(" in alloc.replace(" ", "").replace("\n", ""), (
        f"{segment}: the RoPE destination must still be allocated"
    )


# --------------------------------------------------------------------------- #
# The test doubles are part of the contract
# --------------------------------------------------------------------------- #


_STUBBED_SEGMENT_TESTS = Path(__file__).with_name("test_glm5_full_dsa_graph_segment.py")


def _stub_bindings(path):
    """Map each stubbed kernel name to the stub's own parameter signature.

    Reads the ``monkeypatch.setattr(segments, "<name>", <stub>)`` pairs, so a
    stub that is renamed or re-pointed is followed automatically.
    """
    tree = ast.parse(path.read_text())
    defs = {
        node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    bindings = {}
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and ast.unparse(node.func).endswith("setattr")
            and len(node.args) == 3
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[2], ast.Name)
        ):
            continue
        stub = defs.get(node.args[2].id)
        if stub is not None:
            bindings[node.args[1].value] = stub
    return bindings


def _bind(stub: ast.FunctionDef, n_positional: int, keywords: list[str]) -> None:
    """Raise AssertionError if the call cannot bind to the stub's signature."""
    args = stub.args
    positional = [a.arg for a in args.posonlyargs + args.args]
    kwonly = [a.arg for a in args.kwonlyargs]
    n_defaults = len(args.defaults)
    required = positional[: len(positional) - n_defaults]
    has_var_kw = args.kwarg is not None
    has_var_pos = args.vararg is not None

    if n_positional > len(positional) and not has_var_pos:
        raise AssertionError(
            f"{stub.name} takes {len(positional)} positional parameters, call "
            f"passes {n_positional}"
        )
    filled = set(positional[:n_positional])
    for kw in keywords:
        if kw in filled:
            raise AssertionError(f"{stub.name}: `{kw}` given twice")
        if kw not in positional and kw not in kwonly and not has_var_kw:
            raise AssertionError(
                f"{stub.name} does not accept `{kw}=`; the stub stands in for "
                f"the real kernel, so it must honour the same contract"
            )
        filled.add(kw)
    missing = [p for p in required if p not in filled]
    if missing:
        raise AssertionError(f"{stub.name}: missing {missing}")
    for p in kwonly:
        idx = kwonly.index(p)
        if args.kw_defaults[idx] is None and p not in filled:
            raise AssertionError(f"{stub.name}: missing keyword-only `{p}`")


@pytest.mark.parametrize("segment,class_name", SEGMENTS)
def test_every_stubbed_kernel_accepts_the_production_call(segment, class_name):
    """A stub that rejects a new argument raises before comparing a number.

    The stubs' test names talk about replay matching eager, so a binding
    failure there reads as a numeric regression. Bind them from source instead,
    which runs on a CPU box where the CUDA tests themselves only skip.
    """
    stubs = _stub_bindings(_STUBBED_SEGMENT_TESTS)
    assert "_fused_rmsnorm_rope" in stubs and "absorb_q_out" in stubs, (
        f"the two kernels this change re-contracts must both be stubbed, got "
        f"{sorted(stubs)}"
    )
    fwd = _method(_tree(segment), class_name, "forward")
    checked = 0
    for name, stub in stubs.items():
        for call in _calls(fwd, name):
            _bind(stub, len(call.args), [kw.arg for kw in call.keywords if kw.arg])
            checked += 1
    assert checked, f"{segment}: no stubbed call site found in forward"
