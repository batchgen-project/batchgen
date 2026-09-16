import importlib.util
from pathlib import Path

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="ordered BF16 reduce tests require CUDA",
)


_WRAPPER = None


def _kernel():
    # Load the focused wrapper by path so this kernel test does not import the
    # full batchgen package or build the unrelated core_engine extension.
    global _WRAPPER
    if _WRAPPER is None:
        path = (
            Path(__file__).resolve().parents[1]
            / "batchgen"
            / "moe"
            / "dispatch_scatter_3d.py"
        )
        spec = importlib.util.spec_from_file_location(
            "glm52_ordered_reduce_wrapper", path
        )
        _WRAPPER = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(_WRAPPER)

    return _WRAPPER.reduce_weighted_scatter_bf16_ordered


def _reference(expert_output, topk_pos, topk_indices, topk_weights, sort=True):
    """Scalar CPU replay: acc = bf16(acc + bf16(x * bf16(w))) per valid slot."""
    x = expert_output.cpu()
    pos = topk_pos.cpu().tolist()
    eid = topk_indices.cpu().tolist()
    w16 = topk_weights.cpu().to(torch.bfloat16).float()
    n, k = topk_pos.shape
    out = torch.zeros(n, x.shape[1], dtype=torch.bfloat16)
    for t in range(n):
        order = sorted(range(k), key=lambda s: eid[t][s]) if sort else range(k)
        acc = torch.zeros(x.shape[1], dtype=torch.bfloat16)
        for s in order:
            if pos[t][s] < 0:
                continue
            prod = (x[pos[t][s]].float() * w16[t, s]).to(torch.bfloat16)
            acc = (acc.float() + prod.float()).to(torch.bfloat16)
        out[t] = acc
    return out


def _assert_bitwise(actual, expected):
    a = actual.cpu().view(torch.int16)
    e = expected.cpu().view(torch.int16)
    mismatches = int((a != e).sum().item())
    assert mismatches == 0, f"{mismatches}/{a.numel()} BF16 values differ bitwise"


def _make_inputs(n, h, k, rows, num_experts, invalid_frac, seed, duplicate_ids=False):
    gen = torch.Generator().manual_seed(seed)
    # Wide dynamic range so rounding order is observable.
    scale = torch.pow(10.0, torch.empty(rows, h).uniform_(-3, 3, generator=gen))
    expert_output = (torch.randn(rows, h, generator=gen) * scale).to(torch.bfloat16)
    if duplicate_ids:
        topk_indices = torch.randint(0, 3, (n, k), generator=gen, dtype=torch.int32)
    else:
        topk_indices = torch.stack(
            [torch.randperm(num_experts, generator=gen)[:k] for _ in range(n)]
        ).to(torch.int32)
    topk_pos = torch.randint(0, rows, (n, k), generator=gen, dtype=torch.int32)
    invalid = torch.rand(n, k, generator=gen) < invalid_frac
    topk_pos[invalid] = -1
    topk_weights = torch.rand(n, k, generator=gen, dtype=torch.float32)
    topk_weights = topk_weights / topk_weights.sum(dim=1, keepdim=True) * 2.5
    dev = torch.device("cuda")
    return (
        expert_output.to(dev).contiguous(),
        topk_pos.to(dev).contiguous(),
        topk_indices.to(dev).contiguous(),
        topk_weights.to(dev).contiguous(),
    )


def test_ordered_reduce_production_shape_bitexact():
    n, h, k, rows = 16, 6144, 8, 64
    x, pos, eid, w = _make_inputs(n, h, k, rows, 256, invalid_frac=0.3, seed=20260916)
    # Production's ragged dispatcher returns this metadata as flat [N*K].
    actual = _kernel()(x, pos.reshape(-1), eid, w, n, h, k)
    torch.cuda.synchronize()
    expected = _reference(x, pos, eid, w)
    _assert_bitwise(actual, expected)

    # The test must be able to see ordering: slot-order replay differs.
    slot_order = _reference(x, pos, eid, w, sort=False)
    assert not torch.equal(expected.view(torch.int16), slot_order.view(torch.int16))


@pytest.mark.parametrize("k", [2, 4, 8])
@pytest.mark.parametrize("h", [8, 64, 2048, 2056])
def test_ordered_reduce_k_and_h_sweep_bitexact(k, h):
    n, rows = 7, 13
    x, pos, eid, w = _make_inputs(n, h, k, rows, 32, invalid_frac=0.25, seed=1000 + 17 * k + h)
    actual = _kernel()(x, pos, eid, w, n, h, k)
    torch.cuda.synchronize()
    _assert_bitwise(actual, _reference(x, pos, eid, w))


def test_ordered_reduce_equal_expert_ids_keep_slot_order():
    n, h, k, rows = 9, 256, 8, 11
    x, pos, eid, w = _make_inputs(
        n, h, k, rows, 3, invalid_frac=0.1, seed=4242, duplicate_ids=True
    )
    actual = _kernel()(x, pos, eid, w, n, h, k)
    torch.cuda.synchronize()
    _assert_bitwise(actual, _reference(x, pos, eid, w))


def test_ordered_reduce_is_slot_permutation_invariant():
    n, h, k, rows = 12, 6144, 8, 40
    x, pos, eid, w = _make_inputs(n, h, k, rows, 256, invalid_frac=0.2, seed=77)
    perm = torch.randperm(k, generator=torch.Generator().manual_seed(5)).cuda()
    base = _kernel()(x, pos, eid, w, n, h, k)
    permuted = _kernel()(
        x,
        pos[:, perm].contiguous(),
        eid[:, perm].contiguous(),
        w[:, perm].contiguous(),
        n, h, k,
    )
    torch.cuda.synchronize()
    _assert_bitwise(permuted, base)


def test_ordered_reduce_all_invalid_writes_positive_zero():
    n, h, k, rows = 5, 6144, 8, 4
    x, pos, eid, w = _make_inputs(n, h, k, rows, 256, invalid_frac=0.0, seed=9)
    pos.fill_(-1)
    output = torch.full((n, h), -3.5, dtype=torch.bfloat16, device="cuda")
    result = _kernel()(x, pos, eid, w, n, h, k, output)
    torch.cuda.synchronize()
    assert result.data_ptr() == output.data_ptr()
    _assert_bitwise(output, torch.zeros(n, h, dtype=torch.bfloat16))


def test_ordered_reduce_preallocated_row_slice_leaves_neighbors_untouched():
    n, h, k, rows = 6, 6144, 8, 24
    x, pos, eid, w = _make_inputs(n, h, k, rows, 256, invalid_frac=0.3, seed=31337)
    # Expert rows are also taken from a slice of a larger buffer.
    big_x = torch.full((rows + 2, h), 7.0, dtype=torch.bfloat16, device="cuda")
    big_x[1:rows + 1] = x
    x_slice = big_x[1:rows + 1]

    sentinel = -1.25
    big_out = torch.full((n + 2, h), sentinel, dtype=torch.bfloat16, device="cuda")
    out_slice = big_out[1:n + 1]
    assert out_slice.is_contiguous() and out_slice.data_ptr() % 16 == 0
    assert x_slice.is_contiguous() and x_slice.data_ptr() % 16 == 0

    result = _kernel()(x_slice, pos, eid, w, n, h, k, out_slice)
    torch.cuda.synchronize()
    assert result.data_ptr() == out_slice.data_ptr()
    _assert_bitwise(out_slice, _reference(x, pos, eid, w))
    neighbor = torch.full((h,), sentinel, dtype=torch.bfloat16)
    _assert_bitwise(big_out[0], neighbor)
    _assert_bitwise(big_out[n + 1], neighbor)


def test_ordered_reduce_cuda_graph_replay_bitexact():
    n, h, k, rows = 8, 6144, 8, 32
    x, pos, eid, w = _make_inputs(n, h, k, rows, 256, invalid_frac=0.2, seed=8128)
    output = torch.empty(n, h, dtype=torch.bfloat16, device="cuda")
    kernel = _kernel()

    # Load and warm the extension before capture. The captured path receives a
    # preallocated output and performs no allocation or host synchronization.
    kernel(x, pos.reshape(-1), eid, w, n, h, k, output)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = kernel(x, pos.reshape(-1), eid, w, n, h, k, output)

    x.mul_(torch.tensor(0.75, dtype=torch.bfloat16, device="cuda"))
    graph.replay()
    torch.cuda.synchronize()
    assert result.data_ptr() == output.data_ptr()
    _assert_bitwise(output, _reference(x, pos, eid, w))


def _valid_args(n=3, h=64, k=8, rows=5):
    x, pos, eid, w = _make_inputs(n, h, k, rows, 16, invalid_frac=0.0, seed=1)
    out = torch.empty(n, h, dtype=torch.bfloat16, device="cuda")
    return dict(x=x, pos=pos, eid=eid, w=w, n=n, h=h, k=k, out=out)


def _call(a):
    return _kernel()(a["x"], a["pos"], a["eid"], a["w"], a["n"], a["h"], a["k"], a["out"])


def _misaligned_bf16(shape):
    flat = torch.empty(shape[0] * shape[1] + 1, dtype=torch.bfloat16, device="cuda")
    t = flat[1:].view(*shape)
    assert t.is_contiguous() and t.data_ptr() % 16 != 0
    return t


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda a: a.update(k=3, pos=a["pos"][:, :3].contiguous(),
                                        eid=a["eid"][:, :3].contiguous(),
                                        w=a["w"][:, :3].contiguous()), id="k3"),
        pytest.param(lambda a: a.update(x=a["x"][:, :60].contiguous(), h=60,
                                        out=a["out"][:, :60].contiguous()), id="h_not_mult8"),
        pytest.param(lambda a: a.update(x=a["x"].float()), id="expert_output_dtype"),
        pytest.param(lambda a: a.update(pos=a["pos"].long()), id="pos_dtype"),
        pytest.param(lambda a: a.update(eid=a["eid"].long()), id="indices_dtype"),
        pytest.param(lambda a: a.update(w=a["w"].to(torch.bfloat16)), id="weights_dtype"),
        pytest.param(lambda a: a.update(out=a["out"].float()), id="output_dtype"),
        pytest.param(lambda a: a.update(pos=a["pos"].cpu()), id="pos_cpu"),
        pytest.param(lambda a: a.update(x=a["x"].cpu()), id="expert_output_cpu"),
        pytest.param(lambda a: a.update(pos=a["pos"].t().contiguous().t()), id="pos_noncontig"),
        pytest.param(lambda a: a.update(out=a["out"].t().contiguous().t()), id="output_noncontig"),
        pytest.param(lambda a: a.update(out=torch.empty(a["n"] + 1, a["h"],
                                                        dtype=torch.bfloat16,
                                                        device="cuda")), id="output_shape"),
        pytest.param(lambda a: a.update(pos=a["pos"][:-1].contiguous()), id="pos_numel"),
        pytest.param(lambda a: a.update(w=a["w"][:2].contiguous()), id="weights_rows"),
        pytest.param(lambda a: a.update(x=a["x"][:, :32].contiguous()), id="expert_output_width"),
        pytest.param(lambda a: a.update(out=_misaligned_bf16((a["n"], a["h"]))), id="output_misaligned"),
        pytest.param(lambda a: a.update(x=_misaligned_bf16(tuple(a["x"].shape))),
                     id="expert_output_misaligned"),
    ],
)
def test_ordered_reduce_rejects_contract_violations(mutate):
    args = _valid_args()
    _call(args)  # the unmodified contract is accepted
    mutate(args)
    with pytest.raises(RuntimeError):
        _call(args)
