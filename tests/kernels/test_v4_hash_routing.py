# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                          #
# ---------------------------------------------------------------------------- #

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def _make_case(
    T: int,
    *,
    hidden_size: int = 64,
    n_experts: int = 64,
    topk: int = 6,
    vocab_size: int = 129280,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden_states = torch.randn(
        T, hidden_size, device="cuda", dtype=torch.bfloat16
    )
    gate_weight = torch.randn(
        n_experts, hidden_size, device="cuda", dtype=torch.bfloat16
    )
    tid2eid = torch.randint(
        0, n_experts, (vocab_size, topk), device="cuda", dtype=torch.int32
    )
    input_ids = torch.randint(
        0, vocab_size, (T,), device="cuda", dtype=torch.long
    )
    return input_ids, tid2eid, hidden_states, gate_weight


@pytest.mark.parametrize("T", [1, 32, 128, 1024])
def test_lookup_matches_index_select(T):
    torch.manual_seed(42)
    input_ids, tid2eid, _, _ = _make_case(T)

    assert torch.equal(
        tid2eid[input_ids], torch.index_select(tid2eid, 0, input_ids)
    )


def test_output_shape():
    from batchgen_kernels.moe.v4_hash_routing import hash_routing

    torch.manual_seed(42)
    input_ids, tid2eid, hidden_states, gate_weight = _make_case(32)

    topk_weights, topk_indices = hash_routing(
        input_ids, tid2eid, hidden_states, gate_weight
    )

    assert topk_weights.shape == (32, 6)
    assert topk_indices.shape == (32, 6)


def test_bos_token():
    from batchgen_kernels.moe.v4_hash_routing import hash_routing

    torch.manual_seed(42)
    _, tid2eid, hidden_states, gate_weight = _make_case(1)
    input_ids = torch.zeros(1, device="cuda", dtype=torch.long)

    _, topk_indices = hash_routing(
        input_ids, tid2eid, hidden_states, gate_weight
    )

    assert torch.equal(topk_indices[0], tid2eid[0].long())


def test_max_vocab_id():
    from batchgen_kernels.moe.v4_hash_routing import hash_routing

    torch.manual_seed(42)
    _, tid2eid, hidden_states, gate_weight = _make_case(1)
    input_ids = torch.full((1,), 129279, device="cuda", dtype=torch.long)

    _, topk_indices = hash_routing(
        input_ids, tid2eid, hidden_states, gate_weight
    )

    assert torch.equal(topk_indices[0], tid2eid[129279].long())


def test_all_same_input_id():
    from batchgen_kernels.moe.v4_hash_routing import hash_routing

    torch.manual_seed(42)
    _, tid2eid, hidden_states, gate_weight = _make_case(32)
    input_ids = torch.full((32,), 17, device="cuda", dtype=torch.long)
    hidden_states = hidden_states[:1].expand(32, -1).contiguous()

    topk_weights, topk_indices = hash_routing(
        input_ids, tid2eid, hidden_states, gate_weight
    )

    assert torch.equal(topk_indices, topk_indices[:1].expand_as(topk_indices))
    assert torch.allclose(
        topk_weights, topk_weights[:1].expand_as(topk_weights)
    )


def test_fallback_no_input_ids():
    import torch.nn.functional as F

    from batchgen_kernels.moe.v4_hash_routing import hash_routing

    torch.manual_seed(42)
    _, tid2eid, hidden_states, gate_weight = _make_case(32)

    topk_weights, topk_indices = hash_routing(
        None,
        tid2eid,
        hidden_states,
        gate_weight,
        topk=6,
        route_scale=1.0,
        score_func="sqrtsoftplus",
        norm_topk_prob=True,
    )
    scores = F.softplus(
        F.linear(hidden_states.float(), gate_weight.float())
    ).sqrt()
    expected_indices = torch.topk(scores, k=6, dim=-1)[1]
    expected_weights = scores.gather(-1, expected_indices)
    expected_weights = expected_weights / (
        expected_weights.sum(dim=-1, keepdim=True) + 1e-20
    )

    assert torch.equal(topk_indices, expected_indices)
    assert torch.allclose(topk_weights, expected_weights)


@pytest.mark.parametrize("T", [128, 1024, 4096])
def test_benchmark(T):
    from tests.kernels.conftest import _bench

    torch.manual_seed(42)
    input_ids, tid2eid, _, _ = _make_case(T)

    lookup_ms = _bench(lambda ids, table: table[ids], input_ids, tid2eid)
    index_select_ms = _bench(torch.index_select, tid2eid, 0, input_ids)
    print(
        f"\nK10 benchmark T={T} lookup={lookup_ms:.3f} ms index_select={index_select_ms:.3f} ms"
    )

    assert lookup_ms > 0
    assert index_select_ms > 0
