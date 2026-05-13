import torch

from batchgen.attention.mla import flashinfer_paged_prefill


def test_flashinfer_mla_paged_prefill_builds_wrapper_inputs(monkeypatch):
    calls = {}

    class FakeWrapper:
        def __init__(self, workspace, backend):
            calls["workspace"] = workspace
            calls["backend"] = backend

        def plan(
            self,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_len_arr,
            num_heads,
            head_dim_ckv,
            head_dim_kpe,
            page_size,
            causal,
            sm_scale,
            q_data_type,
            kv_data_type,
        ):
            calls["plan"] = {
                "qo_indptr": qo_indptr.clone(),
                "kv_indptr": kv_indptr.clone(),
                "kv_indices": kv_indices.clone(),
                "kv_len_arr": kv_len_arr.clone(),
                "num_heads": num_heads,
                "head_dim_ckv": head_dim_ckv,
                "head_dim_kpe": head_dim_kpe,
                "page_size": page_size,
                "causal": causal,
                "sm_scale": sm_scale,
                "q_data_type": q_data_type,
                "kv_data_type": kv_data_type,
            }

        def run(self, q_nope, q_pe, ckv_cache, kpe_cache):
            calls["run"] = {
                "q_nope": q_nope,
                "q_pe": q_pe,
                "ckv_cache": ckv_cache,
                "kpe_cache": kpe_cache,
            }
            return torch.ones_like(q_nope)

    flashinfer_paged_prefill._reset_flashinfer_mla_paged_prefill_cache_for_tests()
    monkeypatch.setattr(
        flashinfer_paged_prefill,
        "_WRAPPER_CLASS_FOR_TESTS",
        FakeWrapper,
    )

    query_states = torch.zeros(1, 3, 2, 6)
    compressed_kv_cache = torch.zeros(5, 16, 1, 6)
    page_table = torch.tensor(
        [
            [3, 4, 1],
            [8, 7, 6],
            [2, 0, 9],
        ],
        dtype=torch.int32,
    )
    slot_indices = torch.tensor([2, 0], dtype=torch.int32)
    cache_seqlens = torch.tensor([17, 33], dtype=torch.int32)
    cu_seqlens_q = torch.tensor([0, 1, 3], dtype=torch.int32)

    output = flashinfer_paged_prefill.run_flashinfer_mla_paged_suffix_prefill(
        query_states=query_states,
        compressed_kv_cache=compressed_kv_cache,
        page_table=page_table,
        slot_indices=slot_indices,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        kv_lora_rank=4,
        num_heads=2,
        softmax_scale=0.25,
    )

    plan = calls["plan"]
    assert output.shape == (1, 3, 2, 4)
    assert calls["backend"] == "auto"
    assert torch.equal(plan["qo_indptr"], cu_seqlens_q)
    assert torch.equal(plan["kv_indptr"], torch.tensor([0, 2, 5], dtype=torch.int32))
    assert torch.equal(
        plan["kv_indices"],
        torch.tensor([2, 0, 3, 4, 1], dtype=torch.int32),
    )
    assert torch.equal(plan["kv_len_arr"], cache_seqlens)
    assert plan["num_heads"] == 2
    assert plan["head_dim_ckv"] == 4
    assert plan["head_dim_kpe"] == 2
    assert plan["page_size"] == 16
    assert plan["causal"] is True
    assert plan["sm_scale"] == 0.25
    assert calls["run"]["q_nope"].shape == (3, 2, 4)
    assert calls["run"]["q_pe"].shape == (3, 2, 2)
    assert calls["run"]["ckv_cache"].shape == (5, 16, 4)
    assert calls["run"]["kpe_cache"].shape == (5, 16, 2)
