import ctypes
from types import SimpleNamespace

import pytest
import torch

from batchgen.models.openai.gpt_oss_120b.wrappers import GptOssAttnWrapper
from batchgen.models.wrappers import AttnWrapperBase


class _FakeHostPagedKVWorkerView:
    def __init__(self, k_pages, v_pages):
        self._k_arrays = [self._page_to_ctypes(page) for page in k_pages]
        self._v_arrays = [self._page_to_ctypes(page) for page in v_pages]

    @staticmethod
    def _page_to_ctypes(page: torch.Tensor):
        raw = page.contiguous().view(torch.uint16).flatten().tolist()
        array_type = ctypes.c_uint16 * len(raw)
        return array_type(*raw)

    def get_sequence_layer_page_pointers(self, sequence_id, layer_idx, max_tokens=None):
        return (
            [ctypes.addressof(array) for array in self._k_arrays],
            [ctypes.addressof(array) for array in self._v_arrays],
        )


@pytest.fixture(autouse=True)
def _reset_prefix_reuse_metadata():
    old_mode = AttnWrapperBase.prepack_prefix_reuse_mode
    old_tokens = AttnWrapperBase.prepack_prefix_shared_tokens
    old_lengths = AttnWrapperBase.prepack_full_seq_lengths
    yield
    AttnWrapperBase.prepack_prefix_reuse_mode = old_mode
    AttnWrapperBase.prepack_prefix_shared_tokens = old_tokens
    AttnWrapperBase.prepack_full_seq_lengths = old_lengths


def _make_wrapper(k_page: torch.Tensor, v_page: torch.Tensor) -> GptOssAttnWrapper:
    wrapper = GptOssAttnWrapper.__new__(GptOssAttnWrapper)
    wrapper.layer_idx = 0
    wrapper.num_kv_heads = 1
    wrapper.head_dim = 2
    wrapper.engine_config = SimpleNamespace(
        Host_Paged_KV_Config=SimpleNamespace(page_size=4)
    )
    wrapper.core_engine = SimpleNamespace(
        host_paged_kv_worker_view=_FakeHostPagedKVWorkerView([k_page], [v_page])
    )
    return wrapper


def test_build_prefix_reuse_attention_kv_loads_host_prefix_and_appends_suffix():
    prefix_k = torch.tensor(
        [
            [[1.0, 1.5]],
            [[2.0, 2.5]],
            [[3.0, 3.5]],
            [[4.0, 4.5]],
        ],
        dtype=torch.bfloat16,
    )
    prefix_v = prefix_k + 10
    wrapper = _make_wrapper(prefix_k, prefix_v)

    suffix_k = torch.tensor(
        [
            [[5.0, 5.5]],
            [[6.0, 6.5]],
            [[20.0, 20.5]],
            [[21.0, 21.5]],
            [[22.0, 22.5]],
        ],
        dtype=torch.bfloat16,
    )
    suffix_v = suffix_k + 100
    cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)

    AttnWrapperBase.prepack_prefix_shared_tokens = [4, 0]
    AttnWrapperBase.prepack_full_seq_lengths = [6, 3]

    key, value, cu_k, max_k = wrapper._build_prefix_reuse_attention_kv(
        key=suffix_k,
        value=suffix_v,
        cu_seqlens=cu_seqlens,
        seq_lengths=[2, 3],
        global_sequence_ids=[101, 102],
    )

    torch.testing.assert_close(
        key,
        torch.cat([prefix_k, suffix_k[:2], suffix_k[2:]], dim=0),
    )
    torch.testing.assert_close(
        value,
        torch.cat([prefix_v, suffix_v[:2], suffix_v[2:]], dim=0),
    )
    assert cu_k.tolist() == [0, 6, 9]
    assert max_k == 6


def test_build_prefix_reuse_attention_kv_rejects_inconsistent_lengths():
    prefix_k = torch.ones((4, 1, 2), dtype=torch.bfloat16)
    wrapper = _make_wrapper(prefix_k, prefix_k)

    AttnWrapperBase.prepack_prefix_shared_tokens = [4]
    AttnWrapperBase.prepack_full_seq_lengths = [7]

    with pytest.raises(RuntimeError, match="full length mismatch"):
        wrapper._build_prefix_reuse_attention_kv(
            key=torch.ones((2, 1, 2), dtype=torch.bfloat16),
            value=torch.ones((2, 1, 2), dtype=torch.bfloat16),
            cu_seqlens=torch.tensor([0, 2], dtype=torch.int32),
            seq_lengths=[2],
            global_sequence_ids=[101],
        )


def test_build_full_hit_attention_kv_uses_cached_full_prompt():
    prefix_k = torch.tensor(
        [
            [[1.0, 1.5]],
            [[2.0, 2.5]],
            [[3.0, 3.5]],
            [[4.0, 4.5]],
        ],
        dtype=torch.bfloat16,
    )
    prefix_v = prefix_k + 10
    wrapper = _make_wrapper(prefix_k, prefix_v)

    AttnWrapperBase.prepack_full_seq_lengths = [4]

    key, value, cu_k, max_k = wrapper._build_full_hit_attention_kv(
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        seq_lengths=[1],
        global_sequence_ids=[101],
    )

    torch.testing.assert_close(key, prefix_k)
    torch.testing.assert_close(value, prefix_v)
    assert cu_k.tolist() == [0, 4]
    assert max_k == 4
