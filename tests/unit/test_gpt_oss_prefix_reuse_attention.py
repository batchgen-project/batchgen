import pytest
import torch

from batchgen.models.openai.gpt_oss_120b.wrappers import GptOssAttnWrapper
from batchgen.models.wrappers import AttnWrapperBase


@pytest.fixture(autouse=True)
def _reset_prefix_reuse_metadata():
    old_cu = AttnWrapperBase.prepack_cu_seqlens
    old_max = AttnWrapperBase.prepack_max_seqlen
    old_num = AttnWrapperBase.prepack_num_sequences
    old_seq_lengths = AttnWrapperBase.prepack_seq_lengths
    old_batch = AttnWrapperBase.cur_batch
    old_mode = AttnWrapperBase.prepack_prefix_reuse_mode
    old_tokens = AttnWrapperBase.prepack_prefix_shared_tokens
    old_lengths = AttnWrapperBase.prepack_full_seq_lengths
    yield
    AttnWrapperBase.prepack_cu_seqlens = old_cu
    AttnWrapperBase.prepack_max_seqlen = old_max
    AttnWrapperBase.prepack_num_sequences = old_num
    AttnWrapperBase.prepack_seq_lengths = old_seq_lengths
    AttnWrapperBase.cur_batch = old_batch
    AttnWrapperBase.prepack_prefix_reuse_mode = old_mode
    AttnWrapperBase.prepack_prefix_shared_tokens = old_tokens
    AttnWrapperBase.prepack_full_seq_lengths = old_lengths


def _make_wrapper() -> GptOssAttnWrapper:
    wrapper = GptOssAttnWrapper.__new__(GptOssAttnWrapper)
    wrapper.layer_idx = 0
    return wrapper


def test_gpt_oss_wrapper_no_longer_exposes_host_prefix_kv_reader():
    wrapper = _make_wrapper()

    assert not hasattr(wrapper, "host_prefix_reader")
    assert not hasattr(wrapper, "prefix_attention_kv_builder")


def test_prefix_cache_metadata_rejects_inconsistent_lengths():
    wrapper = _make_wrapper()

    AttnWrapperBase.prepack_cu_seqlens = torch.tensor([0, 2], dtype=torch.int32)
    AttnWrapperBase.prepack_max_seqlen = 2
    AttnWrapperBase.prepack_num_sequences = 1
    AttnWrapperBase.prepack_seq_lengths = [2]
    AttnWrapperBase.cur_batch = [101]
    AttnWrapperBase.prepack_prefix_reuse_mode = True
    AttnWrapperBase.prepack_prefix_shared_tokens = [4]
    AttnWrapperBase.prepack_full_seq_lengths = [7]

    with pytest.raises(RuntimeError, match="full length mismatch"):
        wrapper.prefix_cache_metadata()


def test_clamped_full_hit_metadata_is_normal_prefix_reuse():
    wrapper = _make_wrapper()

    AttnWrapperBase.prepack_cu_seqlens = torch.tensor([0, 1], dtype=torch.int32)
    AttnWrapperBase.prepack_max_seqlen = 1
    AttnWrapperBase.prepack_num_sequences = 1
    AttnWrapperBase.prepack_seq_lengths = [1]
    AttnWrapperBase.cur_batch = [101]
    AttnWrapperBase.prepack_prefix_reuse_mode = True
    AttnWrapperBase.prepack_prefix_shared_tokens = [4]
    AttnWrapperBase.prepack_full_seq_lengths = [5]

    metadata = wrapper.prefix_cache_metadata()

    assert metadata.prefix_reuse_mode is True
    assert metadata.prefix_shared_tokens == [4]
    assert metadata.full_seq_lengths == [5]
