from types import SimpleNamespace

import pytest
import torch

from batchgen.cuda_graph.graph_manager import BatchSizeBucketing, CUDAGraphManager
from batchgen.models.glm.glm5.whole_model_cuda_graph_segments import (
    Glm5WholeModelSegment,
    make_glm5_whole_model_graph_segment_name,
)
from batchgen.models.wrappers import AttnWrapperBase


class _FakeIndexer:
    index_head_dim = 128


class _FakeAttnModule:
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    indexer = _FakeIndexer()


class _FakeSelfAttn:
    module = _FakeAttnModule()


class _FakeLayer:
    self_attn = _FakeSelfAttn()


class _FakeInnerModel:
    layers = [_FakeLayer(), _FakeLayer()]


class _FakeGlm5Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _FakeInnerModel()

    def forward(self, input_ids, position_ids=None, use_cache=False):
        del position_ids, use_cache
        bsz = input_ids.shape[0]
        values = input_ids.to(torch.bfloat16).view(bsz, 1, 1, 1)
        primary = values.expand(bsz, 1, 1, 576)
        aux = (values + 1).expand(bsz, 1, 1, 128)
        for layer_idx in range(2):
            AttnWrapperBase.kv_append_callback(layer_idx, primary + layer_idx, None)
            AttnWrapperBase.kv_append_callback_aux(layer_idx, aux + layer_idx, None)
        logits = torch.zeros(bsz, 1, 8, dtype=torch.bfloat16, device=input_ids.device)
        logits[:, 0, 0] = input_ids[:, 0].to(torch.bfloat16)
        logits[:, 0, 1] = (input_ids[:, 0] + 10).to(torch.bfloat16)
        return SimpleNamespace(logits=logits)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph smoke requires CUDA")
def test_glm5_whole_model_segment_captures_logits_and_kv_callbacks():
    device = torch.device("cuda")
    segment = Glm5WholeModelSegment(
        model=_FakeGlm5Model().to(device),
        device=device,
        world_size=1,
        max_pages_per_seq=4,
        max_aux_pages_per_seq=4,
        vocab_size=8,
        hidden_size=16,
        max_bucket_size=2,
        max_seqlen=16,
    )
    segment.set_capture_inputs(
        input_ids=torch.tensor([[1], [2]], dtype=torch.int64, device=device),
        cache_seqlens=torch.tensor([4, 5], dtype=torch.int32, device=device),
        position_ids=torch.tensor([[3], [4]], dtype=torch.int64, device=device),
        primary_page_table=torch.zeros(2, 4, dtype=torch.int32, device=device),
        aux_page_table=torch.zeros(2, 4, dtype=torch.int32, device=device),
        primary_slot_indices=torch.tensor([0, 1], dtype=torch.int32, device=device),
        aux_slot_indices=torch.tensor([0, 1], dtype=torch.int32, device=device),
        rank_token_counts=torch.tensor([2], dtype=torch.int64, device=device),
        num_valid_tokens=torch.tensor([2], dtype=torch.int32, device=device),
    )

    manager = CUDAGraphManager(BatchSizeBucketing([2]), device=device)
    manager.register_segment(make_glm5_whole_model_graph_segment_name(), segment)
    manager.warmup_and_capture_buckets([2])

    out = manager.replay(
        make_glm5_whole_model_graph_segment_name(),
        2,
        input_ids=torch.tensor([[5], [6]], dtype=torch.int64, device=device),
        cache_seqlens=torch.tensor([8, 9], dtype=torch.int32, device=device),
        position_ids=torch.tensor([[7], [8]], dtype=torch.int64, device=device),
        primary_page_table=torch.zeros(2, 4, dtype=torch.int32, device=device),
        aux_page_table=torch.zeros(2, 4, dtype=torch.int32, device=device),
        primary_slot_indices=torch.tensor([0, 1], dtype=torch.int32, device=device),
        aux_slot_indices=torch.tensor([0, 1], dtype=torch.int32, device=device),
        rank_token_counts=torch.tensor([2], dtype=torch.int64, device=device),
        num_valid_tokens=torch.tensor([2], dtype=torch.int32, device=device),
    )
    torch.cuda.synchronize()

    assert out["logits"][:, :2].to(torch.float32).cpu().tolist() == [
        [5.0, 15.0],
        [6.0, 16.0],
    ]
    assert segment._kv_buffers[0]["key"][:2, 0, 0, 0].to(torch.float32).cpu().tolist() == [5.0, 6.0]
    assert segment._kv_buffers[1]["key"][:2, 0, 0, 0].to(torch.float32).cpu().tolist() == [6.0, 7.0]
    assert segment._aux_kv_buffers[0]["key"][:2, 0, 0, 0].to(torch.float32).cpu().tolist() == [6.0, 7.0]
    assert segment._aux_kv_buffers[1]["key"][:2, 0, 0, 0].to(torch.float32).cpu().tolist() == [7.0, 8.0]
