import pytest
import torch

from batchgen.attention.dsa import sparse_decode_mla


class _ObjectSchedulerMetadata:
    pass


def test_eager_decode_passes_object_scheduler_metadata_without_batchgen_mutation(
    monkeypatch,
):
    scheduler = _ObjectSchedulerMetadata()
    seen = {}

    def fake_get_metadata(*_args, **_kwargs):
        return scheduler, None

    def fake_flash_mla(*args):
        seen["scheduler"] = args[5]
        seen["num_splits"] = args[6]
        return args[0], None

    monkeypatch.setattr(
        sparse_decode_mla,
        "_flash_mla_ops",
        lambda: (fake_flash_mla, fake_get_metadata),
    )

    prepared = sparse_decode_mla.prepare_sparse_flash_mla_decode_inputs(
        query_states=torch.zeros(1, 1, 4, 6),
        sparse_mla_kv=torch.zeros(1, 2, 1, 6),
        sparse_seqlens=torch.tensor([2], dtype=torch.int32),
        num_heads=4,
        softmax_scale=1.0,
        head_dim_v=4,
        page_size=2,
    )
    before = vars(scheduler).copy()

    sparse_decode_mla.run_prepared_sparse_flash_mla_decode(prepared)

    assert seen == {"scheduler": scheduler, "num_splits": None}
    assert vars(scheduler) == before


def test_captured_flashmla_decode_rejects_object_scheduler_api(monkeypatch):
    scheduler = _ObjectSchedulerMetadata()

    monkeypatch.setattr(
        sparse_decode_mla,
        "_flash_mla_ops",
        lambda: (None, lambda *_args, **_kwargs: (scheduler, None)),
    )

    with pytest.raises(TypeError, match="object scheduler API"):
        sparse_decode_mla.prepare_sparse_flash_mla_decode_tensor_metadata(
            torch.tensor([2], dtype=torch.int32),
            num_heads=4,
        )


def test_captured_flashmla_decode_accepts_tensor_scheduler_api(monkeypatch):
    tile_scheduler_metadata = torch.ones(2, 4, dtype=torch.int32)
    num_splits = torch.ones(2, dtype=torch.int32)

    monkeypatch.setattr(
        sparse_decode_mla,
        "_flash_mla_ops",
        lambda: (
            None,
            lambda *_args, **_kwargs: (tile_scheduler_metadata, num_splits),
        ),
    )

    returned = sparse_decode_mla.prepare_sparse_flash_mla_decode_tensor_metadata(
        torch.tensor([2], dtype=torch.int32),
        num_heads=4,
    )

    assert returned[0] is tile_scheduler_metadata
    assert returned[1] is num_splits
