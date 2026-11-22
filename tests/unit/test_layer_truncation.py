"""The --debug-num-layers contract: one resolved value at config parse.

``apply_layer_truncation`` is called once per model initializer, on the rich
config, before any consumer reads it; every layer-count consumer then follows
``num_hidden_layers`` off that one object. These tests pin:

* the production path (None) changes no consumed value and stamps the full
  count;
* truncation resolves the one value, slices the per-layer tables (GLM
  ``indexer_types``, Kimi ``linear_attn_config`` layer lists) to a consistent
  first-N prefix, and re-runs the config's own ``validate()``;
* refusals are by name: non-positive N, N above the checkpoint, a model whose
  initializer never routed through the helper
  (``require_layer_truncation_applied``);
* the real GLM-5.2 and Kimi-Linear config objects — the consumers' single
  source of truth — hold the truncated value and consistent per-layer views.
"""

from dataclasses import asdict
from types import SimpleNamespace

import pytest

from batchgen.config.layer_truncation import (
    FULL_LAYER_COUNT_FIELD,
    LayerTruncationError,
    apply_layer_truncation,
    require_layer_truncation_applied,
)


def _plain_config(num_layers=36, model_type="gpt_oss"):
    return SimpleNamespace(num_hidden_layers=num_layers, model_type=model_type)


# -- the one resolved value --------------------------------------------------

def test_production_path_is_a_stamp_only_no_op():
    config = _plain_config()
    returned = apply_layer_truncation(config, None)
    assert returned is config
    assert config.num_hidden_layers == 36
    assert getattr(config, FULL_LAYER_COUNT_FIELD) == 36


def test_truncation_resolves_the_value_and_stamps_the_full_count():
    config = _plain_config()
    apply_layer_truncation(config, 4)
    assert config.num_hidden_layers == 4
    assert getattr(config, FULL_LAYER_COUNT_FIELD) == 36


def test_full_depth_request_is_allowed_and_reports_equal_counts():
    config = _plain_config()
    apply_layer_truncation(config, 36)
    assert config.num_hidden_layers == 36
    assert getattr(config, FULL_LAYER_COUNT_FIELD) == 36


@pytest.mark.parametrize("bad", [0, -1])
def test_non_positive_layer_counts_are_refused_by_name(bad):
    with pytest.raises(LayerTruncationError, match="--debug-num-layers"):
        apply_layer_truncation(_plain_config(), bad)


def test_a_count_above_the_checkpoint_is_refused_by_name():
    with pytest.raises(LayerTruncationError, match="exceeds the checkpoint"):
        apply_layer_truncation(_plain_config(num_layers=36), 37)


# -- the worker-side backstop ------------------------------------------------

def test_worker_backstop_passes_a_resolved_config_and_the_production_path():
    config = _plain_config()
    apply_layer_truncation(config, 4)
    require_layer_truncation_applied("openai/gpt-oss-120b", config, 4)
    require_layer_truncation_applied("openai/gpt-oss-120b", _plain_config(), None)


def test_worker_backstop_refuses_a_model_that_ignored_the_flag():
    # An initializer that never called apply_layer_truncation: no stamp,
    # full-depth config. Serving it as a truncated run would be silent.
    with pytest.raises(LayerTruncationError, match="deepseek"):
        require_layer_truncation_applied(
            "deepseek-ai/DeepSeek-V3", _plain_config(), 4
        )


# -- GLM-5.2: the rich config consumers read is truncated consistently -------

@pytest.fixture(scope="module")
def glm52():
    from batchgen.config.batchgen_model_config import BatchGenModelConfig

    return BatchGenModelConfig.resolve("zai-org/GLM-5.2-FP8", None)


def test_glm52_default_path_changes_no_consumed_value(glm52):
    before = asdict(glm52)
    apply_layer_truncation(glm52, None)
    assert asdict(glm52) == before  # every declared field byte-identical
    assert getattr(glm52, FULL_LAYER_COUNT_FIELD) == before["num_hidden_layers"]


def test_glm52_truncation_slices_indexer_types_and_stays_consistent():
    from batchgen.config.batchgen_model_config import BatchGenModelConfig
    from batchgen.models.glm.glm5.config import (
        assert_indexer_schedule_consistent,
        dsa_layer_skips_topk,
    )

    config = BatchGenModelConfig.resolve("zai-org/GLM-5.2-FP8", None)
    full = config.num_hidden_layers
    # The per-layer list a GLM-5.2 checkpoint ships, derived from the formula
    # so the pair is consistent at full depth.
    config.indexer_types = [
        "shared" if dsa_layer_skips_topk(config, i) else "full"
        for i in range(full)
    ]
    assert_indexer_schedule_consistent(config)

    apply_layer_truncation(config, 12)
    assert config.num_hidden_layers == 12
    assert len(config.indexer_types) == 12
    assert getattr(config, FULL_LAYER_COUNT_FIELD) == full
    # The sliced list still agrees with the formula over the served prefix —
    # the check the GLM initializer runs after truncation.
    assert_indexer_schedule_consistent(config)


def test_glm52_truncation_below_the_dense_prefix_fails_the_configs_validate():
    from batchgen.config.batchgen_model_config import BatchGenModelConfig

    config = BatchGenModelConfig.resolve("zai-org/GLM-5.2-FP8", None)
    assert config.first_k_dense_replace == 3
    with pytest.raises(ValueError, match="first_k_dense_replace"):
        apply_layer_truncation(config, 3)


# -- Kimi-Linear: the KDA/MLA split keeps a consistent first-N prefix --------

def _kimi_config(num_layers=27):
    from batchgen.models.moonshotai.kimi_linear.config import KimiLinearConfig

    # The 48B layout: 1-in-4 layers are MLA, listed 1-indexed.
    full_attn = [i for i in range(1, num_layers + 1) if i % 4 == 0]
    kda = [i for i in range(1, num_layers + 1) if i % 4 != 0]
    return KimiLinearConfig(
        num_hidden_layers=num_layers,
        linear_attn_config={
            "kda_layers": kda,
            "full_attn_layers": full_attn,
            "num_heads": 32,
            "head_dim": 128,
        },
    )


def test_kimi_truncation_slices_both_layer_lists_to_a_partition_of_first_n():
    config = _kimi_config()
    apply_layer_truncation(config, 10)
    lac = config.linear_attn_config
    kda = set(lac["kda_layers"])
    full_attn = set(lac["full_attn_layers"])
    # The invariant validate_k3_config enforces, over the served prefix.
    assert not (kda & full_attn)
    assert kda | full_attn == set(range(1, 11))
    # The split consumers (KV coordinator, KDA state plan, wrappers) read
    # through these accessors.
    assert config.num_kda_layers == len(kda) == 8
    assert config.num_mla_layers == len(full_attn) == 2
    assert [config.is_kda_layer(i) for i in range(10)] == [
        (i + 1) % 4 != 0 for i in range(10)
    ]


def test_kimi_k31_dsa_layers_slice_identically_to_the_full_attention_list():
    # K3.1: dsa_layers must equal full_attn_layers (validate_k3_config), so
    # both lists slice to the same prefix subset.
    config = _kimi_config()
    config.dsa_config = {
        "dsa_layers": list(config.linear_attn_config["full_attn_layers"]),
        "index_topk": 2048,
        "index_head_dim": 128,
        "index_num_heads": 64,
    }
    apply_layer_truncation(config, 10)
    assert config.dsa_config["dsa_layers"] == [4, 8]
    assert config.linear_attn_config["full_attn_layers"] == [4, 8]


def test_kimi_production_path_keeps_the_full_split():
    config = _kimi_config()
    apply_layer_truncation(config, None)
    assert config.num_hidden_layers == 27
    # The synthetic 1-in-4 split over 27 layers: 6 MLA (4,8,...,24), 21 KDA.
    assert config.num_kda_layers == 21
    assert config.num_mla_layers == 6


def _write_ckpt(tmp_path, files):
    import json

    for name, tensors in files.items():
        (tmp_path / name).write_text(
            json.dumps({"file_name": name[:-5] + ".bin", "total_byte_size": 0,
                        "state_dict": {t: {} for t in tensors}})
        )
    (tmp_path / "config.txt").write_text("not metadata")
    return str(tmp_path)


def _served(n, full):
    from types import SimpleNamespace

    return SimpleNamespace(num_hidden_layers=n, num_hidden_layers_full=full)


def test_all_layers_served_selects_nothing_so_every_file_is_read(tmp_path):
    from batchgen.config.layer_truncation import select_weight_files

    ckpt = _write_ckpt(tmp_path, {"layer_0.json": ["model.layers.0.a"]})
    assert select_weight_files(ckpt, _served(4, 4)) == ([], [])


def test_truncation_selects_files_holding_a_served_or_non_layer_tensor(tmp_path):
    from batchgen.config.layer_truncation import select_weight_files

    ckpt = _write_ckpt(tmp_path, {
        "a.json": ["model.layers.0.x", "model.layers.1.x"],
        "b.json": ["model.layers.1.y", "model.layers.2.y"],   # mixed: layer 1 is served
        "c.json": ["model.layers.2.z", "model.layers.3.z"],   # only layers that are not served
        "d.json": ["model.embed_tokens.weight"],              # not a layer tensor
        "e.json": ["model.layers.10.x"],                      # beyond the checkpoint's depth: kept
        "f.json": ["model.layers.12.mlp.experts.3.w"],        # expert index is not a layer index
    })
    files, unserved = select_weight_files(ckpt, _served(2, 4))
    assert files == ["a.json", "b.json", "d.json", "e.json", "f.json"]
    # b.json is read for its layer-1 tensor; its layer-2 tensor is not published.
    assert unserved == ["model.layers.2.y"]


def test_truncation_refuses_an_empty_selection_and_foreign_metadata(tmp_path):
    import json

    import pytest

    from batchgen.config.layer_truncation import (
        LayerTruncationError,
        select_weight_files,
    )

    ckpt = _write_ckpt(tmp_path, {"c.json": ["model.layers.3.z"]})
    with pytest.raises(LayerTruncationError, match="no converted-checkpoint file"):
        select_weight_files(ckpt, _served(2, 4))
    (tmp_path / "x.json").write_text(json.dumps({"other": 1}))
    with pytest.raises(LayerTruncationError, match="no state_dict"):
        select_weight_files(ckpt, _served(2, 4))


def test_reservation_follows_the_selected_files(tmp_path):
    import json

    import pytest

    from batchgen.config.layer_truncation import (
        LayerTruncationError,
        weight_reservation_bytes,
    )

    for name, size in (("a.json", 5000), ("b.json", 4096), ("c.json", 10)):
        (tmp_path / name).write_text(json.dumps({"total_byte_size": size, "state_dict": {}}))
    ckpt = str(tmp_path)
    mib = 1024 * 1024
    # No selection: the model's own reservation stands.
    assert weight_reservation_bytes(7 * mib, ckpt, []) == 7 * mib
    # a at 0 (5000 B), b on the next 4 KiB boundary (8192) -> ends at 12288; 2 MiB round-up.
    assert weight_reservation_bytes(7 * mib, ckpt, ["a.json", "b.json"]) == 2 * mib
    with pytest.raises(LayerTruncationError, match="more than the model's full reservation"):
        weight_reservation_bytes(mib, ckpt, ["a.json", "b.json"])
