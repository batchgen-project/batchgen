"""The parallel-declaration contract: vocabulary, rendering, worker read."""

import pytest

from batchgen import parallel_declaration as pd


def slot(scheme, residency, group=1):
    return pd.ModuleParallelism(scheme, residency, group_size=group)


def declaration(**overrides):
    values = dict(
        prefill_attention=slot("dp", "streamed"),
        prefill_moe=slot("dp", "streamed"),
        decode_attention=slot("dp", "resident"),
        decode_moe=slot("ep", "resident", group=8),
    )
    values.update(overrides)
    return pd.ParallelDeclaration(**values)


def graph_declaration(mode="graph", buckets=(1, 2, 4)):
    return pd.DecodeGraphDeclaration(mode=mode, buckets=buckets)


# -- vocabulary --------------------------------------------------------------

@pytest.mark.parametrize("scheme, residency, group, rendered", [
    ("dp", "streamed", 1, "dp:streamed"),
    ("dp", "resident", 1, "dp:resident"),
    ("tp", "resident", 8, "tp8:resident"),
    ("tp", "streamed", 1, "tp1:streamed"),
    ("ep", "partial", 4, "ep4:partial"),
    ("ep", "resident", 1, "ep1:resident"),
    ("none", None, 1, "none"),
])
def test_render_tokens(scheme, residency, group, rendered):
    assert slot(scheme, residency, group).render() == rendered


@pytest.mark.parametrize("scheme, residency, group", [
    ("tensor", "resident", 1),      # unknown scheme
    ("dp", "cached", 1),            # unknown residency
    ("dp", None, 1),                # dp needs a residency
    ("dp", "resident", 2),          # dp has no shard group
    ("ep", "resident", 0),          # grouped schemes need >= 1
    ("tp", "resident", True),       # bool is not a width
    ("none", "resident", 1),        # none carries no residency
    ("none", None, 4),              # none carries no group
])
def test_vocabulary_is_closed(scheme, residency, group):
    with pytest.raises(pd.ParallelDeclarationError):
        slot(scheme, residency, group)


def test_declaration_slots_are_typed_and_consistent():
    full = declaration()
    assert full.rendered_slots() == {
        "prefill_attention": "dp:streamed",
        "prefill_moe": "dp:streamed",
        "decode_attention": "dp:resident",
        "decode_moe": "ep8:resident",
    }
    with pytest.raises(pd.ParallelDeclarationError):
        declaration(prefill_moe="dp:streamed")  # token, not a ModuleParallelism
    with pytest.raises(pd.ParallelDeclarationError):
        declaration(prefill_moe=slot("none", None))  # MoE in one phase only
    with pytest.raises(pd.ParallelDeclarationError):
        declaration(decode_attention=slot("none", None))  # attention always exists
    dense = declaration(
        prefill_moe=slot("none", None), decode_moe=slot("none", None)
    )
    assert dense.rendered_slots()["decode_moe"] == "none"


def test_undeclared_is_reserved_and_covers_every_slot():
    assert pd.undeclared_slots() == {
        "prefill_attention": "undeclared",
        "prefill_moe": "undeclared",
        "decode_attention": "undeclared",
        "decode_moe": "undeclared",
    }
    assert pd.UNDECLARED not in pd.SCHEMES and pd.UNDECLARED not in pd.RESIDENCIES


@pytest.mark.parametrize("mode", ["graph", "compare"])
def test_decode_graph_declaration_validates_and_carries_buckets(mode):
    assert graph_declaration(mode).buckets == (1, 2, 4)
    assert pd.DecodeGraphDeclaration.off() == pd.DecodeGraphDeclaration()


def test_decode_graph_declaration_rejects_invalid_values():
    with pytest.raises(pd.ParallelDeclarationError):
        pd.DecodeGraphDeclaration("unknown", (1,))
    with pytest.raises(pd.ParallelDeclarationError):
        pd.DecodeGraphDeclaration("graph", None)
    with pytest.raises(pd.ParallelDeclarationError):
        pd.DecodeGraphDeclaration("off", (1,))
    with pytest.raises(pd.ParallelDeclarationError):
        pd.DecodeGraphDeclaration("graph", (2, 1))


# -- the worker-side read ----------------------------------------------------

class _Declared:
    def parallel_declaration(self, decode_graph_state=None):
        return declaration(decode_graph=decode_graph_state or pd.DecodeGraphDeclaration.off())


class _Undeclared:
    pass


class _WrongType:
    def parallel_declaration(self):
        return {"prefill_attention": "dp:streamed"}


def test_read_is_optional_but_never_coerced():
    assert pd.read_parallel_declaration(_Declared()) == declaration()
    owner = type("Owner", (), {
        "whole_model_graph": True,
        "cuda_graph_manager": object(),
        "whole_model_bucketing": type("Bucketing", (), {"bucket_sizes": [1, 2, 4]})(),
    })()
    read = pd.read_parallel_declaration(_Declared(), owner)
    assert read.decode_graph == graph_declaration()
    assert pd.read_parallel_declaration(_Undeclared()) is None
    with pytest.raises(pd.ParallelDeclarationError, match="_WrongType"):
        pd.read_parallel_declaration(_WrongType())
