"""Host KV debug dump: what a held turn's pages look like on disk, and the
byte compare that decides whether two sessions' Host pages are equal.

The dump is the GPU validation's byte-equality instrument (GLM-5.3 session
hold, step 4b): the live comparison runs on the node against a real server;
these CPU tests hold the writer and the comparer to the layout the Host view
exposes. The fake view hands out real addresses into a CPU tensor laid out
``[layers, pages, layer_page_bytes]``, so the ctypes read is the one the
worker performs.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_k3_prefix_pool_group import _bind, _worker_methods  # noqa: E402

from batchgen.worker import hostkv_debug_dump as dump_lib  # noqa: E402

PAGE_TOKENS = 4
LAYERS = 3
LATENT = 8   # stands in for 576
INDEXER = 4  # stands in for 128
HOST_PAGES = 16
DTYPE = torch.bfloat16
HELD_PAGES = [9, 3, 12]  # token order, deliberately unsorted
KV_TOKENS = 10           # 2 whole pages + 2 rows of the third


class _HostRegion:
    """A Host region whose pages live at real addresses of a CPU tensor."""

    def __init__(self, layers, dim, seed):
        g = torch.Generator().manual_seed(seed)
        self.data = torch.randn(
            (layers, HOST_PAGES, PAGE_TOKENS, dim), generator=g
        ).to(DTYPE)
        self.layer_page_bytes = PAGE_TOKENS * dim * self.data.element_size()

    def k_page_ptr(self, layer, page):
        return self.data[layer, page].data_ptr()

    def page_bytes(self, layer, page):
        return self.data[layer, page].contiguous().view(torch.uint8).numpy().tobytes()


def _geometry(dim, *, v_heads=0, layer_map=None, layers=LAYERS):
    return SimpleNamespace(
        num_layers=layers, num_k_heads=1, k_head_dim=dim, num_v_heads=v_heads,
        kv_dtype=DTYPE, logical_to_physical_layer=layer_map,
    )


def _regions(latent, indexer, pages=HELD_PAGES):
    return [
        dump_lib.region_spec(group=0, view=latent, geometry=_geometry(LATENT),
                             page_tokens=PAGE_TOKENS, page_ids=pages),
        dump_lib.region_spec(group=1, view=indexer, geometry=_geometry(INDEXER),
                             page_tokens=PAGE_TOKENS, page_ids=pages),
    ]


def _dump(root, latent, indexer, *, session="glm53-x-A", request="req-1",
          kv_tokens=KV_TOKENS, pages=HELD_PAGES):
    return dump_lib.dump_held_pages(
        str(root), session_id=session, request_id=request, rank=5,
        kv_tokens=kv_tokens, page_tokens=PAGE_TOKENS,
        regions=_regions(latent, indexer, pages),
    )


# --------------------------------------------------------------------------- #
#  Env contract
# --------------------------------------------------------------------------- #


def test_env_off_by_default_and_prefix_mandatory():
    assert dump_lib.config_from_env({}) == ("", "")
    assert dump_lib.config_from_env({dump_lib.ENV_PREFIX: "x"}) == ("", "")
    with pytest.raises(ValueError, match=dump_lib.ENV_PREFIX):
        dump_lib.config_from_env({dump_lib.ENV_DIR: "/tmp/d"})
    assert dump_lib.config_from_env(
        {dump_lib.ENV_DIR: "/tmp/d", dump_lib.ENV_PREFIX: "glm53-"}
    ) == ("/tmp/d", "glm53-")


# --------------------------------------------------------------------------- #
#  Writer: layout on disk == the Host view's pages, page-major, both regions
# --------------------------------------------------------------------------- #


def test_dump_writes_both_regions_page_major_in_token_order(tmp_path):
    latent, indexer = _HostRegion(LAYERS, LATENT, 1), _HostRegion(LAYERS, INDEXER, 2)
    path = _dump(tmp_path, latent, indexer)
    assert os.path.basename(path) == "hold000_req-1"
    meta = json.loads(Path(path, "meta.json").read_text())
    assert meta["kv_tokens"] == KV_TOKENS and meta["page_tokens"] == PAGE_TOKENS
    assert meta["rank"] == 5 and meta["session_id"] == "glm53-x-A"
    for region, host, group in ((meta["regions"][0], latent, 0), (meta["regions"][1], indexer, 1)):
        assert region["group"] == group
        assert region["page_ids"] == HELD_PAGES
        assert region["layer_page_bytes"] == host.layer_page_bytes
        assert region["page_stride_bytes"] == host.layer_page_bytes
        expected = b"".join(
            host.page_bytes(layer, page)
            for page in HELD_PAGES for layer in range(LAYERS)
        )
        assert Path(path, region["file"]).read_bytes() == expected
    # A second hold of the same session gets its own directory, in order.
    second = _dump(tmp_path, latent, indexer, request="req-2")
    assert os.path.basename(second) == "hold001_req-2"
    assert not [n for n in os.listdir(os.path.dirname(path)) if n.endswith(".partial")]


def test_dump_refuses_a_page_list_that_does_not_cover_the_tokens(tmp_path):
    latent, indexer = _HostRegion(LAYERS, LATENT, 1), _HostRegion(LAYERS, INDEXER, 2)
    with pytest.raises(RuntimeError, match="names 2 page"):
        _dump(tmp_path, latent, indexer, pages=HELD_PAGES[:2])


@pytest.mark.parametrize(
    "geometry, match",
    [
        (_geometry(LATENT, v_heads=1), "V cache"),
        (_geometry(LATENT, layer_map=(0, 2, 1)), "identity layer map"),
    ],
)
def test_region_spec_refuses_what_it_cannot_dump_whole(geometry, match):
    with pytest.raises(RuntimeError, match=match):
        dump_lib.region_spec(group=0, view=_HostRegion(LAYERS, LATENT, 1),
                             geometry=geometry, page_tokens=PAGE_TOKENS,
                             page_ids=HELD_PAGES)


def test_region_spec_refuses_a_page_split_region():
    view = _HostRegion(LAYERS, INDEXER, 2)
    view.is_page_split = True
    with pytest.raises(RuntimeError, match="page-split"):
        dump_lib.region_spec(group=1, view=view, geometry=_geometry(INDEXER),
                             page_tokens=PAGE_TOKENS, page_ids=HELD_PAGES)


# --------------------------------------------------------------------------- #
#  Comparer
# --------------------------------------------------------------------------- #


def _pair(tmp_path, *, b_pages=HELD_PAGES):
    """Two dumps of the same bytes; B's pages may sit at other page ids."""
    latent, indexer = _HostRegion(LAYERS, LATENT, 1), _HostRegion(LAYERS, INDEXER, 2)
    a = _dump(tmp_path / "a", latent, indexer)
    for host in (latent, indexer):
        host.data[:, b_pages] = host.data[:, HELD_PAGES].clone()
    b = _dump(tmp_path / "b", latent, indexer, session="glm53-x-B", pages=b_pages)
    return a, b, latent, indexer


def test_compare_equal_dumps_at_other_page_ids(tmp_path):
    a, b, _, _ = _pair(tmp_path, b_pages=[1, 2, 4])
    result = dump_lib.compare_dumps(a, b)
    assert result["all_equal"]
    assert (result["whole_pages"], result["tail_rows"]) == (2, 2)
    for region in result["regions"].values():
        assert region["pages_compared"] == 3 and region["pages_equal"] == 3
        assert region["same_page_ids"] == 0


def test_compare_names_region_page_layer_and_offset_of_a_difference(tmp_path):
    latent, indexer = _HostRegion(LAYERS, LATENT, 1), _HostRegion(LAYERS, INDEXER, 2)
    a = _dump(tmp_path / "a", latent, indexer)
    # One indexer element, page index 1 (id 3), layer 2, token row 1, column 3.
    indexer.data[2, HELD_PAGES[1], 1, 3] += 1.0
    b = _dump(tmp_path / "b", latent, indexer, session="glm53-x-B")
    result = dump_lib.compare_dumps(a, b)
    assert not result["all_equal"]
    assert result["regions"]["0"]["pages_differing"] == 0
    aux = result["regions"]["1"]
    assert aux["pages_equal"] == 2 and aux["pages_differing"] == 1
    (diff,) = aux["diffs"]
    row = INDEXER * 2
    assert diff["page_index"] == 1 and diff["page_id_a"] == 3
    assert diff["first_diff_layer"] == 2
    assert diff["first_diff_token"] == 1 * PAGE_TOKENS + 1
    assert row + 3 * 2 <= diff["first_diff_offset_in_layer_page"] < row + 4 * 2
    assert diff["layers_differing"] == 1 and diff["max_abs_diff_bf16"] > 0


def test_compare_partial_page_reads_only_the_common_rows(tmp_path):
    latent, indexer = _HostRegion(LAYERS, LATENT, 1), _HostRegion(LAYERS, INDEXER, 2)
    a = _dump(tmp_path / "a", latent, indexer)
    # Row 3 of the third page is past KV_TOKENS (10 = 8 + 2 rows): ignored.
    latent.data[0, HELD_PAGES[2], 3] += 1.0
    b = _dump(tmp_path / "b", latent, indexer, session="glm53-x-B")
    assert dump_lib.compare_dumps(a, b)["all_equal"]
    # ...but inside the compared rows it is found, and --tokens narrows it.
    latent.data[0, HELD_PAGES[2], 1] += 1.0
    c = _dump(tmp_path / "c", latent, indexer, session="glm53-x-C")
    assert not dump_lib.compare_dumps(a, c)["all_equal"]
    assert dump_lib.compare_dumps(a, c, tokens=9)["all_equal"]
    with pytest.raises(ValueError, match="outside"):
        dump_lib.compare_dumps(a, c, tokens=KV_TOKENS + 1)


def test_compare_cli_exit_code(tmp_path, capsys):
    a, b, _, _ = _pair(tmp_path)
    assert dump_lib._main(["compare", a, b]) == 0
    assert json.loads(capsys.readouterr().out)["all_equal"] is True


# --------------------------------------------------------------------------- #
#  The worker hook (the real method, compiled without the CUDA stack)
# --------------------------------------------------------------------------- #


def _worker(monkeypatch, tmp_path, prefix="glm53-x-"):
    monkeypatch.setitem(
        sys.modules,
        "batchgen.kv_cache.host_kv_mananger_config",
        SimpleNamespace(
            build_gpu_kv_config=lambda *a, **k: _geometry(LATENT),
            build_gpu_kv_config_aux=lambda *a, **k: _geometry(INDEXER),
        ),
    )
    worker = SimpleNamespace(
        rank=5, huggingface_ckpt_name="zai-org/GLM-5.3-FP8",
        _session_hold=SimpleNamespace(page_tokens=PAGE_TOKENS),
    )
    return _bind(worker, _worker_methods(("_debug_hostkv_dump",), {
        "logging": logging,
        "hostkv_debug_dump_lib": dump_lib,
        "_DEBUG_HOSTKV_DUMP_DIR": str(tmp_path),
        "_DEBUG_HOSTKV_DUMP_SESSION_PREFIX": prefix,
    }))


def test_worker_hook_dumps_a_matching_session_from_both_views(monkeypatch, tmp_path):
    latent, indexer = _HostRegion(LAYERS, LATENT, 1), _HostRegion(LAYERS, INDEXER, 2)
    worker = _worker(monkeypatch, tmp_path)
    seq = SimpleNamespace(session_id="glm53-x-A", uuid="turn_1")
    worker._debug_hostkv_dump(
        seq, {1: indexer, 0: latent},
        {0: tuple(HELD_PAGES), 1: tuple(HELD_PAGES)}, KV_TOKENS,
    )
    (dump,) = os.listdir(tmp_path / "glm53-x-A")
    meta = json.loads((tmp_path / "glm53-x-A" / dump / "meta.json").read_text())
    assert [r["group"] for r in meta["regions"]] == [0, 1]
    assert [r["layer_page_bytes"] for r in meta["regions"]] == [
        latent.layer_page_bytes, indexer.layer_page_bytes,
    ]


def test_worker_hook_skips_other_sessions(monkeypatch, tmp_path):
    latent, indexer = _HostRegion(LAYERS, LATENT, 1), _HostRegion(LAYERS, INDEXER, 2)
    worker = _worker(monkeypatch, tmp_path)
    for sid in ("glm53-4b-prime-1", None):
        worker._debug_hostkv_dump(
            SimpleNamespace(session_id=sid, uuid="turn_p"), {0: latent, 1: indexer},
            {0: tuple(HELD_PAGES), 1: tuple(HELD_PAGES)}, KV_TOKENS,
        )
    assert os.listdir(tmp_path) == []
