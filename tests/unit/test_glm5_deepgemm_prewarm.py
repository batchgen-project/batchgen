# ---------------------------------------------------------------------------- #
#  BatchGen — GLM-5                                                             #
#  Copyright (c) 2025-2026 BatchGen Team                                        #
#  Licensed under the Apache License, Version 2.0                               #
# ---------------------------------------------------------------------------- #
"""The GLM-5 DeepGEMM prewarm's site table, inventory and entries, on CPU.

Under test: ``batchgen/models/glm/glm5/deepgemm_prewarm.py``.

  * the site table's N/K/groups derive from the GLM-5.2 config fields to the
    values the serving GEMMs see;
  * inventory pin: every DeepGEMM call in the GLM-5 serving sources is named
    by ``CALL_SITES`` with its call count, and every named site is reached;
  * the entries cover every kernel key of their row ranges, are identical on
    every rank, and add the MoE masked GEMMs only with DeepEP and the paged
    MQA scorer only on the page-split cache;
  * the runners hand DeepGEMM the serving call's dtypes, shapes and scale
    layouts (recording fakes);
  * the version gate and the late-compile count.

Loaded by file path (no ``import batchgen``); no CUDA, Triton or DeepGEMM.

Run: python -m pytest tests/unit/test_glm5_deepgemm_prewarm.py -q
"""

import ast
import collections
import importlib.util
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[2]
_GLM5_DIR = _ROOT / "batchgen" / "models" / "glm" / "glm5"

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


DGP = _load(_GLM5_DIR / "deepgemm_prewarm.py", "_glm5_deepgemm_prewarm_under_test")
TP = _load(_GLM5_DIR / "triton_prewarm.py", "_glm5_triton_prewarm_for_deepgemm")
DG = DGP._dg

NUM_SMS = 132
MAX_ROWS = 131_072
SLOTS = 128


def _shapes(sparse_prefill=True):
    """GLM-5.2 dimensions (``config.py``)."""
    return TP.Glm5PrewarmShapes(
        hidden_size=6144, q_lora_rank=2048, kv_lora_rank=512,
        qk_nope_head_dim=192, qk_rope_head_dim=64, v_head_dim=256,
        num_heads=64, intermediate_size=12288, shared_intermediate_size=2048,
        expert_intermediate_size=2048, index_n_heads=32, index_head_dim=128,
        index_topk=2048, num_routed_experts=256, max_position_embeddings=1048576,
        sparse_prefill=sparse_prefill,
    )


def _stacks(groups=32):
    return DGP.MoeStacks(
        gate_up=torch.zeros((groups, 4096, 6144), dtype=torch.float8_e4m3fn,
                            device="meta"),
        gate_up_scale=torch.zeros((groups, 32, 48), device="meta"),
        down=torch.zeros((groups, 6144, 2048), dtype=torch.float8_e4m3fn,
                         device="meta"),
        down_scale=torch.zeros((groups, 48, 16), device="meta"),
    )


class _Recorder:
    """A fake serving function that keeps every call."""

    def __init__(self, result=None):
        self.calls = []
        self.result = result

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result(*args, **kwargs) if callable(self.result) else None


def _fake_quant(x, num_valid_tokens=None, scale_tma_aligned=False):
    """``act_quant``'s layouts: K-major ``[M, K/128]``, or the transposed
    view of a ``[K/128, align4(M)]`` allocation."""
    rows, dim = x.shape
    blocks = dim // 128
    if scale_tma_aligned:
        scale = torch.empty((blocks, TP.align4(rows))).transpose(0, 1)[:rows]
    else:
        scale = torch.empty((rows, blocks))
    return torch.empty((rows, dim), dtype=torch.float8_e4m3fn), scale


# --------------------------------------------------------------------------- #
#  Site table
# --------------------------------------------------------------------------- #
def _nk(sites):
    return {site.name: (site.n, site.k, site.via) for site in sites}


def test_prefill_sites_derive_the_serving_shapes():
    assert _nk(DGP.dense_sites(_shapes(), DGP.PHASE_PREFILL)) == {
        "q_a": (2048, 6144, "w8a16"),
        "kv_a": (576, 6144, "w8a16"),
        "q_b": (16384, 2048, "w8a16"),
        "kv_b": (28672, 512, "w8a16"),
        "o_proj": (6144, 16384, "w8a16"),
        "indexer_wk": (128, 6144, "w8a16"),
        "indexer_wq_b": (4096, 2048, "w8a16"),
        "dense_mlp_gate_up": (12288, 6144, "w8a16"),
        "dense_mlp_down": (6144, 12288, "w8a16"),
        "shared_gate_up_unfused": (2048, 6144, "w8a16"),
        "shared_down_unfused": (6144, 2048, "w8a16"),
    }
    dense = _nk(DGP.dense_sites(_shapes(False), DGP.PHASE_PREFILL))
    assert "indexer_wq_b" not in dense


def test_decode_sites_follow_the_merge_and_wk_policies():
    sites = _nk(DGP.dense_sites(_shapes(), DGP.PHASE_DECODE))
    assert sites == {
        "q_a": (2048, 6144, "w8a8"),
        "kv_a": (576, 6144, "w8a8"),
        "q_b": (16384, 2048, "w8a8"),
        "q_b_wq_b": (20480, 2048, "w8a8"),
        "o_proj": (6144, 16384, "w8a8"),
        "indexer_wk_graph": (128, 6144, "wk"),
        "dense_mlp_gate_up": (12288, 6144, "w8a16"),
        "dense_mlp_down": (6144, 12288, "w8a16"),
        "expert_gate_up": (2048, 6144, "w8a16"),
        "expert_down": (6144, 2048, "w8a16"),
        "shared_gate_up_fused": (4096, 6144, "tma"),
        "shared_down_fused": (6144, 2048, "tma"),
    }
    off = _nk(DGP.dense_sites(_shapes(), DGP.PHASE_DECODE,
                              merge_q_b_wq_b=False, wk_deepgemm=False))
    assert "q_b_wq_b" not in off and "indexer_wk_graph" not in off
    with pytest.raises(ValueError):
        DGP.dense_sites(_shapes(), DGP.PHASE_GEOMETRY)


def test_masked_sites():
    assert [(s.name, s.n, s.k, s.groups)
            for s in DGP.masked_dense_sites(_shapes())] == [
        ("qkv_a", 2624, 6144, 1), ("q_b", 16384, 2048, 1),
        ("q_b_wq_b", 20480, 2048, 1), ("o_proj", 6144, 16384, 1)]
    assert "q_b_wq_b" not in [
        s.name for s in DGP.masked_dense_sites(_shapes(), merge_q_b_wq_b=False)]
    assert [(s.name, s.n, s.k, s.groups)
            for s in DGP.masked_moe_sites(_shapes(), 32)] == [
        ("moe_gate_up", 4096, 6144, 32), ("moe_down", 6144, 2048, 32)]


def test_expected_m_per_expert_is_the_moe_paths():
    moe = _load(_GLM5_DIR / "moe_deepgemm_masked.py",
                "_glm5_moe_deepgemm_masked_for_prewarm")
    for rows in (1, 2, 31, 32, 33, 128, 129, 4096):
        for world, topk, experts in ((8, 8, 256), (16, 8, 256), (1, 1, 32)):
            kwargs = dict(local_rows=rows, world_size=world, topk=topk,
                          num_experts_global=experts)
            assert DGP.expected_m_per_expert(**kwargs) == \
                moe.expected_m_per_expert(**kwargs)
    assert DGP.MOE_MASKED_ROWS % moe.MASKED_BLOCK_M == 0


def test_page_split_scale_bytes_match_the_cache_format():
    quant = _load(_ROOT / "batchgen_kernels" / "attention" / "dsa"
                  / "indexer_k_quant.py", "_indexer_k_quant_for_prewarm")
    assert DGP.PAGE_SPLIT_SCALE_BYTES == quant.SCALE_BYTES_PER_TOKEN


# --------------------------------------------------------------------------- #
#  Inventory pin
# --------------------------------------------------------------------------- #
def _scan_calls():
    """``(path, enclosing function, callee) -> count`` over the inventory."""
    files = set()
    for rel in DGP.INVENTORY_PATHS:
        path = _ROOT / rel
        files.update([path] if path.is_file() else path.rglob("*.py"))
    callees = set(DGP.INVENTORY_CALLEES)
    found = collections.Counter()

    class Visitor(ast.NodeVisitor):
        def __init__(self, rel):
            self.rel, self.stack = rel, []

        def _scope(self, node):
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        visit_ClassDef = visit_FunctionDef = visit_AsyncFunctionDef = _scope

        def visit_Call(self, node):
            func, callee = node.func, None
            if (isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "deep_gemm"
                    and func.attr not in DGP.INVENTORY_NON_KERNELS):
                callee = "deep_gemm." + func.attr
            elif isinstance(func, ast.Name) and func.id in callees:
                callee = func.id
            if callee is not None:
                scope = ".".join(self.stack) or "<module>"
                found[(self.rel, scope, callee)] += 1
            self.generic_visit(node)

    for path in sorted(files):
        if path.name in ("triton_prewarm.py", "deepgemm_prewarm.py"):
            continue
        rel = path.relative_to(_ROOT).as_posix()
        Visitor(rel).visit(ast.parse(path.read_text()))
    return found


def test_inventory_names_every_deepgemm_call():
    found = _scan_calls()
    pinned = {key: count for key, (count, _) in DGP.CALL_SITES.items()}
    missing = {k: v for k, v in found.items() if k not in pinned}
    stale = {k: v for k, v in pinned.items() if k not in found}
    changed = {k: (pinned[k], found[k]) for k in pinned.keys() & found.keys()
               if pinned[k] != found[k]}
    assert missing == {}, "DeepGEMM calls the site table does not name"
    assert stale == {}, "site-table calls that no longer exist"
    assert changed == {}, "call counts changed (pinned, found)"


def test_inventory_sites_are_table_sites():
    names = DGP.table_site_names()
    referenced = set()
    for key, (_, reaches) in DGP.CALL_SITES.items():
        if isinstance(reaches, str):
            assert reaches == "wrapper" or reaches.startswith("unreachable: "), key
            continue
        assert set(reaches) <= names, key
        referenced.update(reaches)
    # Every table site has a serving call.
    assert names - referenced == set()


# --------------------------------------------------------------------------- #
#  Entries
# --------------------------------------------------------------------------- #
def _names(entries):
    return [e.name for e in entries]


def _keys_of(sites, rows):
    return {(s.n, s.k, DG.best_config(m, s.n, s.k).kernel_key())
            for s in sites for m in rows}


def test_prefill_entries_cover_every_key_of_the_row_range():
    shapes = _shapes()
    entries = DGP.prefill_entries(shapes, "cpu", max_rows=MAX_ROWS,
                                  num_sms=NUM_SMS)
    names = _names(entries)
    assert len(names) == len(set(names))
    assert names[-1] == "deepgemm/mqa_logits/h32d128"
    sites = DGP.dense_sites(shapes, DGP.PHASE_PREFILL)
    by_shape = collections.defaultdict(list)
    for name in names[:-1]:
        _, via, nk, m = name.split("/")
        n, k = (int(v) for v in nk.split("x"))
        by_shape[(via, n, k)].append(int(m[1:]))
    assert set(by_shape) == {(s.via, s.n, s.k) for s in sites}
    for site in sites:
        rows = by_shape[(site.via, site.n, site.k)]
        assert max(rows) <= MAX_ROWS
        # Brute force over the first 8192 rows of the range, one m per
        # ceil(m / 16) class (the config's only dependence on m).
        reached = {DG.best_config(m, site.n, site.k).kernel_key()
                   for m in range(16, 8193, 16)}
        assert reached <= {DG.best_config(m, site.n, site.k).kernel_key()
                           for m in rows}, site
    dense = _names(DGP.prefill_entries(_shapes(False), "cpu",
                                       max_rows=MAX_ROWS, num_sms=NUM_SMS))
    assert not [n for n in dense if "mqa_logits" in n or "/4096x2048/" in n]


def test_first_entry_of_each_predicted_key_comes_first():
    entries = DGP.decode_entries(_shapes(), "cpu", max_rows=MAX_ROWS,
                                 num_sms=NUM_SMS, merge_q_b_wq_b=True,
                                 wk_deepgemm=True)
    keys = []
    for name in _names(entries):
        _, _, nk, m = name.split("/")
        n, k = (int(v) for v in nk.split("x"))
        keys.append(DG.best_config(int(m[1:]), n, k).kernel_key())
    distinct = len(set(keys))
    assert len(set(keys[:distinct])) == distinct


def test_geometry_entries_cover_every_bucket():
    shapes = _shapes()
    entries = DGP.geometry_entries(
        shapes, "cpu", max_decode_rows=SLOTS, num_sms=NUM_SMS,
        merge_q_b_wq_b=True, page_split=True, moe_stacks=_stacks(),
        world_size=8, topk=8)
    names = _names(entries)
    assert len(names) == len(set(names))
    for site in DGP.masked_dense_sites(shapes):
        ems = [int(n.rsplit("/em", 1)[1]) for n in names
               if n.startswith(f"deepgemm/masked/{site.n}x{site.k}/g1/")]
        reached = {DG.best_config(64, site.n, site.k, masked=True,
                                  expected_m=b).kernel_key()
                   for b in range(1, SLOTS + 1)}
        assert {DG.best_config(64, site.n, site.k, masked=True,
                               expected_m=e).kernel_key()
                for e in ems} == reached
    expected = {DGP.expected_m_per_expert(local_rows=b, world_size=8, topk=8,
                                          num_experts_global=256)
                for b in range(1, SLOTS + 1)}
    for site in DGP.masked_moe_sites(shapes, 32):
        ems = [int(n.rsplit("/em", 1)[1]) for n in names
               if n.startswith(f"deepgemm/masked/{site.n}x{site.k}/g32/")]
        assert ems and set(ems) <= expected
        reached = {DG.best_config(64, site.n, site.k, groups=32, masked=True,
                                  expected_m=e).kernel_key() for e in expected}
        assert {DG.best_config(64, site.n, site.k, groups=32, masked=True,
                               expected_m=e).kernel_key()
                for e in ems} == reached
    assert [n for n in names if n.startswith("deepgemm/paged_mqa/")] == [
        f"deepgemm/paged_mqa/b{b}" for b in (32, 64, 96, 128)]


def test_geometry_entries_without_deepep_or_page_split():
    names = _names(DGP.geometry_entries(
        _shapes(), "cpu", max_decode_rows=SLOTS, num_sms=NUM_SMS,
        merge_q_b_wq_b=False, page_split=False))
    assert not [n for n in names if "/g32/" in n or "paged_mqa" in n
                or "/20480x2048/" in n]
    assert DGP.paged_mqa_batches(1) == [32]
    assert DGP.paged_mqa_batches(33) == [32, 64]
    assert DGP.paged_mqa_batches(160) == [32, 64, 96, 128, 160]


def test_entry_lists_are_deterministic_across_ranks():
    def build():
        shapes = _shapes()
        entries = DGP.prefill_entries(shapes, "cpu", max_rows=MAX_ROWS,
                                      num_sms=NUM_SMS)
        entries += DGP.decode_entries(shapes, "cpu", max_rows=MAX_ROWS,
                                      num_sms=NUM_SMS, merge_q_b_wq_b=True,
                                      wk_deepgemm=True)
        entries += DGP.geometry_entries(
            shapes, "cpu", max_decode_rows=SLOTS, num_sms=NUM_SMS,
            merge_q_b_wq_b=True, page_split=True, moe_stacks=_stacks(),
            world_size=8, topk=8)
        return _names(entries)

    first = build()
    DGP._normal_reps.cache_clear()
    assert build() == first
    shares = [TP._core.partition(first, i, 8) for i in range(8)]
    assert sorted(n for share in shares for n in share) == sorted(first)


def test_geometry_key_carries_the_slot_capacity():
    def geometry(slots):
        return TP.Glm5DecodeGeometry(
            primary_width=3169, aux_width=3169, primary_page_size=64,
            aux_page_size=64, aux_page_split=True, max_decode_rows=slots)

    assert geometry(128).key() != geometry(256).key()
    manager = SimpleNamespace(
        _gpu_page_table_manager=SimpleNamespace(max_slots=384))
    assert TP._page_table_slots(manager, "primary") == 384
    with pytest.raises(TP.Glm5PrewarmGeometryError, match="slot capacity"):
        TP._page_table_slots(SimpleNamespace(), "primary")


# --------------------------------------------------------------------------- #
#  Runners: serving layouts
# --------------------------------------------------------------------------- #
def _weight_ok(weight, scale, n, k):
    assert weight.shape == (n, k) and weight.dtype == torch.float8_e4m3fn
    assert weight.is_contiguous()
    assert scale.shape == (-(-n // 128), -(-k // 128))
    assert scale.dtype == torch.float32 and scale.is_contiguous()


def test_w8a16_runner_hands_the_wrapper_bf16_rows():
    gemm = _Recorder()
    site = DGP.DenseSite("kv_b", 28672, 512, DGP.VIA_W8A16)
    DGP._run_dense(site, 17, "cpu", fns={"gemm": gemm})
    (weight, scale, x), kwargs = gemm.calls[0]
    _weight_ok(weight, scale, 28672, 512)
    assert x.shape == (17, 512) and x.dtype == torch.bfloat16
    assert x.is_contiguous() and kwargs == {}


def test_w8a8_runner_passes_k_major_scales():
    quant, gemm = _Recorder(_fake_quant), _Recorder()
    site = DGP.DenseSite("q_a", 2048, 6144, DGP.VIA_W8A8)
    DGP._run_dense(site, 5, "cpu", fns={"quant": quant, "gemm": gemm})
    (x,), qkw = quant.calls[0]
    assert x.shape == (5, 6144) and qkw == {}
    (a, a_scale, weight, scale), kwargs = gemm.calls[0]
    assert a.dtype == torch.float8_e4m3fn and a_scale.stride() == (48, 1)
    _weight_ok(weight, scale, 2048, 6144)
    assert kwargs == {}


def test_tma_runner_passes_mn_major_scales_and_a_bf16_output():
    quant, gemm = _Recorder(_fake_quant), _Recorder()
    site = DGP.DenseSite("shared_gate_up_fused", 4096, 6144, DGP.VIA_TMA)
    DGP._run_dense(site, 13, "cpu", fns={"quant": quant, "gemm": gemm})
    assert quant.calls[0][1] == {"scale_tma_aligned": True}
    ((a, a_scale), (weight, scale), out), kwargs = gemm.calls[0]
    assert a_scale.stride() == (1, 16)          # align4(13)
    _weight_ok(weight, scale, 4096, 6144)
    assert out.shape == (13, 4096) and out.dtype == torch.bfloat16
    assert out.is_contiguous() and kwargs == {}


def test_wk_runner_is_the_graph_call():
    quant, gemm, wrap = _Recorder(_fake_quant), _Recorder(), _Recorder(
        lambda w, s: ("weights", w, s))
    site = DGP.DenseSite("indexer_wk_graph", 128, 6144, DGP.VIA_WK)
    DGP._run_dense(site, 3, "cpu",
                   fns={"quant": quant, "gemm": gemm, "wrap": wrap})
    (x,), qkw = quant.calls[0]
    assert qkw["scale_tma_aligned"] is True
    assert qkw["num_valid_tokens"].dtype == torch.int32
    assert int(qkw["num_valid_tokens"]) == 3
    (a, a_scale, weights, out), _ = gemm.calls[0]
    _weight_ok(weights[1], weights[2], 128, 6144)
    assert out.shape == (3, 128) and out.dtype == torch.bfloat16


def test_masked_dense_runner_is_the_decode_graph_call():
    quant, gemm = _Recorder(_fake_quant), _Recorder()
    site = DGP.MaskedSite("qkv_a", 2624, 6144, 1)
    DGP._run_masked_dense(site, 65, "cpu", fns={"quant": quant, "gemm": gemm})
    (x,), qkw = quant.calls[0]
    assert x.shape == (65, 6144) and qkw["scale_tma_aligned"] is True
    assert int(qkw["num_valid_tokens"]) == 65
    (a, a_scale, weight, scale), kwargs = gemm.calls[0]
    assert a_scale.stride() == (1, 68)
    _weight_ok(weight, scale, 2624, 6144)
    assert kwargs["expected_m"] == 65 and int(kwargs["num_valid_tokens"]) == 65
    assert kwargs["out"].shape == (65, 2624)
    assert kwargs["out"].dtype == torch.bfloat16


def test_masked_moe_runner_matches_the_deepep_layout():
    fn = _Recorder()
    weight = torch.zeros((4, 4096, 6144), dtype=torch.float8_e4m3fn)
    scale = torch.ones((4, 32, 48))
    DGP._run_masked_moe(weight, scale, 9, "cpu", fn=fn)
    ((a, a_scale), (w, s), out, masked_m, expected_m), kwargs = fn.calls[0]
    rows = DGP.MOE_MASKED_ROWS
    assert w is weight and s is scale
    assert a.shape == (4, rows, 6144) and a.dtype == torch.float8_e4m3fn
    assert a_scale.shape == (4, rows, 48)
    assert a_scale.stride() == (48 * rows, 1, rows)
    assert out.shape == (4, rows, 4096) and out.dtype == torch.bfloat16
    assert masked_m.dtype == torch.int32 and int(masked_m.abs().sum()) == 0
    assert expected_m == 9 and kwargs == {}


def test_paged_mqa_runner_uses_a_page_split_scratch_cache():
    fn = _Recorder()
    DGP._run_paged_mqa(96, 32, 128, 64, "cpu", fn=fn)
    _, kwargs = fn.calls[0]
    cache = kwargs.pop("aux_blocked_k")
    assert cache.shape == (1, 64, 1, 132) and cache.dtype == torch.uint8
    assert kwargs == {"batch": 96, "num_heads": 32, "index_head_dim": 128,
                      "max_pages_per_seq": 1, "max_seqlen": 64,
                      "page_size": 64}


def test_mqa_logits_runner_is_the_sparse_prefill_call():
    fn = _Recorder()
    DGP._run_mqa_logits(32, 128, "cpu", fn=fn)
    (q, (k, k_scale), weights, starts, ends), kwargs = fn.calls[0]
    assert q.shape == (DGP.MQA_ROWS, 32, 128) and q.dtype == torch.float8_e4m3fn
    assert k.shape == (DGP.MQA_SCORE_WIDTH, 128)
    assert k_scale.shape == (DGP.MQA_SCORE_WIDTH,)
    assert weights.shape == (DGP.MQA_ROWS, 32)
    assert weights.dtype == torch.float32
    assert starts.dtype == ends.dtype == torch.int32
    assert kwargs == {"max_seqlen_k": DGP.MQA_SCORE_WIDTH,
                      "clean_logits": False}
    assert DGP.MQA_SCORE_WIDTH % 256 == 0 and DGP.MQA_ROWS % 4 == 0


def test_moe_stacks_reads_the_fused_layout_only():
    stack = torch.zeros((2, 8, 4), dtype=torch.uint8)
    fused = SimpleNamespace(
        fp8_gate_up_fused=True, fp8_gate_up_w3d=stack,
        fp8_gate_up_ws3d="gu_scale", fp8_down_w3d=stack,
        fp8_down_ws3d="down_scale")
    model = SimpleNamespace(model=SimpleNamespace(layers=[
        SimpleNamespace(mlp=SimpleNamespace()),
        SimpleNamespace(mlp=fused),
    ]))
    stacks = DGP.moe_stacks(model)
    assert stacks.gate_up.dtype == torch.float8_e4m3fn
    assert stacks.gate_up_scale == "gu_scale"
    fused.fp8_gate_up_fused = False
    assert DGP.moe_stacks(model) is None


# --------------------------------------------------------------------------- #
#  Version gate and late-compile count
# --------------------------------------------------------------------------- #
def test_version_mismatch_warns_with_both_versions(monkeypatch, caplog):
    monkeypatch.setattr(DGP, "_VERSION_CHECKED", False)
    monkeypatch.setattr(DG, "installed_version", lambda: "0.2.0")
    with caplog.at_level(logging.WARNING, logger=DGP.logger.name):
        DGP.check_version(rank=0)
        DGP.check_version(rank=0)
    warned = [r.getMessage() for r in caplog.records]
    assert len(warned) == 1
    assert "0.2.0" in warned[0] and DG.SUPPORTED_DEEPGEMM_VERSION in warned[0]
    caplog.clear()
    monkeypatch.setattr(DGP, "_VERSION_CHECKED", False)
    monkeypatch.setattr(DG, "installed_version",
                        lambda: DG.SUPPORTED_DEEPGEMM_VERSION)
    with caplog.at_level(logging.WARNING, logger=DGP.logger.name):
        DGP.check_version(rank=0)
    assert caplog.records == []


def test_late_compiles_are_counted_logged_and_strict(monkeypatch, tmp_path,
                                                     caplog):
    monkeypatch.setenv("DG_JIT_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(DGP, "_BASELINE", None)
    cache = tmp_path / "cache"
    assert DGP.jit_cache_dir() == cache
    assert DGP.cache_kernel_count() == 0
    # Before the first geometry phase there is nothing to compare against.
    (cache / "kernel.sm90_fp8_gemm_1d2d.a").mkdir(parents=True)
    assert DGP.check_late_compiles(0, "configure_prefill") == 0
    (cache / "tmp.partial").mkdir()
    (cache / "kernel.stray_file").write_text("")
    assert DGP.record_baseline(0) == 1
    assert DGP.check_late_compiles(0, "configure_prefill") == 0
    (cache / "kernel.sm90_fp8_gemm_1d2d.b").mkdir()
    (cache / "kernel.transpose_fp32.c").mkdir()
    with caplog.at_level(logging.WARNING, logger=DGP.logger.name):
        assert DGP.check_late_compiles(3, "configure_decoding") == 2
    message, = [r.getMessage() for r in caplog.records]
    assert message.startswith("[GLM5_DEEPGEMM_WARMUP] rank=3 DeepGEMM "
                              "compiled 2 kernels after the prewarm")
    assert "configure_decoding" in message
    # Reported once: the baseline moves to the new count.
    assert DGP.check_late_compiles(3, "configure_prefill") == 0
    (cache / "kernel.sm90_paged_mqa_logits_metadata.d").mkdir()
    with pytest.raises(DGP.LateDeepGemmCompileError, match="compiled 1"):
        DGP.check_late_compiles(0, "configure_prefill", strict=True)


def test_imports_without_deepgemm_or_triton(monkeypatch):
    for name in ("deep_gemm", "triton", "triton.language"):
        monkeypatch.setitem(sys.modules, name, None)
    module = _load(_GLM5_DIR / "deepgemm_prewarm.py",
                   "_glm5_deepgemm_prewarm_no_gpu_deps")
    assert callable(module.geometry_entries)
