"""TP-sharded latent projections for the resident-EP MoE (DECODE_CONCURRENCY_PLAN.md).

Each rank of a TP group applies its column slice of ``routed_expert_down_proj``
to ALL of the group's rows in rank-padded order, so the EP all_gather of those
``[G*ntp, cols]`` slices assembles into the rank-major ``[world*ntp, K_latent]``
global latent the dispatch consumes today. The helpers must reproduce, bit for
bit, the unsharded gather of ``down_proj(local_rows)`` per rank.

The module imports the marlin extension, so the helpers are exec'd in
isolation together with ``balanced_row_split``.
"""
import ast
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "batchgen" / "moe" / "fused_moe_mxfp4_resident.py"
RESHARD = ROOT / "batchgen" / "models" / "moonshotai" / "kimi_linear" / "moe_tp_reshard.py"


def _load():
    ns = {"torch": torch, "List": list, "Tuple": tuple}
    tree = ast.parse(RESHARD.read_text())
    for name in ("balanced_row_split", "scatter_rows"):
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module([fn], []), str(RESHARD), "exec"), ns)
    tree = ast.parse(SRC.read_text())
    for name in ("rank_padded_row_map", "pad_rows_rank_major", "unpad_rows_rank_major",
                 "assemble_gathered_latent_slices"):
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        src = ast.get_source_segment(SRC.read_text(), fn)
        # the helper imports balanced_row_split from the reshard module; bind the
        # exec'd copy instead of importing the package
        src = src.replace(
            "from batchgen.models.moonshotai.kimi_linear.moe_tp_reshard import balanced_row_split",
            "balanced_row_split = _brs")
        ns["_brs"] = ns["balanced_row_split"]
        exec(src, ns)
    return ns


@pytest.mark.parametrize("rows_per_node", [(5, 5), (7, 3), (8, 8), (1, 6)])
def test_sharded_gather_assembles_the_unsharded_global_latent(rows_per_node):
    ns = _load()
    G, nodes, H, K = 4, 2, 12, 8
    cols = K // G
    world = G * nodes
    ntp = max(-(-r // G) for r in rows_per_node)  # ceil(max rows / G)
    torch.manual_seed(0)
    W = torch.randn(K, H)                       # full down_proj weight [K, H]
    x_nodes = [torch.randn(r, H) for r in rows_per_node]

    # reference: today's per-rank contribution down_proj(local rows) padded to ntp
    ref = torch.zeros(world * ntp, K)
    for n in range(nodes):
        for c in range(G):
            local = ns["scatter_rows"](x_nodes[n], G, c)
            r = n * G + c
            ref[r * ntp : r * ntp + local.shape[0]] = local @ W.t()

    # sharded: rank (n, c) applies W[c*cols:(c+1)*cols] to ALL group rows in rank-padded order
    contributions = []
    for n in range(nodes):
        p2g, g2p = ns["rank_padded_row_map"](x_nodes[n].shape[0], G, ntp, torch.device("cpu"))
        x_pad = ns["pad_rows_rank_major"](x_nodes[n], p2g)
        assert x_pad.shape == (G * ntp, H)
        assert torch.equal(ns["unpad_rows_rank_major"](x_pad, g2p), x_nodes[n])
        for c in range(G):
            contributions.append(x_pad @ W[c * cols:(c + 1) * cols].t())
    gathered = torch.cat(contributions, dim=0)   # ncclAllGather: rank-major
    assembled = ns["assemble_gathered_latent_slices"](gathered, world, G, ntp, cols)
    assert assembled.shape == ref.shape
    assert torch.allclose(assembled, ref, atol=1e-5, rtol=1e-5)


def test_row_map_rejects_rows_beyond_ntp():
    ns = _load()
    with pytest.raises(ValueError):
        ns["rank_padded_row_map"](9, 2, 4, torch.device("cpu"))


def test_forward_ep_sharded_contract_is_static():
    src = SRC.read_text()
    body = src[src.index("def _forward_ep("):src.index("def build_resident_ep_mxfp4_layers(")]
    assert "assemble_gathered_latent_slices(gathered, world, G, ntp, cols)" in body
    assert "dist.all_gather_into_tensor(gathered_rows, y_rows, group=self.latent_tp_group)" in body
    # the norm is applied to this rank's own full-latent rows before the node gather
    assert body.index("self.norm(y_rows[:T])") < body.index("dist.all_gather_into_tensor(gathered_rows")
    assert "unpad_rows_rank_major(part, group_to_padded)" in body


def test_graph_buffer_views_only_pass_declared_fields():
    """The per-bucket view constructor must name only dataclass fields; the
    first sharded run died at graph capture with an unexpected `latent_slice`
    keyword (the pool grew the buffers, the view type did not)."""
    seg = ROOT / "batchgen" / "models" / "moonshotai" / "kimi_linear" / "moe_cuda_graph_segments.py"
    tree = ast.parse(seg.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_K3MoEGraphBuffers")
    fields = {n.target.id for n in cls.body if isinstance(n, ast.AnnAssign)}
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_K3MoEGraphBuffers"
    ]
    assert calls
    for call in calls:
        passed = {kw.arg for kw in call.keywords}
        assert passed == fields, (passed - fields, fields - passed)
