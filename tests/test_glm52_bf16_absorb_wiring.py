"""Slice 2c-2, commit 2: the model side is wired to the BF16 absorb container.

Structural (AST / source) assertions only, so they run on a CPU-only dev
machine: importing these GLM-5 modules needs Triton.

What must hold after commit 2:
  * exactly ONE container is built, in both construction sites -- if the two
    diverged, the eager DSA route and the graph route would run different
    numerics for the same layer;
  * no FP8 absorb container is constructed anywhere any more, so the HBM
    accounting (+14.68 MB/layer net) is real rather than additive;
  * every absorb call site goes through the dispatch;
  * the Triton kernel and its container still EXIST (repo rule: kernel
    versions are superseded, never deleted);
  * the core scaffolding is untouched -- the attribute keeps its historical
    `_fp8_absorb_weights` name.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
GLM5 = ROOT / "batchgen/models/glm/glm5"
ATTN = ROOT / "batchgen/attention"
KERNELS = ROOT / "batchgen_kernels/attention/dsa"


def _calls(path: pathlib.Path) -> set[str]:
    out = set()
    for n in ast.walk(ast.parse(path.read_text())):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name):
                out.add(f.id)
            elif isinstance(f, ast.Attribute):
                out.add(f.attr)
    return out


ABSORB_SITES = [
    GLM5 / "cuda_graph_segments.py",
    GLM5 / "reuse_topk_segment.py",
    GLM5 / "wrappers.py",
    ATTN / "dsa/glm5_decode_selector.py",
    ATTN / "mla/flashmla_backend.py",
]


@pytest.mark.parametrize("path", ABSORB_SITES, ids=lambda p: p.name)
def test_no_site_calls_the_fp8_entry_points_directly(path):
    """All five sites must dispatch, or the two decode routes diverge."""
    banned = {"fp8_q_absorb", "fp8_q_absorb_out",
              "fp8_out_absorb", "fp8_out_absorb_out"}
    leaked = banned & _calls(path)
    assert not leaked, f"{path.name} still calls {sorted(leaked)} directly"


def test_only_the_bf16_container_is_constructed():
    """FP8AbsorbWeights must no longer be built -- that is the HBM premise."""
    for path in ABSORB_SITES:
        calls = _calls(path)
        assert "FP8AbsorbWeights" not in calls, (
            f"{path.name} still constructs FP8AbsorbWeights; the net "
            "+14.68 MB/layer accounting assumes the fp8 copies are gone"
        )
    built = [p.name for p in ABSORB_SITES if "BF16AbsorbWeights" in _calls(p)]
    assert sorted(built) == ["flashmla_backend.py", "wrappers.py"], (
        f"expected exactly the two known construction sites, found {built}"
    )


def test_no_runtime_container_switch_exists():
    """A selectable container would let the two routes build different ones."""
    src = (ATTN / "mla/flashmla_backend.py").read_text()
    assert "_ABSORB_CONTAINER" not in src, (
        "a runtime container switch is a divergence foot-gun: the eager DSA "
        "route and the graph route must build the same container"
    )


def test_both_construction_sites_agree_on_the_container():
    """The anti-divergence invariant, stated as a test.

    `flashmla_backend` selects via _ABSORB_CONTAINER; `wrappers` builds bf16
    unconditionally. They must not be able to disagree.
    """
    fb = (ATTN / "mla/flashmla_backend.py").read_text()
    wr = (GLM5 / "wrappers.py").read_text()
    assert "BF16AbsorbWeights(" in fb and "FP8AbsorbWeights(" not in fb
    assert "BF16AbsorbWeights(" in wr and "FP8AbsorbWeights(" not in wr


def test_triton_kernel_and_fp8_container_still_exist():
    """Superseded, not deleted."""
    names = {n.name for n in ast.walk(ast.parse((KERNELS / "fp8_absorb.py").read_text()))
             if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    for kept in ("_absorb_fp8_wgmma_kernel", "FP8AbsorbWeights",
                 "fp8_q_absorb_out", "fp8_out_absorb_out"):
        assert kept in names, f"{kept} must survive as the superseded path"


def test_core_scaffolding_attribute_name_is_preserved():
    """Renaming the attribute would make this a core-class change."""
    for path in (GLM5 / "wrappers.py", ATTN / "dsa/glm5_decode_selector.py",
                 ATTN / "mla/flashmla_backend.py"):
        assert "_fp8_absorb_weights" in path.read_text(), (
            f"{path.name} must keep the historical attribute name that "
            "batchgen_worker.py reads"
        )


def test_reuse_topk_records_why_num_valid_tokens_is_ignored():
    """The one semantic change must be argued at its call sites, not implied."""
    src = (GLM5 / "reuse_topk_segment.py").read_text()
    assert "absorb_q_out(" in src and "absorb_out_out(" in src
    assert "IGNORES it" in src or "ignored by the BF16 path" in src, (
        "the call sites must state that the BF16 container ignores "
        "num_valid_tokens and why"
    )
    assert "act_quant" in src


def test_hbm_cost_is_documented_at_the_construction_site():
    """+1.15 GB/GPU is the open trade; it must be visible where it is incurred."""
    src = (GLM5 / "wrappers.py").read_text()
    assert "+14.68 MB/layer" in src and "1.15 GB" in src
    assert "KV cache" in src
