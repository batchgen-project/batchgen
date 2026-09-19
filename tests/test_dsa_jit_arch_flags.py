"""Static guard on the DSA indexer JIT arch flags (no CUDA, no torch import).

nvcc's `-arch=sm_90a` shorthand made the cold H200 build emit compute_90 PTX,
which ptxas then rejected for the `wgmma.*` instructions. Only the explicit
two-item `-gencode arch=compute_90a,code=sm_90a` form keeps the virtual arch
at 90a, so pin it here.
"""

import ast
import types
from pathlib import Path

_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "batchgen_kernels"
    / "attention"
    / "dsa"
    / "fused_indexer_kv_proj_cuda.py"
)


def _capture_extra_cuda_cflags():
    """Run build_module() alone, with load_inline stubbed, and return its flags."""
    tree = ast.parse(_SOURCE.read_text())
    build_module = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_module"
    )

    captured = {}

    def fake_load_inline(**kwargs):
        captured.update(kwargs)
        return object()

    namespace = {
        "os": types.SimpleNamespace(environ={}),
        "load_inline": fake_load_inline,
        "_module_cache": {},
        "CPP_SOURCE": "",
        "CUDA_SOURCE": "",
        "print": lambda *args, **kwargs: None,
    }
    exec(
        compile(ast.Module(body=[build_module], type_ignores=[]), str(_SOURCE), "exec"),
        namespace,
    )
    namespace["build_module"]()
    return captured["extra_cuda_cflags"]


def test_build_module_uses_explicit_sm90a_gencode():
    flags = _capture_extra_cuda_cflags()

    assert "-arch=sm_90a" not in flags, f"shorthand arch flag still present: {flags}"
    assert flags.count("-gencode") == 1, f"expected exactly one -gencode: {flags}"
    assert flags[flags.index("-gencode") + 1] == "arch=compute_90a,code=sm_90a", (
        f"unexpected -gencode argument: {flags}"
    )
