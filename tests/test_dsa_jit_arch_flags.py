import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = (
    ROOT
    / "batchgen_kernels"
    / "attention"
    / "dsa"
    / "fused_indexer_kv_proj_cuda.py"
)


def _load_build_module(captured_flags):
    tree = ast.parse(BUILDER.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_module"
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[function], type_ignores=[])
    )

    def load_inline(**kwargs):
        captured_flags.extend(kwargs["extra_cuda_cflags"])
        return object()

    namespace = {
        "_module_cache": {},
        "os": type("FakeOS", (), {"environ": {}}),
        "CPP_SOURCE": "",
        "CUDA_SOURCE": "",
        "load_inline": load_inline,
    }
    exec(compile(module, str(BUILDER), "exec"), namespace)
    return namespace["build_module"]


def test_indexer_wgmma_jit_builds_only_sm90a_cubin():
    flags = []
    _load_build_module(flags)()

    assert "-arch=sm_90a" not in flags
    assert flags[flags.index("-gencode") + 1] == (
        "arch=compute_90a,code=sm_90a"
    )
