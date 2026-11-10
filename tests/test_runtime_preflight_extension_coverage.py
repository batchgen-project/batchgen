"""Every AOT extension a model's code can load is in its preflight contract.

A partial ``batchgen_kernels`` build used to pass the runtime preflight
because each contract listed only a hand-picked subset of the extensions its
generation path imports; the missing one surfaced minutes into decode.  This
test derives the set statically from the model's own package (imports and
``batchgen_kernels.<pkg>._C_<name>`` string literals, followed transitively
through shared ``batchgen``/``batchgen_kernels`` modules) and requires the
contract to cover it, so adding a new extension use to a model path without
declaring it fails CI instead of failing a deployment.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

import batchgen.runtime_preflight as preflight

REPO = Path(__file__).resolve().parents[1]
_EXT = re.compile(r"^batchgen_kernels\.\w+\._C_\w+$")

# The model packages that implement each contract's generation path.
_MODEL_PACKAGES = {
    "gpt_oss": ("batchgen.models.openai.gpt_oss_120b",),
    "glm_moe_dsa": ("batchgen.models.glm.glm5",),
    "glm_moe_dsa_5_2": ("batchgen.models.glm.glm5",),
    "glm_moe_dsa_5_3": ("batchgen.models.glm.glm5",),
    "deepseek_v2": ("batchgen.models.deepseek.deepseekv2",),
    "deepseek_v3": ("batchgen.models.deepseek.deepseekv3",),
    "deepseek_v4": (
        "batchgen.models.deepseek.deepseekv4_flash",
        "batchgen.models.deepseek.deepseekv4_pro",
    ),
    "mixtral": ("batchgen.models.mixtral",),
    "minimax_m25": ("batchgen.models.minimax.minimax_m25",),
    "kimi_linear": (
        "batchgen.models.moonshotai.kimi_linear",
        "batchgen.models.moonshotai.kimi_k3",
    ),
    "kimi_k3": (
        "batchgen.models.moonshotai.kimi_linear",
        "batchgen.models.moonshotai.kimi_k3",
    ),
    "kimi_k31": (
        "batchgen.models.moonshotai.kimi_linear",
        "batchgen.models.moonshotai.kimi_k3",
    ),
    "kimi_k25": ("batchgen.models.moonshotai.kimi_k25",),
}

# Extensions the package's code can reach that the model type's serving path
# provably never loads. Kimi-Linear, Kimi-K3 and Kimi-K3.1 share one package;
# the DSA indexer kernels run only for Kimi-K3.1, whose route-B path loads
# exactly dsa_runtime.K31_DSA_AOT_EXTENSIONS (held to the code by
# tests/test_kimi_k31_dsa_runtime.py); the rest are GLM-5 indexer kernels in
# the shared indexer package. Each entry must stay reachable (checked below),
# so a stale exclusion fails.
_GLM5_ONLY_INDEXER = {
    "batchgen_kernels.attention._C_fast_hadamard_transform",
    "batchgen_kernels.attention._C_fused_rope_hadamard",
    "batchgen_kernels.attention._C_indexer_kv_proj_wgmma",
    "batchgen_kernels.attention._C_fused_indexer_k_write_fp8",
}
_K31_ONLY_INDEXER = {
    "batchgen_kernels.attention._C_fused_indexer_k_norm_rope_write_fp8",
    "batchgen_kernels.attention._C_dsa_fast_topk",
}
_OFF_PATH = {
    "kimi_linear": _GLM5_ONLY_INDEXER | _K31_ONLY_INDEXER,
    "kimi_k3": _GLM5_ONLY_INDEXER | _K31_ONLY_INDEXER,
    "kimi_k31": _GLM5_ONLY_INDEXER,
}

# Shared model-layer modules a model package may import; every other
# ``batchgen.models.*`` package belongs to a different contract.
_SHARED_MODEL_MODULES = (
    "batchgen.models.wrappers",
    "batchgen.models.base_embedding",
    "batchgen.models.weight_reconciler",
)

# Not followed: the dev-mode JIT table names every extension, and the worker,
# server and converter are shared by all models.
_NOT_FOLLOWED = (
    "batchgen_kernels._jit_registry",
    "batchgen.batchgen_worker",
    "batchgen.server",
    "batchgen.launch_http_server",
    "batchgen.ckpt_converter",
)


def _module_index() -> dict[str, Path]:
    index = {}
    for top in ("batchgen", "batchgen_kernels"):
        for path in (REPO / top).rglob("*.py"):
            rel = path.relative_to(REPO)
            if "3rd" in rel.parts or rel.name == "setup.py":
                continue
            parts = list(rel.with_suffix("").parts)
            if parts[-1] == "__init__":
                parts = parts[:-1]
            index[".".join(parts)] = path
    return index


def _under(name: str, prefixes) -> bool:
    return any(name == p or name.startswith(p + ".") for p in prefixes)


def _scan(name: str, path: Path, index) -> tuple[set[str], set[str]]:
    tree = ast.parse(path.read_text())
    package = name if path.name == "__init__.py" else name.rsplit(".", 1)[0]
    imported, extensions = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")[: len(package.split(".")) - node.level + 1]
                module = ".".join(base + ([node.module] if node.module else []))
            else:
                module = node.module or ""
            imported.add(module)
            imported.update(f"{module}.{alias.name}" for alias in node.names)
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and _EXT.match(node.value)
        ):
            extensions.add(node.value)
    extensions.update(name for name in imported if _EXT.match(name))
    return {name for name in imported if name in index}, extensions


def _reachable_extensions(packages, index) -> set[str]:
    todo = [m for m in index if _under(m, packages)]
    seen, found = set(), set()
    while todo:
        module = todo.pop()
        if module in seen:
            continue
        seen.add(module)
        edges, extensions = _scan(module, index[module], index)
        found |= extensions
        for nxt in edges:
            if _under(nxt, _NOT_FOLLOWED):
                continue
            if _under(nxt, ("batchgen.models",)) and not (
                _under(nxt, packages) or _under(nxt, _SHARED_MODEL_MODULES)
            ):
                continue
            todo.append(nxt)
    return found


def _built_extensions() -> tuple[set[str], set[str]]:
    """(sm90a-only, every-arch) extension names from batchgen_kernels/setup.py."""

    tree = ast.parse((REPO / "batchgen_kernels" / "setup.py").read_text())
    lists = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in {"_sm90a_extensions", "_sm80_extensions"}
        ):
            names = set()
            for call in ast.walk(node.value):
                if isinstance(call, ast.Call):
                    for kw in call.keywords:
                        if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                            names.add(kw.value.value)
            lists[node.targets[0].id] = names
    # The attention decode kernel ships prebuilt for sm_90a as package data.
    sm90a = lists["_sm90a_extensions"] | {
        "batchgen_kernels.attention._C_gqa_mha_decode_bf16"
    }
    return sm90a, lists["_sm80_extensions"]


@pytest.fixture(scope="module")
def index():
    return _module_index()


def test_every_contract_has_a_model_package():
    assert set(_MODEL_PACKAGES) == set(preflight._CONTRACTS)


@pytest.mark.parametrize("model_type", sorted(preflight._CONTRACTS))
def test_contract_covers_every_reachable_extension(model_type, index):
    contract = preflight._CONTRACTS[model_type]
    reachable = _reachable_extensions(_MODEL_PACKAGES[model_type], index)
    declared = set(contract.extensions) | set(contract.hopper_extensions)
    off_path = _OFF_PATH.get(model_type, set())
    assert off_path <= reachable, sorted(off_path - reachable)
    assert not off_path & declared, sorted(off_path & declared)
    missing = sorted(reachable - declared - off_path)
    assert not missing, (
        f"{model_type}: extensions its model code loads are absent from the "
        f"runtime contract, so a partial kernel build would pass preflight: {missing}"
    )


@pytest.mark.parametrize("model_type", sorted(preflight._CONTRACTS))
def test_hopper_only_extensions_are_sm90a_builds(model_type, index):
    sm90a, every_arch = _built_extensions()
    contract = preflight._CONTRACTS[model_type]
    declared = set(contract.extensions) | set(contract.hopper_extensions)
    assert declared <= sm90a | every_arch, sorted(declared - sm90a - every_arch)
    # Conditioning an every-arch extension on the device would let a partial
    # build pass on a non-Hopper GPU.
    assert not set(contract.hopper_extensions) & every_arch, sorted(
        set(contract.hopper_extensions) & every_arch
    )


def _run_with_device(monkeypatch, model_type, capability):
    imported = []
    site = Path("/opt/conda/lib/python3.11/site-packages")
    monkeypatch.setattr(preflight, "_check_torch", lambda: None)
    monkeypatch.setattr(preflight, "_resolve_model_type", lambda _: model_type)
    monkeypatch.setattr(preflight, "_check_tokenizer", lambda _: None)
    monkeypatch.setattr(preflight, "_check_core_engine", lambda: None)
    monkeypatch.setattr(preflight, "_check_deepgemm", lambda: None)
    monkeypatch.setattr(preflight, "_check_ucx", lambda _: None)
    monkeypatch.setattr(preflight, "_batchgen_root", lambda: site)
    monkeypatch.setattr(preflight.torch.cuda, "is_available", lambda: capability is not None)
    monkeypatch.setattr(preflight.torch.cuda, "get_device_capability", lambda *a: capability)
    monkeypatch.setattr(preflight.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        preflight.torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(multi_processor_count=132),
    )

    def fake_import(name, *, origin=None):
        imported.append(name)
        return SimpleNamespace(__file__=str(site / "m.so"), attention_decode_bf16=lambda: None)

    monkeypatch.setattr(preflight, "_import_required", fake_import)
    preflight.run_runtime_preflight(SimpleNamespace(model="m"))
    return imported


def test_hopper_device_imports_hopper_extensions(monkeypatch):
    imported = _run_with_device(monkeypatch, "gpt_oss", (9, 0))
    assert "batchgen_kernels.moe._C_expert_mxfp4_wgmma" in imported
    assert "batchgen_kernels.moe._C_mxfp4_dequant_cute" in imported


def test_non_hopper_device_skips_only_hopper_extensions(monkeypatch):
    imported = _run_with_device(monkeypatch, "gpt_oss", (10, 0))
    assert "batchgen_kernels.moe._C_expert_mxfp4_wgmma" not in imported
    assert "batchgen_kernels.moe._C_mxfp4_dequant_cute" in imported


def test_missing_moe_extension_fails_preflight(monkeypatch):
    """The GLM-5.3 MoE GEMM extension missing from a Hopper build is refused."""

    def broken(name, *, origin=None):
        if name == "batchgen_kernels.moe._C_fp8_blockwise_gemm":
            raise preflight.RuntimePreflightError(f"required runtime module {name!r} failed")
        return SimpleNamespace(__file__="/x.so", attention_decode_bf16=lambda: None)

    _run_with_device(monkeypatch, "glm_moe_dsa_5_3", (9, 0))
    monkeypatch.setattr(preflight, "_import_required", broken)
    with pytest.raises(preflight.RuntimePreflightError, match="_C_fp8_blockwise_gemm"):
        preflight.run_runtime_preflight(SimpleNamespace(model="zai-org/GLM-5.3-FP8"))
