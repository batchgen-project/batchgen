"""GLM-5 releases with the same geometry take the same code path everywhere.

Under test: ``batchgen/models/glm/glm5/releases.py``, every table generated
from it, and the literal copies outside the model layer it pins. From #383 until 06f164ad the host-KV indexer table, a hand-maintained
alias list, sent GLM-5.3 to the GLM-5/5.1 78-layer bf16 indexer profile while
every other table treated it as GLM-5.2. The checks below make that class of
drift unrepresentable:

  * pairwise: for every same-geometry release pair and identifier spelling,
    every routing decision (DECISIONS) agrees;
  * isomorphism: each consumer table's keys for one release, with the minor
    version substituted, are exactly the other release's keys;
  * literal scan: no file outside an explicit allowlist spells a GLM-5.x
    identifier as a string literal, so a new hand-listed alias fails here;
  * guarded copies: the five allowlisted literal tables that may not import
    the registry equal what it generates;
  * legacy snapshot: every generated table equals what main listed by hand;
  * real-config anchor: each released checkpoint config.json selects the
    indexer layer set its release's host-KV profile stores;
  * registry integrity, and a torch-free import of the release table.

CPU only: no GPU, Triton or network. Run:
    python -m pytest tests/unit/test_glm5_family_routing_equivalence.py -q
"""

from __future__ import annotations

import ast
import contextlib
import copy
import dataclasses
import importlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import tokenize
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_FIXTURES = _ROOT / "tests" / "fixtures" / "hf_configs"


# --------------------------------------------------------------------------- #
#  Loading the consumers without GPU / native dependencies
# --------------------------------------------------------------------------- #


@contextlib.contextmanager
def _temporary_modules(modules):
    """Install ``modules`` in sys.modules for the block, then restore."""
    saved = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def _load_file(name, relpath, stubs=None):
    """Exec one source file under a private module name."""
    spec = importlib.util.spec_from_file_location(name, _ROOT / relpath)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # The private name is registered only while the file executes (dataclasses
    # resolve their module through sys.modules), then removed with the stubs.
    with _temporary_modules({**(stubs or {}), name: module}):
        spec.loader.exec_module(module)
    return module


def _real_module(fqname, relpath):
    """The real ``fqname``; a private copy when another test left a stub.

    Some CPU tests register torch-free stand-ins (no ``__file__``) for
    batchgen packages and leave them installed. A stub cannot answer for the
    real table, so load the source file privately in that case.
    """
    module = sys.modules.get(fqname)
    if module is None:
        module = importlib.import_module(fqname)
    if getattr(module, "__file__", None) and Path(module.__file__).resolve() == (
        _ROOT / relpath
    ).resolve():
        return module
    package, _, leaf = fqname.rpartition(".")
    return _load_file(f"{package}._glm5_routing_{leaf}", relpath)


def _host_kv_stubs():
    gpu = types.ModuleType("batchgen.kv_cache.gpu_paged_kv_manager")
    gpu.GPUPagedKVConfig = object
    native = types.ModuleType("batchgen.native_loader")
    native.core_engine = SimpleNamespace()
    return {
        "batchgen.kv_cache.gpu_paged_kv_manager": gpu,
        "batchgen.native_loader": native,
    }


def _triton_stubs():
    """Import-time stand-in for Triton, used only when Triton is absent."""
    try:
        importlib.import_module("triton.language")
        return {}
    except ImportError:
        pass
    triton = types.ModuleType("triton")
    language = types.ModuleType("triton.language")
    language.constexpr = object
    triton.language = language
    triton.jit = lambda fn=None, **_: fn if fn is not None else (lambda f: f)
    return {"triton": triton, "triton.language": language}


FAMILY = _real_module(
    "batchgen.models.glm.glm5.releases", "batchgen/models/glm/glm5/releases.py"
)
MR = _real_module("batchgen.config.model_registry", "batchgen/config/model_registry.py")
TR = _real_module(
    "batchgen.config.tokenizer_registry", "batchgen/config/tokenizer_registry.py"
)
BMC = _real_module(
    "batchgen.config.batchgen_model_config", "batchgen/config/batchgen_model_config.py"
)
RP = _real_module("batchgen.runtime_preflight", "batchgen/runtime_preflight.py")
HK = _load_file(
    "_glm5_routing_host_kv", "batchgen/kv_cache/host_kv_mananger_config.py",
    stubs=_host_kv_stubs(),
)
CG = _load_file(
    "_glm5_routing_cuda_graph_policy", "batchgen/models/glm/glm5/cuda_graph_policy.py"
)
PU = _load_file("_glm5_routing_process_utils", "batchgen/server/process_utils.py")
PR = _load_file("_glm5_routing_prefix_reuse", "batchgen/prefix_reuse/config.py")
SP = _load_file(
    "_glm5_routing_sparse_prefill", "batchgen/models/glm/glm5/sparse_prefill.py",
    stubs=_triton_stubs(),
)
TP = _load_file(
    "_glm5_routing_triton_prewarm", "batchgen/models/glm/glm5/triton_prewarm.py"
)
SCHEDULE = _load_file(
    "_glm5_routing_dsa_schedule", "batchgen/models/glm/glm5/dsa_schedule.py"
)




def _recording_registry(fqname, decorator, table):
    """A registry stand-in that records what a source file registers."""
    module = types.ModuleType(fqname)
    setattr(module, table, {})

    def register(key):
        def deco(cls):
            getattr(module, table)[key] = cls
            return cls
        return deco

    setattr(module, decorator, register)
    return module


# The GLM config and tokenizer files, each exec'd against a registry this test
# owns, so what they register is read directly rather than through whichever
# registry instance an earlier test left in sys.modules.
_CONFIG_REGISTRY = _recording_registry(
    "batchgen.config.model_registry", "register_config", "CONFIG_REGISTRY",
)
CFG = _load_file(
    "batchgen.models.glm.glm5._glm5_routing_config",
    "batchgen/models/glm/glm5/config.py",
    stubs={"batchgen.config.model_registry": _CONFIG_REGISTRY},
)
_TOKENIZER_REGISTRY = _recording_registry(
    "batchgen.config.tokenizer_registry", "register_tokenizer", "TOKENIZER_REGISTRY",
)
_load_file(
    "batchgen.models.glm.glm5._glm5_routing_tokenizer",
    "batchgen/models/glm/glm5/tokenizer.py",
    stubs={"batchgen.config.tokenizer_registry": _TOKENIZER_REGISTRY},
)


@pytest.fixture(autouse=True)
def _pin_real_model_registry(monkeypatch):
    # prefix_reuse imports the registry lazily, at call time.
    monkeypatch.setitem(sys.modules, "batchgen.config.model_registry", MR)


def _config_module():
    return CFG


# --------------------------------------------------------------------------- #
#  Routing decisions
# --------------------------------------------------------------------------- #


def _outcome(fn, *args, **kwargs):
    """A decision's value, or the exception class that refused the name."""
    try:
        return ("ok", fn(*args, **kwargs))
    except (ValueError, KeyError, RuntimeError) as exc:
        return ("refused", type(exc).__name__)


def _releases_of_type(model_type):
    """Geometries of the releases carrying ``model_type`` (identity erased)."""
    if model_type is None:
        return None
    return frozenset(
        r.geometry for r in FAMILY.GLM5_RELEASES if r.model_type == model_type
    )


def _resolved_config(name):
    return BMC.BatchGenModelConfig.resolve(name, checkpoint_path=None)


def _config_defaults_without_identity(name):
    values = dataclasses.asdict(_resolved_config(name))
    values.pop("model_type")
    values.pop("_name_or_path", None)
    return values


def _host_config(num_layers):
    return SimpleNamespace(
        page_size_tokens=64, num_pages=1024, sequence_table_capacity=2048,
        num_layers=num_layers, num_k_heads=1, k_head_dim=576,
        k_element_size_bytes=2, num_v_heads=0, v_head_dim=0,
        v_element_size_bytes=2,
    )


def _prefix_reuse(name):
    model_type = PR.require_prefix_cache_model_support(name)
    kwargs = dict(
        model_name=name, kv_dtype="bfloat16", host_kv_config=_host_config(78),
        shm_name="glm5_routing_test",
    )
    try:
        runtime = PR.build_prefix_cache_runtime_config(
            aux_host_kv_config=_host_config(21), **kwargs
        )
    except ValueError:
        runtime = PR.build_prefix_cache_runtime_config(**kwargs)
    return (
        PR.prefix_reuse_requires_prefill_pool(model_type),
        tuple(spec.semantic for spec in runtime.group_specs),
    )


def _sparse_prefill_decision(name):
    model_type = _resolved_config(name).model_type
    return (
        SP.should_use_glm52_sparse_prefill(model_type, 4096, 2048),
        SP.should_use_glm52_sparse_prefill(model_type, 2048, 2048),
    )


def _sparse_prefill_membership(name):
    model_type = _resolved_config(name).model_type
    return (
        model_type in FAMILY.SPARSE_PREFILL_MODEL_TYPES,
        model_type in TP.SPARSE_PREFILL_MODEL_TYPES,
        model_type in PR._DSA_MODEL_TYPES,
    )


def _preflight_contract(name):
    return RP._CONTRACTS[RP._detect_model_type_from_identifier(name)]


DECISIONS = (
    ("primary_host_kv_profile", lambda n: _outcome(HK._resolve_profile, n)),
    ("indexer_host_kv_profile", lambda n: _outcome(HK._resolve_indexer_profile, n)),
    ("is_dsa_model", lambda n: _outcome(HK.is_dsa_model, n)),
    ("model_byte_size", lambda n: _outcome(PU.get_model_byte_size, n)),
    ("cuda_graph_fp8_default", lambda n: _outcome(CG.is_glm5_fp8_graph_default_model, n)),
    ("cuda_graph_whole_model", lambda n: _outcome(
        CG.glm5_whole_model_cuda_graph_requested_for_model, n, enable_cuda_graph=True,
    )),
    ("registry_model_type_geometry", lambda n: _outcome(
        lambda x: _releases_of_type(MR._detect_model_type_from_identifier(x)), n,
    )),
    ("preflight_model_type_geometry", lambda n: _outcome(
        lambda x: _releases_of_type(RP._detect_model_type_from_identifier(x)), n,
    )),
    ("preflight_contract", lambda n: _outcome(_preflight_contract, n)),
    ("prefix_reuse", lambda n: _outcome(_prefix_reuse, n)),
    ("sparse_prefill_decision", lambda n: _outcome(_sparse_prefill_decision, n)),
    ("sparse_prefill_membership", lambda n: _outcome(_sparse_prefill_membership, n)),
    ("config_variant_matched", lambda n: _outcome(
        lambda x: BMC.BatchGenModelConfig._match_variant(x) is not None, n,
    )),
    ("config_defaults", lambda n: _outcome(_config_defaults_without_identity, n)),
    ("tokenizer_module", lambda n: _outcome(
        lambda x: TR._TOKENIZER_MODULES.get(TR.resolve_tokenizer_type(x)), n,
    )),
)

IDENTIFIER_FORMS = ("zai-org/GLM-{v}-FP8", "zai-org/GLM-{v}", "glm-{v}-fp8", "glm-{v}")


def _label(release):
    return f"GLM-{release.version}"


_PAIRS = FAMILY.same_geometry_pairs()


def test_the_release_table_has_same_geometry_pairs():
    # Without a pair the pairwise and isomorphism tests would pass vacuously.
    assert {(_label(a), _label(b)) for a, b in _PAIRS} == {
        ("GLM-5.3", "GLM-5.2"), ("GLM-5.1", "GLM-5"),
    }


@pytest.mark.parametrize("decision", DECISIONS, ids=[name for name, _ in DECISIONS])
@pytest.mark.parametrize("form", IDENTIFIER_FORMS)
@pytest.mark.parametrize(
    "pair", _PAIRS, ids=[f"{_label(a)}~{_label(b)}" for a, b in _PAIRS],
)
def test_same_geometry_releases_route_identically(pair, form, decision):
    a, b = pair
    _, resolve = decision
    name_a, name_b = form.format(v=a.version), form.format(v=b.version)
    assert resolve(name_a) == resolve(name_b), (name_a, name_b)


def test_decisions_see_the_geometry_split():
    # A decision set that cannot tell GLM-5.2 from GLM-5 proves nothing.
    resolvers = dict(DECISIONS)
    for name in ("indexer_host_kv_profile", "sparse_prefill_decision",
                 "prefix_reuse", "config_defaults"):
        assert (resolvers[name]("zai-org/GLM-5.2-FP8")
                != resolvers[name]("zai-org/GLM-5-FP8")), name


# --------------------------------------------------------------------------- #
#  Alias isomorphism per consumer table
# --------------------------------------------------------------------------- #


def _version_regex(release):
    if release.minor is None:
        return re.compile(r"(glm[-_]?)5(?![._]?\d)", re.I)
    return re.compile(rf"(glm[-_]?)5([._]?){release.minor}(?!\d)", re.I)


def _mentions(key, release):
    return _version_regex(release).search(key) is not None


def _swap(key, src, dst):
    """Spell ``key``'s ``src`` version as ``dst`` (``src`` is a minor release)."""
    assert src.minor is not None

    def repl(match):
        prefix, sep = match.group(1), match.group(2)
        if dst.minor is None:
            return f"{prefix}5"
        return f"{prefix}5{sep}{dst.minor}"

    return _version_regex(src).sub(repl, key)


def _glm_items(table):
    items = table.items() if isinstance(table, dict) else ((k, None) for k in table)
    return [(k, v) for k, v in items if "glm" in k.lower()]


# (name, table getter, whether values are routing and must match)
CONSUMER_TABLES = (
    ("host_kv_primary", lambda: HK._PROFILE_ALIASES, True),
    ("host_kv_indexer", lambda: HK._INDEXER_PROFILE_ALIASES, True),
    ("cuda_graph_fp8_default", lambda: CG._GLM5_FP8_GRAPH_DEFAULT_PATTERNS, False),
    ("model_name_patterns", lambda: MR.MODEL_NAME_PATTERNS, False),
    ("config_variants", lambda: BMC._SUPPORTED_VARIANTS, False),
    ("tokenizer_name_patterns", lambda: TR.TOKENIZER_NAME_PATTERNS, False),
    ("model_byte_sizes", lambda: PU.MODEL_BYTE_SIZES, True),
    # runtime_preflight's identifier rows are excluded by design: dead rows are
    # dropped, so GLM-5 keeps a row GLM-5.1 does not need.
)


@pytest.mark.parametrize(
    "table", CONSUMER_TABLES, ids=[name for name, _, _ in CONSUMER_TABLES],
)
@pytest.mark.parametrize(
    "pair", _PAIRS, ids=[f"{_label(a)}~{_label(b)}" for a, b in _PAIRS],
)
def test_alias_tables_are_isomorphic_across_same_geometry_releases(pair, table):
    a, b = pair
    _, getter, values_route = table
    items = _glm_items(getter())
    of_a = {_swap(k, a, b): v for k, v in items if _mentions(k, a)}
    of_b = {k: v for k, v in items if _mentions(k, b)}
    assert of_a, f"no {_label(a)} key: the comparison would be vacuous"
    assert set(of_a) == set(of_b)
    if values_route:
        assert of_a == of_b


# --------------------------------------------------------------------------- #
#  Literal scan: GLM-5.x identifiers live only in the release table
# --------------------------------------------------------------------------- #

# A string literal whose whole value is a GLM-5.x identifier (a routing alias,
# model_type or config class name). Prose that mentions a version, e.g. an
# error message, is not a routing literal and is not matched. Known limits:
# an alias assembled at run time (split of one packed literal, implicit
# concatenation) or a suffixed spelling such as "glm-5.3-fp8-mtp" is not
# caught here, and base GLM-5 spellings without a minor are out of scope; the
# pairwise and isomorphism checks above still cover every generated table.
_IDENTIFIER = re.compile(
    r"(?:zai-org/)?glm[-_ ]?5[._]?\d+(?:[-_]fp8)?|glm_moe_dsa_5_\d+|glm5\d+config",
    re.I,
)

# Tables that keep literal GLM rows; the guarded-copy tests below pin each one
# to the release registry.
GUARDED_COPY_FILES = frozenset({
    # These sit below or above the model layer (L4) and cannot import the
    # release registry under the layer boundary rule, so their tables stay
    # literal until the model registration seam (sub-project
    # batchgen-kv-profile-registration-seam) lands.
    "batchgen/kv_cache/host_kv_mananger_config.py",
    "batchgen/prefix_reuse/config.py",
    "batchgen/runtime_preflight.py",
    "batchgen/server/process_utils.py",
    # The tokenizer registry must load without any model package: its CPU
    # tests (test_kimi_k3_tokenizer.py, test_glm52_tokenizer.py) load it
    # against an empty batchgen.models, so it cannot import the registry.
    "batchgen/config/tokenizer_registry.py",
})

LITERAL_ALLOWLIST = GUARDED_COPY_FILES | frozenset({
    # The release registry itself.
    "batchgen/models/glm/glm5/releases.py",
    # Config classes declare their own model_type and register under it.
    "batchgen/models/glm/glm5/config.py",
    # Tokenizer classes register under their chat-template identities.
    "batchgen/models/glm/glm5/tokenizer.py",
    # The GLM-5.3 chat contract (reasoning_effort / enable_thinking) keys on
    # the served name; it is a chat semantic, not a code path. (io_struct.py,
    # the other half of that contract, names GLM-5.3 only in help prose.)
    "batchgen/server/batch_scheduler.py",
})


def _string_values(source):
    kinds = {tokenize.STRING}
    fstring_middle = getattr(tokenize, "FSTRING_MIDDLE", None)
    if fstring_middle is not None:
        kinds.add(fstring_middle)
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type not in kinds:
            continue
        try:
            value = ast.literal_eval(token.string)
        except (ValueError, SyntaxError):
            value = re.sub(r"^[A-Za-z]*(['\"]{1,3})|(['\"]{1,3})$", "", token.string)
        if isinstance(value, str):
            yield value


def _files_with_glm5x_literals():
    found = {}
    for path in sorted((_ROOT / "batchgen").rglob("*.py")):
        hits = sorted({
            value for value in _string_values(path.read_text(encoding="utf-8"))
            if _IDENTIFIER.fullmatch(value.strip())
        })
        if hits:
            found[path.relative_to(_ROOT).as_posix()] = hits
    return found


def test_glm5x_identifier_literals_live_only_in_the_allowlist():
    found = _files_with_glm5x_literals()
    unexpected = {f: hits for f, hits in found.items() if f not in LITERAL_ALLOWLIST}
    assert not unexpected, (
        "GLM-5.x identifiers spelled outside batchgen/models/glm/glm5/releases.py; "
        f"generate them from the release registry instead: {unexpected}"
    )
    assert set(found) == LITERAL_ALLOWLIST, (
        f"stale allowlist entries: {sorted(LITERAL_ALLOWLIST - set(found))}"
    )


def test_literal_scan_sees_a_hand_listed_alias():
    source = 'ALIASES = {"glm-5.3": "glm5_indexer", "msg": f"GLM-5.3 is {x}"}\n'
    assert [v for v in _string_values(source) if _IDENTIFIER.fullmatch(v)] == ["glm-5.3"]


# --------------------------------------------------------------------------- #
#  Guarded copies: literal tables outside L4 equal what the registry generates
# --------------------------------------------------------------------------- #

_GLM5_HOST_KV_PROFILES = frozenset(
    {FAMILY.GLM5_PRIMARY_HOST_KV_PROFILE}
    | {r.geometry.indexer_host_kv_profile for r in FAMILY.GLM5_RELEASES}
)


def _glm_alias_rows(aliases):
    """Rows naming a GLM model or routing to a GLM-5 profile."""
    return {
        alias: profile for alias, profile in aliases.items()
        if "glm" in alias.lower() or profile in _GLM5_HOST_KV_PROFILES
    }


def test_guarded_copy_host_kv_aliases_equal_the_registry():
    # Exact-key lookups: content matters, insertion order does not.
    primary = FAMILY.primary_host_kv_aliases()
    assert len(primary) == len(set(primary))
    assert _glm_alias_rows(HK._PROFILE_ALIASES) == {
        alias: FAMILY.GLM5_PRIMARY_HOST_KV_PROFILE for alias in primary
    }
    indexer = {
        alias: profile
        for profile, aliases in FAMILY.indexer_host_kv_aliases().items()
        for alias in aliases
    }
    assert _glm_alias_rows(HK._INDEXER_PROFILE_ALIASES) == indexer


def test_guarded_copy_prefix_reuse_types_equal_the_registry():
    glm_types = set(FAMILY.model_types())
    assert {t for t in PR._DSA_MODEL_TYPES if t.startswith("glm")} == set(
        FAMILY.SPARSE_PREFILL_MODEL_TYPES
    )
    assert {t for t in PR._POOL_ONLY_MODEL_TYPES if t in glm_types} == set(
        FAMILY.SPARSE_PREFILL_MODEL_TYPES
    )


def test_guarded_copy_preflight_rows_and_contracts_equal_the_registry():
    # First case-insensitive match wins, so the row order is compared too.
    rows = [row for row in RP._KNOWN_IDENTIFIER_PATTERNS if "glm" in row[0].lower()]
    assert rows == list(FAMILY.preflight_identifier_rows())
    glm_contracts = {t: c for t, c in RP._CONTRACTS.items() if t.startswith("glm")}
    assert set(glm_contracts) == set(FAMILY.model_types())
    assert len(set(glm_contracts.values())) == 1, "one DSA contract for every release"


def test_guarded_copy_tokenizer_rows_equal_the_registry():
    # Patterns are substring-matched in order; module rows are exact lookups.
    assert _glm_items(TR.TOKENIZER_NAME_PATTERNS) == list(
        FAMILY.tokenizer_name_patterns().items()
    )
    glm_types = {r.tokenizer_type for r in FAMILY.GLM5_RELEASES}
    assert {
        t: m for t, m in TR._TOKENIZER_MODULES.items()
        if t in glm_types or m == FAMILY.GLM5_TOKENIZER_MODULE
    } == FAMILY.tokenizer_modules()


def test_guarded_copy_model_byte_sizes_equal_the_registry():
    # The partial match takes the first key a name contains, so order matters.
    assert _glm_items(PU.MODEL_BYTE_SIZES) == list(FAMILY.model_byte_sizes().items())


# --------------------------------------------------------------------------- #
#  Legacy snapshot: the hand-maintained tables on main before the generator
# --------------------------------------------------------------------------- #

_GIB = 1024**3
_LEGACY_PRIMARY_ALIASES = {
    alias: "glm5_mla"
    for alias in (
        "zai-org/glm-5.2-fp8", "zai-org/glm-5.2", "glm-5.2-fp8", "glm-5.2",
        "zai-org/glm-5.3-fp8", "zai-org/glm-5.3", "glm-5.3-fp8", "glm-5.3",
        "zai-org/glm-5-fp8", "zai-org/glm-5", "glm-5-fp8", "glm-5",
        "zai-org/glm-5.1-fp8", "zai-org/glm-5.1", "glm-5.1-fp8", "glm-5.1",
    )
}
_LEGACY_INDEXER_ALIASES = {
    **{alias: "glm5_2_indexer" for alias in (
        "zai-org/glm-5.2-fp8", "zai-org/glm-5.2", "glm-5.2-fp8", "glm-5.2",
        "zai-org/glm-5.3-fp8", "zai-org/glm-5.3", "glm-5.3-fp8", "glm-5.3",
    )},
    **{alias: "glm5_indexer" for alias in (
        "zai-org/glm-5-fp8", "zai-org/glm-5", "glm-5-fp8", "glm-5",
        "zai-org/glm-5.1-fp8", "zai-org/glm-5.1", "glm-5.1-fp8", "glm-5.1",
    )},
}
_LEGACY_CUDA_GRAPH_PATTERNS = (
    "zai-org/glm-5-fp8", "zai-org/glm-5.1-fp8", "zai-org/glm-5.2-fp8",
    "zai-org/glm-5.3-fp8", "glm-5-fp8", "glm_5_fp8", "glm-5.1-fp8",
    "glm_5.1_fp8", "glm-5.2-fp8", "glm-5.3-fp8", "glm_5.2_fp8", "glm_5.3_fp8",
    "glm5-fp8", "glm5_fp8", "glm51-fp8", "glm51_fp8", "glm52-fp8", "glm53-fp8",
    "glm52_fp8", "glm53_fp8",
)
_LEGACY_MODEL_NAME_PATTERNS = [
    ("GLM-5.3-FP8", "glm_moe_dsa_5_3"), ("GLM-5.3", "glm_moe_dsa_5_3"),
    ("GLM-5.2-FP8", "glm_moe_dsa_5_2"), ("GLM-5.2", "glm_moe_dsa_5_2"),
    ("GLM-5.1-FP8", "glm_moe_dsa"), ("GLM-5.1", "glm_moe_dsa"),
    ("GLM-5-FP8", "glm_moe_dsa"), ("GLM-5", "glm_moe_dsa"),
]
_LEGACY_CONFIG_VARIANTS = [
    (pattern, ("batchgen.models.glm.glm5.config", cls))
    for pattern, cls in (
        ("GLM-5.3-FP8", "GLM53Config"), ("GLM-5.3", "GLM53Config"),
        ("GLM-5.2-FP8", "GLM52Config"), ("GLM-5.2", "GLM52Config"),
        ("GLM-5.1-FP8", "GLM5Config"), ("GLM-5.1", "GLM5Config"),
        ("GLM-5-FP8", "GLM5Config"), ("GLM-5", "GLM5Config"),
    )
]
_LEGACY_TOKENIZER_PATTERNS = [
    ("GLM-5.3-FP8", "glm_moe_dsa_5_3"), ("GLM-5.3", "glm_moe_dsa_5_3"),
    ("GLM-5.2-FP8", "glm_moe_dsa_5_2"), ("GLM-5.2", "glm_moe_dsa_5_2"),
    ("GLM-5.1-FP8", "glm_moe_dsa_5_1"), ("GLM-5.1", "glm_moe_dsa_5_1"),
    ("GLM-5-FP8", "glm_moe_dsa"), ("GLM-5", "glm_moe_dsa"),
]
_LEGACY_TOKENIZER_MODULES = [
    (t, "batchgen.models.glm.glm5.tokenizer")
    for t in ("glm_moe_dsa", "glm_moe_dsa_5_1", "glm_moe_dsa_5_2", "glm_moe_dsa_5_3")
]
_LEGACY_MODEL_BYTE_SIZES = [
    ("zai-org/GLM-5-FP8", 760 * _GIB), ("zai-org/GLM-5", 1400 * _GIB),
    ("zai-org/GLM-5.1-FP8", 760 * _GIB), ("zai-org/GLM-5.1", 1400 * _GIB),
    ("zai-org/GLM-5.2-FP8", 760 * _GIB), ("zai-org/GLM-5.2", 1400 * _GIB),
    ("zai-org/GLM-5.3-FP8", 760 * _GIB), ("zai-org/GLM-5.3", 1400 * _GIB),
]
_LEGACY_PREFLIGHT_ROWS = [
    ("GLM-5.3", "glm_moe_dsa_5_3"), ("GLM-5.2", "glm_moe_dsa_5_2"),
    ("GLM-5", "glm_moe_dsa"),
]
_LEGACY_SPARSE_PREFILL_MODEL_TYPES = ("glm_moe_dsa_5_2", "glm_moe_dsa_5_3")


def test_host_kv_aliases_match_the_legacy_tables():
    assert dict(_glm_items(HK._PROFILE_ALIASES)) == _LEGACY_PRIMARY_ALIASES
    assert dict(_glm_items(HK._INDEXER_PROFILE_ALIASES)) == _LEGACY_INDEXER_ALIASES


def test_cuda_graph_patterns_match_the_legacy_table():
    generated = CG._GLM5_FP8_GRAPH_DEFAULT_PATTERNS
    assert len(generated) == len(set(generated))
    assert set(generated) == set(_LEGACY_CUDA_GRAPH_PATTERNS)


def test_ordered_name_tables_match_the_legacy_tables():
    # Substring-matched in iteration order, so the order is compared too.
    assert _glm_items(MR.MODEL_NAME_PATTERNS) == _LEGACY_MODEL_NAME_PATTERNS
    assert list(BMC._SUPPORTED_VARIANTS.items()) == _LEGACY_CONFIG_VARIANTS
    assert _glm_items(TR.TOKENIZER_NAME_PATTERNS) == _LEGACY_TOKENIZER_PATTERNS
    assert _glm_items(TR._TOKENIZER_MODULES) == _LEGACY_TOKENIZER_MODULES
    assert _glm_items(PU.MODEL_BYTE_SIZES) == _LEGACY_MODEL_BYTE_SIZES
    assert _glm_items(dict(RP._KNOWN_IDENTIFIER_PATTERNS)) == _LEGACY_PREFLIGHT_ROWS


def test_preflight_contracts_match_the_legacy_contract():
    legacy = RP.RuntimeContract(
        modules=("flash_attn_interface", "flash_mla", "libucx"),
        extensions=(RP._COMMON_EXTENSIONS + RP._GLM5_DSA_EXTENSIONS
                    + RP._DSA_MOE_WGMMA_EXTENSIONS),
        require_deepgemm=True,
    )
    glm = {t: c for t, c in RP._CONTRACTS.items() if t.startswith("glm")}
    assert glm == {t: legacy for t in ("glm_moe_dsa", "glm_moe_dsa_5_2", "glm_moe_dsa_5_3")}


def test_sparse_prefill_sets_match_the_legacy_sets():
    assert FAMILY.SPARSE_PREFILL_MODEL_TYPES == _LEGACY_SPARSE_PREFILL_MODEL_TYPES
    assert TP.SPARSE_PREFILL_MODEL_TYPES == _LEGACY_SPARSE_PREFILL_MODEL_TYPES
    assert PR._DSA_MODEL_TYPES == frozenset(_LEGACY_SPARSE_PREFILL_MODEL_TYPES)


# --------------------------------------------------------------------------- #
#  Real-config anchor
# --------------------------------------------------------------------------- #


def _release_for_checkpoint(dir_name):
    """The most-specific release whose ``GLM-5[.x]`` token names the dir."""
    for release in FAMILY.GLM5_RELEASES:
        if re.search(rf"GLM-{re.escape(release.version)}(?![.\d])", dir_name, re.I):
            return release
    raise AssertionError(f"no GLM-5 release matches checkpoint dir {dir_name!r}")


def _stored_indexer_layers(profile, num_layers):
    layer_map = profile.logical_to_physical_layer
    if layer_map is None:
        assert profile.num_layers == num_layers
        return tuple(range(num_layers))
    assert len(layer_map) == num_layers
    return tuple(layer for layer, physical in enumerate(layer_map) if physical >= 0)


def _check_checkpoint_anchor(config_dir):
    release = _release_for_checkpoint(config_dir.name)
    hf = json.loads((config_dir / "config.json").read_text())
    assert hf["model_type"] == "glm_moe_dsa"
    assert hf["architectures"] == ["GlmMoeDsaForCausalLM"]

    config_cls = getattr(_config_module(), release.config_class)
    config = config_cls.from_hf(copy.deepcopy(hf))
    assert type(config).__name__ == release.config_class
    assert config.model_type == release.model_type

    geometry = release.geometry
    assert hf.get("index_topk_freq", 1) == geometry.index_topk_freq
    assert hf.get("index_skip_topk_offset") == geometry.index_skip_topk_offset
    num_layers = hf["num_hidden_layers"]
    full_layers = tuple(
        layer for layer in range(num_layers)
        if not SCHEDULE.dsa_layer_skips_topk(config, layer)
    )
    profile = HK._PROFILE_REGISTRY[geometry.indexer_host_kv_profile]
    assert full_layers == _stored_indexer_layers(profile, num_layers), (
        "full_layers of the checkpoint differ from the layers its indexer "
        f"profile {geometry.indexer_host_kv_profile!r} stores"
    )
    return release


_CHECKPOINT_DIRS = sorted(p.parent for p in _FIXTURES.glob("*/config.json"))


def test_checkpoint_fixtures_exist():
    assert {p.name for p in _CHECKPOINT_DIRS} >= {"GLM-5.2-FP8", "GLM-5.3-FP8"}


@pytest.mark.parametrize("config_dir", _CHECKPOINT_DIRS, ids=lambda p: p.name)
def test_released_config_selects_its_release_indexer_layers(config_dir):
    _check_checkpoint_anchor(config_dir)


def _synthetic_checkpoint(tmp_path, dir_name, edit):
    hf = json.loads((_FIXTURES / "GLM-5.2-FP8" / "config.json").read_text())
    edit(hf)
    target = tmp_path / dir_name
    target.mkdir()
    (target / "config.json").write_text(json.dumps(hf))
    return target


def _as_base_glm5(hf):
    for key in ("index_topk_freq", "index_skip_topk_offset", "indexer_types",
                "index_share_for_mtp_iteration"):
        hf.pop(key, None)


def test_anchor_holds_for_a_synthetic_every_layer_config(tmp_path):
    release = _check_checkpoint_anchor(
        _synthetic_checkpoint(tmp_path, "GLM-5-FP8", _as_base_glm5)
    )
    assert release.minor is None


def test_anchor_rejects_a_schedule_its_profile_does_not_store(tmp_path):
    # The incident in reverse: a GLM-5.3 checkpoint routed to a profile that
    # stores a different layer set must fail the anchor.
    with pytest.raises(AssertionError):
        _check_checkpoint_anchor(
            _synthetic_checkpoint(tmp_path, "GLM-5.3-FP8", _as_base_glm5)
        )


def test_anchor_layer_check_rejects_the_incident_profile(monkeypatch):
    # The incident exactly: the real GLM-5.3 config (geometry fields intact)
    # with its release routed to the 78-layer bf16 indexer profile. Only the
    # stored-layer comparison can catch this one.
    release = _release_for_checkpoint("GLM-5.3-FP8")
    wrong = dataclasses.replace(
        release,
        geometry=dataclasses.replace(
            release.geometry, indexer_host_kv_profile="glm5_indexer"
        ),
    )
    monkeypatch.setattr(
        FAMILY,
        "GLM5_RELEASES",
        tuple(wrong if r is release else r for r in FAMILY.GLM5_RELEASES),
    )
    with pytest.raises(AssertionError, match="full_layers"):
        _check_checkpoint_anchor(_FIXTURES / "GLM-5.3-FP8")


# --------------------------------------------------------------------------- #
#  Registry integrity
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("release", FAMILY.GLM5_RELEASES, ids=_label)
def test_release_resolves_to_its_own_registered_identity(release):
    config_cls = getattr(CFG, release.config_class)
    assert config_cls.model_type == release.model_type
    assert _CONFIG_REGISTRY.CONFIG_REGISTRY[release.model_type] is config_cls

    assert release.tokenizer_type in _TOKENIZER_REGISTRY.TOKENIZER_REGISTRY
    assert TR._TOKENIZER_MODULES[release.tokenizer_type] == FAMILY.GLM5_TOKENIZER_MODULE

    for name in (f"zai-org/GLM-{release.version}-FP8", f"GLM-{release.version}"):
        assert MR._detect_model_type_from_identifier(name) == release.model_type
        assert RP._detect_model_type_from_identifier(name) == release.model_type
        assert TR.resolve_tokenizer_type(name) == release.tokenizer_type
        assert BMC.BatchGenModelConfig._match_variant(name) == (
            FAMILY.GLM5_CONFIG_MODULE, release.config_class,
        )

    assert FAMILY.GLM5_PRIMARY_HOST_KV_PROFILE in HK._PROFILE_REGISTRY
    assert release.geometry.indexer_host_kv_profile in HK._PROFILE_REGISTRY


@pytest.mark.parametrize("table", [
    MR.MODEL_NAME_PATTERNS, BMC._SUPPORTED_VARIANTS, TR.TOKENIZER_NAME_PATTERNS,
], ids=["model_name_patterns", "config_variants", "tokenizer_name_patterns"])
def test_substring_tables_list_specific_before_broad(table):
    rows = list(table.items())
    for i, (earlier, earlier_value) in enumerate(rows):
        for later, later_value in rows[i + 1:]:
            if earlier in later and earlier_value != later_value:
                pytest.fail(f"{later!r} is unreachable behind {earlier!r}")


def test_release_table_is_most_specific_first():
    versions = [r.version for r in FAMILY.GLM5_RELEASES]
    for i, earlier in enumerate(versions):
        for later in versions[i + 1:]:
            assert f"GLM-{earlier}" not in f"GLM-{later}", (earlier, later)


# --------------------------------------------------------------------------- #
#  Torch-free import
# --------------------------------------------------------------------------- #


def test_release_table_imports_without_torch_or_other_batchgen_modules():
    probe = (
        "import sys\n"
        "import batchgen.models.glm.glm5.releases\n"
        "assert 'torch' not in sys.modules, 'torch imported'\n"
        "extra = sorted(m for m in sys.modules if m.startswith('batchgen')\n"
        "               and m not in ('batchgen', 'batchgen.models',\n"
        "                             'batchgen.models.glm',\n"
        "                             'batchgen.models.glm.glm5',\n"
        "                             'batchgen.models.glm.glm5.releases'))\n"
        "assert not extra, extra\n"
    )
    env = dict(os.environ, PYTHONPATH=str(_ROOT))
    result = subprocess.run(
        [sys.executable, "-c", probe], cwd=_ROOT, env=env,
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
