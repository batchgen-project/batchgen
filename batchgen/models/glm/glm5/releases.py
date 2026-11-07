"""GLM-5 release registry: the one source of per-release routing names.

A release is described once, by its minor version and its geometry, in
``GLM5_RELEASES``. The model-layer tables that route a GLM-5.x name or
model_type to a code path -- CUDA-graph defaults, config and model-type
detection, the sparse DSA prefill gate -- are GENERATED from it, so two
releases with the same geometry cannot be routed apart by a stale
hand-maintained list.

The other tables keep literal copies of their GLM rows. Layers below or above
the model layer (host-KV profiles, prefix reuse, runtime preflight, hugepage
sizing) may not import this module, and the tokenizer registry must stay
loadable without any model package (tests load it against an empty
``batchgen.models``). The generators for those copies live here too, and
tests/unit/test_glm5_family_routing_equivalence.py pins each copy to them: a
release added here without its copies fails that test.

Pure standard library: no torch and no other ``batchgen`` import, so the
lightest consumer (a CPU test, the config registries) can import it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

_GIB = 1024**3

# Host-KV profile of the MLA latent (primary) cache. Every GLM-5 release has
# the same 78-layer MLA graph (compressed_kv_dim=576), so it is family-wide.
GLM5_PRIMARY_HOST_KV_PROFILE = "glm5_mla"

# Module of every GLM-5 config class and of every GLM-5 tokenizer class.
GLM5_CONFIG_MODULE = "batchgen.models.glm.glm5.config"
GLM5_TOKENIZER_MODULE = "batchgen.models.glm.glm5.tokenizer"


@dataclass(frozen=True)
class Glm5Geometry:
    """Checkpoint facts that select a code path.

    Two releases with equal geometry must route identically everywhere except
    their own identity (model_type, config class, tokenizer / chat contract).
    """

    # Host-KV profile of the DSA indexer (aux) cache.
    indexer_host_kv_profile: str
    # config.json ``index_topk_freq`` (1 = every layer computes top-k).
    index_topk_freq: int
    # config.json ``index_skip_topk_offset`` (None when the checkpoint has none).
    index_skip_topk_offset: Optional[int]
    # Host weight envelope (bytes) for hugepage sizing. For the 754B GLM-5
    # line: FP8 = 675GB FP8 experts + 17.3GB attn+DSA + 4.5GB embed/dense
    # ~= 700GB + buffer; BF16 = 1350GB experts + 17.3GB attn+DSA + 5.4GB
    # shared + 4.5GB embed/dense ~= 1380GB + buffer.
    fp8_weight_bytes: int
    bf16_weight_bytes: int

    @property
    def sparse_dsa_prefill(self) -> bool:
        """Whether long prompts take the packed sparse DSA prefill path.

        Only a checkpoint that reuses top-k across layers stores a reduced
        (page-split FP8) indexer cache, which the dense prefill cannot fill.
        """
        return self.index_topk_freq > 1


@dataclass(frozen=True)
class Glm5Release:
    """One released GLM-5 checkpoint line."""

    # Minor version token ("3" for GLM-5.3); None for the base GLM-5.
    minor: Optional[str]
    # model_type the config registry, preflight and prefix reuse key on.
    model_type: str
    # Class name in GLM5_CONFIG_MODULE.
    config_class: str
    # Tokenizer type (chat template identity) in the tokenizer registry.
    tokenizer_type: str
    geometry: Glm5Geometry

    @property
    def version(self) -> str:
        """``"5"`` or ``"5.<minor>"``."""
        return "5" if self.minor is None else f"5.{self.minor}"

    @property
    def glued_version(self) -> str:
        """``"5"`` or ``"5<minor>"`` (the ``glm52`` spelling)."""
        return "5" if self.minor is None else f"5{self.minor}"


# GLM-5 / GLM-5.1: every one of the 78 layers computes top-k and owns a bf16
# AoS indexer K slot.
GLM5_BASE_GEOMETRY = Glm5Geometry(
    indexer_host_kv_profile="glm5_indexer",
    index_topk_freq=1,
    index_skip_topk_offset=None,
    fp8_weight_bytes=760 * _GIB,
    bf16_weight_bytes=1400 * _GIB,
)

# GLM-5.2 / GLM-5.3: index_topk_freq=4 / index_skip_topk_offset=3, so only 21
# of 78 layers run the indexer; their keys live in the page-split FP8 profile.
GLM52_GEOMETRY = Glm5Geometry(
    indexer_host_kv_profile="glm5_2_indexer",
    index_topk_freq=4,
    index_skip_topk_offset=3,
    fp8_weight_bytes=760 * _GIB,
    bf16_weight_bytes=1400 * _GIB,
)

# Most-specific first: "GLM-5" is a substring of every "GLM-5.x", so the
# substring-matched tables generated below must try every minor before it.
GLM5_RELEASES: Tuple[Glm5Release, ...] = (
    Glm5Release("3", "glm_moe_dsa_5_3", "GLM53Config", "glm_moe_dsa_5_3", GLM52_GEOMETRY),
    Glm5Release("2", "glm_moe_dsa_5_2", "GLM52Config", "glm_moe_dsa_5_2", GLM52_GEOMETRY),
    # GLM-5.1 shares GLM-5's model_type and config class; only its chat
    # template differs.
    Glm5Release("1", "glm_moe_dsa", "GLM5Config", "glm_moe_dsa_5_1", GLM5_BASE_GEOMETRY),
    Glm5Release(None, "glm_moe_dsa", "GLM5Config", "glm_moe_dsa", GLM5_BASE_GEOMETRY),
)


def _ascending() -> Tuple[Glm5Release, ...]:
    return tuple(reversed(GLM5_RELEASES))


def _unique(values) -> Tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def host_kv_identifiers(release: Glm5Release) -> Tuple[str, ...]:
    """Lower-case names the host-KV tables resolve for one release."""
    v = release.version
    return (f"zai-org/glm-{v}-fp8", f"zai-org/glm-{v}", f"glm-{v}-fp8", f"glm-{v}")


def primary_host_kv_aliases() -> Tuple[str, ...]:
    """Every alias of GLM5_PRIMARY_HOST_KV_PROFILE."""
    return tuple(
        alias for release in GLM5_RELEASES for alias in host_kv_identifiers(release)
    )


def indexer_host_kv_aliases() -> Dict[str, Tuple[str, ...]]:
    """Indexer (aux) host-KV profile -> its aliases, from each geometry."""
    groups: Dict[str, Tuple[str, ...]] = {}
    for release in GLM5_RELEASES:
        profile = release.geometry.indexer_host_kv_profile
        groups[profile] = groups.get(profile, ()) + host_kv_identifiers(release)
    return groups


def cuda_graph_fp8_default_patterns() -> Tuple[str, ...]:
    """Lower-case substrings of the FP8 names that capture the decode graph by
    default under ``--enable-cuda-graph``."""
    patterns = []
    for release in GLM5_RELEASES:
        v, glued = release.version, release.glued_version
        patterns += [
            f"zai-org/glm-{v}-fp8",
            f"glm-{v}-fp8",
            f"glm_{v}_fp8",
            f"glm{glued}-fp8",
            f"glm{glued}_fp8",
        ]
    return tuple(patterns)


def _name_patterns() -> Tuple[Tuple[str, Glm5Release], ...]:
    """Case-sensitive substring patterns, most-specific first."""
    return tuple(
        (pattern, release)
        for release in GLM5_RELEASES
        for pattern in (f"GLM-{release.version}-FP8", f"GLM-{release.version}")
    )


def model_name_patterns() -> Dict[str, str]:
    """Pattern -> model_type rows of ``model_registry.MODEL_NAME_PATTERNS``."""
    return {pattern: release.model_type for pattern, release in _name_patterns()}


def config_variants() -> Dict[str, Tuple[str, str]]:
    """Pattern -> (module, class) rows of ``_SUPPORTED_VARIANTS``."""
    return {
        pattern: (GLM5_CONFIG_MODULE, release.config_class)
        for pattern, release in _name_patterns()
    }


def tokenizer_name_patterns() -> Dict[str, str]:
    """Pattern -> tokenizer_type rows of ``TOKENIZER_NAME_PATTERNS``."""
    return {pattern: release.tokenizer_type for pattern, release in _name_patterns()}


def tokenizer_modules() -> Dict[str, str]:
    """tokenizer_type -> module rows of the tokenizer registry."""
    return {
        tokenizer_type: GLM5_TOKENIZER_MODULE
        for tokenizer_type in _unique(r.tokenizer_type for r in _ascending())
    }


def model_byte_sizes() -> Dict[str, int]:
    """Hugepage-sizing rows of ``process_utils.MODEL_BYTE_SIZES``.

    Ascending by release: the lookup's partial match takes the first key a
    name contains (or is contained in), so the order is part of its result.
    """
    sizes: Dict[str, int] = {}
    for release in _ascending():
        sizes[f"zai-org/GLM-{release.version}-FP8"] = release.geometry.fp8_weight_bytes
        sizes[f"zai-org/GLM-{release.version}"] = release.geometry.bf16_weight_bytes
    return sizes


def model_types() -> Tuple[str, ...]:
    """Every distinct GLM-5 model_type, oldest release first."""
    return _unique(release.model_type for release in _ascending())


def sparse_prefill_model_types() -> Tuple[str, ...]:
    """model_types whose geometry takes the packed sparse DSA prefill."""
    return _unique(
        release.model_type
        for release in _ascending()
        if release.geometry.sparse_dsa_prefill
    )


SPARSE_PREFILL_MODEL_TYPES: Tuple[str, ...] = sparse_prefill_model_types()


def preflight_identifier_rows() -> Tuple[Tuple[str, str], ...]:
    """GLM rows of ``runtime_preflight._KNOWN_IDENTIFIER_PATTERNS``.

    That table is matched case-insensitively and lists no dead row: a pattern
    is dropped when the first LATER pattern it contains already resolves it
    to the same model_type (``GLM-5.1`` is resolved by ``GLM-5``).
    """
    rows = tuple(model_name_patterns().items())
    kept = []
    for index, (pattern, model_type) in enumerate(rows):
        contained = [
            later_type
            for later, later_type in rows[index + 1:]
            if later.lower() in pattern.lower()
        ]
        if contained and contained[0] == model_type:
            continue
        kept.append((pattern, model_type))
    return tuple(kept)


def same_geometry_pairs() -> Tuple[Tuple[Glm5Release, Glm5Release], ...]:
    """Every pair of releases that must route identically."""
    return tuple(
        (a, b)
        for i, a in enumerate(GLM5_RELEASES)
        for b in GLM5_RELEASES[i + 1:]
        if a.geometry == b.geometry
    )
