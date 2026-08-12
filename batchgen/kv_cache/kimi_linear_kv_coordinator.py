"""Hybrid KV/state coordinator for the ``kimi_linear`` model family.

Kimi-Linear (and Kimi-K3) interleave two fundamentally different attention
mechanisms across their layers:

  * **MLA layers** (NoPE Multi-head Latent Attention) keep a *paged* compressed
    KV cache. Storage grows with sequence length (one entry per token), so it is
    served by :class:`GPUPagedKVCacheManager` with ``num_k_heads=1`` and
    ``k_head_dim=compressed_kv_dim`` (=576: ``kv_lora_rank`` 512 + ``qk_rope`` 64)
    and ``num_v_heads=0`` (MLA stores only the joint compressed KV, no separate V).

  * **KDA layers** (Kimi Delta Attention, a gated linear-attention variant) keep a
    *fixed-size recurrent state* plus short-conv states per sequence — the storage
    does NOT grow with sequence length. These are served by ``KDAStateGPUManager``
    (one state item per active sequence).

Which mechanism a given global layer uses is decided by
``KimiLinearConfig.is_kda_layer(idx)`` (KDA layers are 1-indexed in
``linear_attn_config.kda_layers`` — layer ``idx`` is KDA iff ``idx + 1`` is in
that list). This coordinator composes the two sub-managers and builds two
**complementary** ``logical_to_physical_layer`` maps over the global layer index
space:

    global layer idx is KDA  ->  kda_map[idx] = <next KDA physical slot>,
                                 mla_map[idx] = -1
    global layer idx is MLA  ->  mla_map[idx] = <next MLA physical slot>,
                                 kda_map[idx] = -1

Because the "other" manager is given ``-1`` for every layer it does not own,
asking the wrong manager to resolve a layer raises ``KeyError`` loudly (via
``coordinator_utils.resolve_from_layer_mapping``) instead of silently reading the
wrong physical slot — this is the miswiring guardrail.

The coordinator exposes a single lifecycle / allocation / release surface that
keeps the two sub-managers in lock-step:

  * :meth:`initialize` / :meth:`shutdown` fan out to both managers.
  * :meth:`allocate` reserves MLA pages **and** a KDA state slot for a batch of
    sequences atomically, rolling back everything if either side fails.
  * :meth:`release_sequence` frees the MLA pages **and** the KDA state slot (and
    the KDA manager's per-sequence bookkeeping / block_reps row) in one call.
  * routing accessors send a global layer to the correct sub-manager, letting a
    miswired layer surface as ``KeyError``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from batchgen.kv_cache.gpu_paged_kv_manager import (
    GPUPagedKVCacheManager,
    GPUPagedKVConfig,
)

logger = logging.getLogger(__name__)


def build_kimi_linear_layer_maps(
    config: Any,
) -> Tuple[List[int], List[int], List[bool], int, int]:
    """Builds complementary MLA/KDA ``logical_to_physical_layer`` maps.

    Returns ``(mla_map, kda_map, layer_is_kda, num_mla_layers, num_kda_layers)``
    where ``mla_map``/``kda_map`` are length ``num_hidden_layers`` and use ``-1``
    for layers the corresponding manager does not own.
    """
    num_layers = int(config.num_hidden_layers)
    mla_map: List[int] = []
    kda_map: List[int] = []
    layer_is_kda: List[bool] = []
    mla_slot = 0
    kda_slot = 0
    for idx in range(num_layers):
        if config.is_kda_layer(idx):
            kda_map.append(kda_slot)
            mla_map.append(-1)
            layer_is_kda.append(True)
            kda_slot += 1
        else:
            mla_map.append(mla_slot)
            kda_map.append(-1)
            layer_is_kda.append(False)
            mla_slot += 1
    return mla_map, kda_map, layer_is_kda, mla_slot, kda_slot


class KimiLinearGPUKVCoordinator:
    """Composes an MLA paged-KV manager and a KDA state manager.

    See the module docstring for the layer-split rationale. Sequences are tracked
    identically by both managers: every active sequence owns MLA pages (for the
    MLA layers) *and* one KDA state slot (for the KDA layers).

    Args:
        mla_manager: paged-KV manager covering the MLA layers. Its
            ``logical_to_physical_layer`` must map KDA layers to ``-1``.
        kda_manager: KDA state manager covering the KDA layers. Its
            ``logical_to_physical_layer`` must map MLA layers to ``-1``.
        layer_is_kda: optional per-global-layer boolean classification. If not
            given it is derived from ``config`` (if provided) or from the KDA
            manager's layer mapping.
        config: optional ``KimiLinearConfig`` used to derive ``layer_is_kda`` and
            for diagnostics.
    """

    def __init__(
        self,
        *,
        mla_manager: GPUPagedKVCacheManager,
        kda_manager: Any,
        layer_is_kda: Optional[Sequence[bool]] = None,
        config: Any = None,
    ) -> None:
        self.mla_manager = mla_manager
        self.kda_manager = kda_manager
        self.config = config

        if layer_is_kda is not None:
            self._layer_is_kda = [bool(v) for v in layer_is_kda]
        elif config is not None:
            self._layer_is_kda = [
                bool(config.is_kda_layer(idx))
                for idx in range(int(config.num_hidden_layers))
            ]
        else:
            self._layer_is_kda = self._derive_layer_is_kda(kda_manager)

        self._active_sequences: set[int] = set()

    # ------------------------------------------------------------------ #
    #  Factory
    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        device: Any,
        num_pages: int,
        page_size_tokens: int,
        num_state_items: int,
        kv_dtype: torch.dtype = torch.bfloat16,
        state_dtype: torch.dtype = torch.float32,
        conv_dtype: torch.dtype = torch.bfloat16,
        cuda_graph_max_pages_per_sequence: Optional[int] = None,
        cuda_graph_max_slots: Optional[int] = None,
    ) -> "KimiLinearGPUKVCoordinator":
        """Builds both sub-managers with complementary layer maps.

        Requires ``KDAStateGPUManager``/``KDAStateGPUConfig`` to be importable from
        ``batchgen.kv_cache.kda_state_gpu_manager``.
        """
        try:
            from batchgen.kv_cache.kda_state_gpu_manager import (
                KDAStateGPUConfig,
                KDAStateGPUManager,
            )
        except ImportError as exc:  # pragma: no cover - depends on peer module
            raise ImportError(
                "KimiLinearGPUKVCoordinator.from_config requires "
                "batchgen.kv_cache.kda_state_gpu_manager (KDAStateGPUManager, "
                "KDAStateGPUConfig). Construct the coordinator directly with "
                "pre-built managers if that module is unavailable."
            ) from exc

        mla_map, kda_map, layer_is_kda, num_mla, num_kda = (
            build_kimi_linear_layer_maps(config)
        )

        compressed_kv_dim = int(
            getattr(config, "compressed_kv_dim", None)
            or (int(config.kv_lora_rank) + int(config.qk_rope_head_dim))
        )

        mla_cfg = GPUPagedKVConfig(
            num_layers=num_mla,
            num_pages=int(num_pages),
            page_size_tokens=int(page_size_tokens),
            num_k_heads=1,
            k_head_dim=compressed_kv_dim,
            num_v_heads=0,
            v_head_dim=0,
            kv_dtype=kv_dtype,
            cuda_graph_max_pages_per_sequence=cuda_graph_max_pages_per_sequence,
            cuda_graph_max_slots=cuda_graph_max_slots,
            logical_to_physical_layer=mla_map,
        )
        mla_manager = GPUPagedKVCacheManager(config=mla_cfg, device=device)

        conv_dim = int(config.kda_num_heads) * int(config.kda_head_dim)
        kda_cfg = KDAStateGPUConfig(
            num_kda_layers=num_kda,
            num_state_items=int(num_state_items),
            num_heads=int(config.kda_num_heads),
            head_dim=int(config.kda_head_dim),
            conv_dim=conv_dim,
            conv_width=int(config.kda_conv_size),
            logical_to_physical_layer=kda_map,
        )
        # KDAStateGPUManager mirrors the compressed-state manager constructor
        # signature (config= + device=).
        kda_manager = KDAStateGPUManager(config=kda_cfg, device=device)

        return cls(
            mla_manager=mla_manager,
            kda_manager=kda_manager,
            layer_is_kda=layer_is_kda,
            config=config,
        )

    # ------------------------------------------------------------------ #
    #  Lifecycle
    # ------------------------------------------------------------------ #
    def initialize(self, device: Any = None) -> Dict[str, Any]:
        """Initializes both sub-managers (device is fixed at construction)."""
        results: Dict[str, Any] = {}
        results["mla"] = self._call_first(self.mla_manager, ("initialize",))
        results["kda"] = self._call_first(self.kda_manager, ("initialize",))
        logger.info(
            "KimiLinearGPUKVCoordinator initialized (mla_layers=%d, kda_layers=%d)",
            self.num_mla_layers,
            self.num_kda_layers,
        )
        return results

    def shutdown(self, *, empty_cuda_cache: bool = False) -> Dict[str, Any]:
        """Tears down both sub-managers."""
        results: Dict[str, Any] = {}
        results["kda"] = self._call_first(
            self.kda_manager,
            ("shutdown", "destroy"),
            empty_cuda_cache=empty_cuda_cache,
        )
        results["mla"] = self._call_first(
            self.mla_manager,
            ("destroy", "shutdown"),
            empty_cuda_cache=empty_cuda_cache,
        )
        self._active_sequences.clear()
        return results

    # alias for callers that mirror the paged-manager API
    def destroy(self, *, empty_cuda_cache: bool = False) -> Dict[str, Any]:
        return self.shutdown(empty_cuda_cache=empty_cuda_cache)

    @property
    def is_initialized(self) -> bool:
        mla_ok = bool(getattr(self.mla_manager, "is_initialized", False))
        kda_ok = getattr(self.kda_manager, "is_initialized", None)
        if kda_ok is None:
            kda_ok = getattr(self.kda_manager, "_is_initialized", True)
        return bool(mla_ok and kda_ok)

    # ------------------------------------------------------------------ #
    #  Allocation (atomic across both managers)
    # ------------------------------------------------------------------ #
    def allocate(
        self,
        sequence_ids: Sequence[int],
        num_tokens: Sequence[int],
    ) -> Dict[int, List[int]]:
        """Allocates MLA pages *and* a KDA state slot for a batch of sequences.

        Both sides are reserved atomically: if the KDA side (or the MLA side
        mid-batch) fails, every allocation performed in this call is rolled back
        before the exception propagates.

        Returns the MLA page allocation dict ``{seq_id: [page, ...]}``.
        """
        seq_ids = [int(s) for s in sequence_ids]
        toks = [int(t) for t in num_tokens]
        if len(seq_ids) != len(toks):
            raise ValueError(
                "allocate: sequence_ids and num_tokens must be the same length"
            )
        if not seq_ids:
            return {}

        pre_mla = set(self.mla_manager._sequences.keys())
        kda_done: List[int] = []
        try:
            pages = self.mla_manager.allocate_pages_for_sequences(seq_ids, toks)
            for seq_id in seq_ids:
                self.kda_manager.allocate_state_item(seq_id)
                kda_done.append(seq_id)
        except Exception:
            self._rollback_allocation(pre_mla, kda_done)
            raise

        self._active_sequences.update(seq_ids)
        return pages

    def _rollback_allocation(
        self, pre_mla: set[int], kda_done: Sequence[int]
    ) -> None:
        if kda_done:
            try:
                self.kda_manager.release_sequence_states(list(kda_done))
            except Exception:  # pragma: no cover - best-effort cleanup
                logger.exception("KDA rollback failed during allocate()")
        new_mla = [
            seq_id
            for seq_id in self.mla_manager._sequences.keys()
            if seq_id not in pre_mla
        ]
        if new_mla:
            try:
                self.mla_manager.free_pages_for_sequences(new_mla)
            except Exception:  # pragma: no cover - best-effort cleanup
                logger.exception("MLA rollback failed during allocate()")

    def release_sequence(self, sequence_ids: Sequence[int] | int) -> None:
        """Frees MLA pages and the KDA state slot for the given sequence(s).

        Releases the KDA per-sequence recurrent/conv state item (and the
        manager's block_reps row) together with the MLA pages, keeping both
        managers in lock-step.
        """
        if isinstance(sequence_ids, int):
            seq_ids = [int(sequence_ids)]
        else:
            seq_ids = [int(s) for s in sequence_ids]
        if not seq_ids:
            return

        # KDA release is idempotent-friendly (silently ignores unknown ids in the
        # peer manager); the paged manager raises on unknown ids, so filter.
        self.kda_manager.release_sequence_states(seq_ids)
        known_mla = [s for s in seq_ids if s in self.mla_manager._sequences]
        if known_mla:
            self.mla_manager.free_pages_for_sequences(known_mla)
        for seq_id in seq_ids:
            self._active_sequences.discard(seq_id)

    # ------------------------------------------------------------------ #
    #  Layer routing
    # ------------------------------------------------------------------ #
    def is_kda_layer(self, layer_idx: int) -> bool:
        return bool(self._layer_is_kda[int(layer_idx)])

    def is_mla_layer(self, layer_idx: int) -> bool:
        return not self.is_kda_layer(layer_idx)

    def manager_for_layer(self, layer_idx: int) -> Tuple[str, Any]:
        """Returns ``("kda", kda_manager)`` or ``("mla", mla_manager)``."""
        if self.is_kda_layer(layer_idx):
            return "kda", self.kda_manager
        return "mla", self.mla_manager

    def resolve_mla_physical_layer(self, layer_idx: int) -> int:
        """Resolves an MLA physical slot; raises ``KeyError`` for a KDA layer."""
        return int(self.mla_manager.resolve_physical_layer(int(layer_idx)))

    def resolve_kda_physical_layer(self, layer_idx: int) -> int:
        """Resolves a KDA physical slot; raises ``KeyError`` for an MLA layer."""
        return int(self.kda_manager.resolve_physical_layer(int(layer_idx)))

    def get_mla_layer_kv_with_page_table(self, layer_idx: int):
        """Routes an MLA layer to the paged manager.

        The paged manager resolves the layer through its ``logical_to_physical``
        map, so a KDA layer (mapped to ``-1``) raises ``KeyError``.
        """
        return self.mla_manager.get_layer_kv_with_page_table(int(layer_idx))

    def get_kda_layer_recurrent_view(self, layer_idx: int):
        """Routes a KDA layer to the state manager (KeyError if it is an MLA layer)."""
        if self.is_mla_layer(layer_idx):
            raise KeyError(
                f"get_kda_layer_recurrent_view: layer {layer_idx} is an MLA layer, "
                "not served by the KDA state manager"
            )
        return self.kda_manager.get_layer_recurrent_view(int(layer_idx))

    def get_kda_layer_conv_views(self, layer_idx: int):
        """Routes a KDA layer to the state manager (KeyError if it is an MLA layer)."""
        if self.is_mla_layer(layer_idx):
            raise KeyError(
                f"get_kda_layer_conv_views: layer {layer_idx} is an MLA layer, "
                "not served by the KDA state manager"
            )
        return self.kda_manager.get_layer_conv_views(int(layer_idx))

    def prepare_decode_step(
        self,
        sequence_ids: Sequence[int],
        raw_positions: Sequence[int] | torch.Tensor,
    ) -> None:
        """Prepares KDA decode-step state bookkeeping for the batch."""
        if hasattr(self.kda_manager, "prepare_decode_step"):
            self.kda_manager.prepare_decode_step(sequence_ids, raw_positions)

    # ------------------------------------------------------------------ #
    #  Introspection
    # ------------------------------------------------------------------ #
    @property
    def num_layers(self) -> int:
        return len(self._layer_is_kda)

    @property
    def num_kda_layers(self) -> int:
        return sum(self._layer_is_kda)

    @property
    def num_mla_layers(self) -> int:
        return len(self._layer_is_kda) - self.num_kda_layers

    @property
    def active_sequence_ids(self) -> List[int]:
        return sorted(self._active_sequences)

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _derive_layer_is_kda(kda_manager: Any) -> List[bool]:
        mapping = getattr(kda_manager, "_logical_to_physical_layer", None)
        if mapping is None:
            cfg = getattr(kda_manager, "config", None)
            mapping = getattr(cfg, "logical_to_physical_layer", None)
        if mapping is None:
            raise ValueError(
                "Cannot derive layer classification: KDA manager exposes no "
                "logical_to_physical_layer; pass layer_is_kda or config explicitly"
            )
        return [int(v) >= 0 for v in mapping]

    @staticmethod
    def _call_first(manager: Any, method_names: Sequence[str], **kwargs) -> Any:
        for name in method_names:
            method = getattr(manager, name, None)
            if callable(method):
                try:
                    return method(**kwargs)
                except TypeError:
                    # method does not accept the passed kwargs (e.g. no
                    # empty_cuda_cache); retry without them.
                    return method()
        return None


__all__ = [
    "KimiLinearGPUKVCoordinator",
    "build_kimi_linear_layer_maps",
]
