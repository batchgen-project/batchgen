# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-K3.1                                                         #
#  Copyright (c) 2025-2026 BatchGen Team                                        #
#  Licensed under the Apache License, Version 2.0                               #
# ---------------------------------------------------------------------------- #
"""Kimi-K3.1 sparse-attention indexer — eager reference (parity oracle).

The indexer is the ONLY architectural delta between ``moonshotai/Kimi-K3`` and
``moonshotai/Kimi-K3.1-Pro``: five tensors on each of the 24 full-attention
layers, scoring every cached token so MLA attends to a top-k subset instead of
the whole context.

Provenance
----------
Kimi-K3's own remote code is the reference the rest of this package is ported
from: its ``config.json`` maps ``AutoModelForCausalLM`` to
``modeling_kimi_k3.KimiK3ForConditionalGeneration`` and
``text_config.auto_map`` maps to ``modeling_kimi_linear.KimiLinearForCausalLM``
(vendored and md5-pinned at ``tests/kimi_k3_oracle_assets/``).  That file has
no indexer: K3 is dense.

**Kimi-K3.1-Pro publishes no modeling code at all.**  Its ``config.json``
``auto_map`` carries ``AutoConfig`` alone, its ``text_config`` carries no
``auto_map``, and both ``modeling_kimi_linear.py`` and ``modeling_kimi_k3.py``
are absent from the repository (HTTP 404 on each raw path, verified
2026-11-05).  The one published Python statement about the indexer is
``configuration_kimi_k3.py``, whose ``KimiLinearConfig.__init__`` takes
``dsa_config: Optional[dict] = None`` and stores it with no assertions at all
(unlike ``linear_attn_config``, which it does assert on).  So there is no
upstream formula to transcribe, and the eight declared keys plus the 120
shipped tensors are the whole upstream specification.

The formula below is therefore the DSA-family implementation this repository
already carries for GLM-5, re-derived for K3.1's declared geometry:

  * ``batchgen/models/glm/glm5/model.py:384-520`` (``Glm5Indexer``) — the
    module layout: MQA single-head index K via ``wk`` + ``k_norm``,
    multi-head index Q via ``wq_b`` off the MLA ``q_a`` intermediate, per-head
    gates via ``weights_proj``, aggregate over heads, top-k.
  * ``batchgen/models/glm/glm5/model.py:600-691``
    (``score_and_select_relu_gated``) — the scoring formula, including WHY the
    per-head ReLU precedes head weighting (without it negative per-head scores
    cancel positive ones during the sum and distort the selection) and why the
    invalid positions are masked to ``-inf`` BEFORE the ReLU.

K3.1's geometry differs from GLM-5's in exactly two ways, both read off
``dsa_config`` and both enforced at config parse
(``kimi_k3/config.py:parse_dsa_config``):

  * ``index_rope_dim == 0`` — NoPE on the index path.  GLM-5 rotates the first
    64 dims of index Q/K and then applies a Hadamard rotation fused with that
    RoPE; with no rotary dimension there is no rotation to apply on either
    side, so index keys are cached in the same space the query scores them in
    without one.  Any non-zero value raises rather than silently reusing
    GLM-5's rotation.
  * ``index_k_norm_type == "layernorm"`` — the checkpoint ships
    ``indexer.k_norm.weight`` AND ``indexer.k_norm.bias``, which only a
    LayerNorm has.  GLM-5 is also a LayerNorm here, so the code is shared;
    the config key is what makes the agreement checked rather than assumed.

Weight shapes (byte-exact: the released K3.1 index carries 120 tensors that
K3's does not — 5 patterns x 24 layers — and their bf16 bytes sum to
14,877,184 per layer, which is exactly ``total_size`` K3.1 minus K3 divided by
24; see ``kimi_linear/k3/tensor_map.py``)::

    indexer.wq_b.weight          [index_num_heads*index_head_dim, q_lora_rank]
    indexer.wk.weight            [index_head_dim, hidden_size]
    indexer.weights_proj.weight  [index_num_heads, hidden_size]
    indexer.k_norm.weight        [index_head_dim]
    indexer.k_norm.bias          [index_head_dim]

Scope: this module is the eager CPU reference.  It materializes a dense
``[batch, n_heads, q_len, kv_len]`` score tensor and is sized for the parity
oracle and for short prompts, never for production context lengths.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import KimiDsaConfig, KimiK3Config


#: ``attention_mask`` here is an ADDITIVE bias: ``0`` where a key is visible to
#: a query and a large negative (``-inf``) where it is not — both the plain
#: causal form and the packed block-diagonal form built by
#: ``model.py:_build_causal_mask`` / ``_build_block_causal_mask``.  Visibility
#: is therefore ``bias >= 0``, which holds for ``-inf`` and for a
#: ``finfo.min`` sentinel alike, and does not assume which one the caller used.
_VISIBLE_BIAS = 0.0


class KimiK3Indexer(nn.Module):
    """Per-layer index scorer.  Submodule name MUST stay ``indexer``.

    ``apply_weights`` (``batchgen/models/wrappers/base.py``) silently skips a
    checkpoint name that is not a module parameter, so the attribute path
    ``self_attn.indexer.{wq_b,wk,k_norm,weights_proj}`` is the contract with
    ``K3_MLA_INDEXER_TENSOR_NAMES`` in ``kimi_linear/k3/tensor_map.py``: a
    rename on either side loads zeros, silently.
    """

    def __init__(self, config: KimiK3Config, layer_idx: int) -> None:
        super().__init__()
        dsa: Optional[KimiDsaConfig] = config.dsa_config
        if dsa is None:
            raise ValueError(
                "KimiK3Indexer requires a parsed dsa_config; a layer without "
                "one is dense Kimi-K3 and must not build an indexer at all")
        self._build(dsa, int(config.hidden_size), int(config.q_lora_rank),
                    layer_idx)

    @classmethod
    def from_dims(cls, dsa: KimiDsaConfig, *, hidden_size: int,
                  q_lora_rank: int, layer_idx: int) -> "KimiK3Indexer":
        """Build from a parsed ``dsa_config`` without a :class:`KimiK3Config`.

        The serving model holds a ``KimiLinearConfig`` (whose ``dsa()`` parses
        through the same :func:`parse_dsa_config`), so it constructs the SAME
        class the eager reference does: the parity oracle then tests the
        serving module's own parameters and math, not a second transcription.
        """
        if dsa is None:
            raise ValueError("KimiK3Indexer.from_dims requires a parsed dsa_config")
        self = cls.__new__(cls)
        nn.Module.__init__(self)
        self._build(dsa, int(hidden_size), int(q_lora_rank), layer_idx)
        return self

    def _build(self, dsa: KimiDsaConfig, hidden_size: int, q_lora_rank: int,
               layer_idx: int) -> None:
        self.layer_idx = layer_idx
        self.index_n_heads = dsa.index_num_heads
        self.index_head_dim = dsa.index_head_dim
        self.index_topk = dsa.index_topk
        # Folded into the Q.K product, not into the head gates: with the ReLU
        # in between, scaling the input decides which values clip to zero, so
        # the two placements are NOT interchangeable here.
        self.softmax_scale = float(dsa.index_head_dim) ** -0.5

        self.wk = nn.Linear(hidden_size, dsa.index_head_dim, bias=False)
        self.k_norm = nn.LayerNorm(dsa.index_head_dim)
        self.wq_b = nn.Linear(q_lora_rank, dsa.index_dim, bias=False)
        self.weights_proj = nn.Linear(hidden_size, dsa.index_num_heads, bias=False)

    # ------------------------------------------------------------------ #
    #  Pieces (separately testable)                                       #
    # ------------------------------------------------------------------ #
    def index_keys(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """``[batch, kv_len, hidden]`` -> ``[batch, kv_len, index_head_dim]``.

        One MQA key per token, shared by all index heads — this is what the
        serving path caches per token (``index_head_dim`` elements, NOT
        ``index_num_heads * index_head_dim``).  NoPE: nothing is rotated.
        """
        return self.k_norm(self.wk(hidden_states))

    def index_queries(self, q_a: torch.Tensor) -> torch.Tensor:
        """MLA ``q_a`` -> ``[batch, q_len, index_n_heads, index_head_dim]``.

        Reads the POST-``q_a_layernorm`` q-LoRA intermediate, so the index Q
        costs one GEMM on a 1536-wide activation rather than a second
        projection off the 7168-wide hidden.
        """
        batch, q_len = q_a.shape[:2]
        q = self.wq_b(q_a)
        return q.view(batch, q_len, self.index_n_heads, self.index_head_dim)

    def head_gates(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """``[batch, q_len, index_n_heads]`` per-head importance, fp32.

        ``n_heads ** -0.5`` keeps the aggregate's scale independent of the head
        count, matching the GLM-5 reference's ``weights * n_heads**-0.5``.
        """
        gates = self.weights_proj(hidden_states).float()
        return gates * (float(self.index_n_heads) ** -0.5)

    def scores(
        self,
        q_a: torch.Tensor,
        hidden_states: torch.Tensor,
        index_keys: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Aggregate index score per (query, key) pair, fp32 ``[B, Sq, Skv]``.

        ``score[b,q,k] = sum_h gate[b,q,h] * relu(scale * q[b,q,h].k[b,k])``

        ``attention_mask`` (the MLA additive bias) masks invisible keys to
        ``-inf`` BEFORE the ReLU, so they contribute exactly 0 to the sum
        instead of leaking post-ReLU energy into it — a padded or
        cross-sequence key must not be able to change a visible key's rank.
        """
        q = self.index_queries(q_a).float()              # [B, Sq, H, D]
        k = index_keys.float()                           # [B, Skv, D]
        logits = torch.einsum("bqhd,bkd->bhqk", q, k) * self.softmax_scale
        if attention_mask is not None:
            logits = logits.masked_fill(
                _invisible(attention_mask, logits.shape[-1]).unsqueeze(1),
                float("-inf"),
            )
        logits = F.relu(logits)
        gates = self.head_gates(hidden_states)           # [B, Sq, H]
        return torch.einsum("bhqk,bqh->bqk", logits, gates)


def _invisible(attention_mask: torch.Tensor, kv_len: int) -> torch.Tensor:
    """``[B, Sq, Skv]`` bool: True where the additive bias hides the key.

    Accepts the ``[1, 1, Sq, Skv]`` broadcast form the model builds and the
    ``[B, 1, Sq, Skv]`` per-row form; the head axis is dropped (the mask is
    head-independent in this model, and index selection is per query row).
    """
    bias = attention_mask[:, 0, :, :kv_len]
    return bias < _VISIBLE_BIAS


def select_topk(scores: torch.Tensor, invisible: torch.Tensor,
                index_topk: int) -> torch.Tensor:
    """``[B, Sq, Skv]`` bool: the keys each query row attends to.

    ``scores`` is the aggregate index score, ``invisible`` the broadcast
    visibility complement.  Two steps, both load-bearing:

    * Invisible keys are driven to ``-inf`` BEFORE the top-k, so they rank
      last.  ``weights_proj`` is unconstrained, so a visible key's aggregate
      can be NEGATIVE while an invisible key's is exactly 0; ranking against
      the un-masked aggregate would then drop a visible key in favour of an
      invisible one, and the row would attend to fewer than
      ``min(visible, index_topk)`` keys.
    * ``topk`` always returns exactly ``k`` indices, so a row with fewer than
      ``k`` visible keys gets invisible ones among them.  Intersecting with
      visibility is what keeps this function's result a subset of the visible
      set on its own, rather than only after a caller re-applies a mask that
      already hides those keys.
    """
    scores = scores.masked_fill(invisible, float("-inf"))
    effective_topk = min(int(index_topk), scores.shape[-1])
    selected = torch.zeros_like(scores, dtype=torch.bool)
    selected.scatter_(-1, scores.topk(effective_topk, dim=-1).indices, True)
    return selected & ~invisible


def dsa_attention_bias(
    indexer: KimiK3Indexer,
    q_a: torch.Tensor,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """The MLA additive bias restricted to the indexer's selected keys.

    Returns ``attention_mask`` with every UNSELECTED position driven to
    ``-inf``.  Two consequences are load-bearing:

    * When every visible key is selected the returned tensor is **equal to
      ``attention_mask``**, element for element, so the MLA attention that
      consumes it is bit-identical to the dense computation.  That is the
      cheap oracle: a query row that can see at most ``index_topk`` keys has
      no selection to make, and sparse MLA must then reproduce dense MLA
      exactly (``tests/test_kimi_k31_dsa_indexer.py``).
    * Selection is restricted to VISIBLE keys, so it can never cross a causal
      or (in the packed layout) a sequence boundary, whatever the scores say.

    See :func:`select_topk` for the two ordering constraints the selection
    itself has to respect.
    """
    if attention_mask is None:
        raise ValueError(
            "Kimi-K3.1 DSA needs the MLA additive attention bias to know which "
            "keys a query may see; it is built for every prefill with "
            "seq_len > 1 (model.py:_build_causal_mask / "
            "_build_block_causal_mask). A None mask reaches here only on a "
            "decode-shaped (seq_len == 1) call, which this prefill-only "
            "reference does not implement.")

    kv_len = hidden_states.shape[1]
    index_keys = indexer.index_keys(hidden_states)
    scores = indexer.scores(q_a, hidden_states, index_keys, attention_mask)

    invisible = _invisible(attention_mask, kv_len).expand_as(scores)
    selected = select_topk(scores, invisible, indexer.index_topk)

    # Expanded (not mutated): the model shares ONE [1, 1, S, S] causal mask
    # across all 24 DSA layers, so writing into it would make layer 3's
    # selection the next layer's causal mask.
    bias = attention_mask[:, :, :, :kv_len].expand(
        scores.shape[0], -1, -1, -1)
    return bias.masked_fill(~selected.unsqueeze(1), float("-inf"))
