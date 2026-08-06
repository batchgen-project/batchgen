# ---------------------------------------------------------------------------- #
#  BatchGen                                                                     #
#  copyright (c) EfficientMoE team 2025                                         #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
# ---------------------------------------------------------------------------- #

"""Block Attention Residuals (K3) — serving-path wiring.

K3 REPLACES the classic pre-norm residual body with depth-attention over block
boundaries, so getting this wrong does not fail loudly: it silently changes
every layer's input.  The semantics below are transcribed from the M2 eager
model ``batchgen/models/moonshotai/kimi_k3/model.py``, which is bit-exact to
the HF oracle and is the ground truth for this file:

  * ``prefix_sum`` and ``hidden_states`` DIVERGE — the depth-mix output feeds
    the norms and NEVER the accumulator;
  * at a block boundary (``layer_idx % attn_res_block_size == 0``) the PRE-mix
    ``prefix_sum`` is appended to ``block_residual`` and the accumulator RESETS
    by assignment, not by add;
  * ``block_residual`` is intra-forward scratch, re-zeroed every forward;
  * the mixer is ONE query over ``nb+1`` keys, fp32 interior, with a rank-1 key
    projection (``norm.weight * proj.weight``);
  * the output stage is mixer-THEN-norm, in that order.

The layer body itself already lives in ``model.py``
(``KimiDecoderLayer._forward_attn_residual``); this module supplies the two
pieces the SERVING path was missing — the between-layer carry and the
output-stage mix — plus the mixer both paths share.

Why a carrier exists
--------------------
``KimiLinearModel.forward`` threads ``block_residual`` explicitly and is what
the decode step runs, so decode needs nothing from here.  PREFILL does not go
through it: the worker drives ``model.model.layers`` itself and keeps only
``layer_outputs[0]``, then calls ``model.model.norm`` directly.  Under that
caller nothing seeds ``block_residual``, nothing carries it between layers
(``torch.cat([None, ...])`` dies on layer 0), and the output depth mix is
skipped.  ``batchgen_worker.py`` is core-scope, so the model side closes the
gap itself: :func:`decoder_layer_forward_block_residual` parks the tensor on
:class:`BlockResidualCarrier` between layers, and the pre-hook returned by
:func:`make_output_block_residual_pre_hook` consumes it just before the final
norm.  Every hand-off is checked — a stale, missing or out-of-order carrier
raises rather than quietly producing a different model.

Import-light on purpose (torch only, no relative imports): a test can
file-path load this module without importing ``batchgen`` or ``fla``.
"""

from typing import Optional

import torch
import torch.nn as nn


# ============================================================================
#  Depth mixer (memory-lean)
# ============================================================================


def attn_res_score_weight(proj: nn.Linear, norm) -> torch.Tensor:
    """Rank-1 key-projection fold: ``norm.weight * proj.weight``, fp32.

    The reference does this fold itself; the key normalization and the score
    dot therefore collapse into one weighted sum over the hidden axis.
    """
    return norm.weight.float() * proj.weight.squeeze(0).float()


def apply_attn_res(prefix_sum: torch.Tensor,
                   block_residual: torch.Tensor,
                   proj: nn.Linear,
                   norm,
                   chunk_size: int = 1024) -> torch.Tensor:
    """Memory-lean Block-Attention-Residual depth mixer.

    Port of ``kimi_k3/model.py::_apply_attn_res_lean`` (M2), which is gated
    bit-identical to the unchunked reference at max_abs < 1e-6 in fp32
    (tests/test_kimi_k3_model.py::test_attn_res_lean_equiv).

    Per token: 1 query over ``nb+1`` keys, fp32 throughout;
    ``scores[:, j] = (v_j * rsqrt(mean(v_j^2) + eps)) . w`` with
    ``w = norm.weight * proj.weight``; the value matmul uses the UNNORMALIZED
    fp32 ``v``.  Every op is token-parallel — which is also why packing needs
    no per-sequence awareness here — so the mixer runs in token CHUNKS with the
    verbatim reference op order inside each chunk.  Nothing of shape
    ``(T, nb+1, hidden)`` is ever materialized in fp32: the transient is
    ``O(chunk_size * (nb+1) * hidden)``, independent of T.  That matters in
    serving, where one packed prefill micro-batch is the whole T.

    Args:
        prefix_sum: ``(num_tokens, hidden)``.
        block_residual: ``(num_tokens, num_blocks, hidden)``; num_blocks may
            be 0, in which case this is a 1-key softmax (identity on ``v``).
        proj: the stage's ``*_res_proj`` (``nn.Linear(hidden, 1, bias=False)``).
        norm: the stage's ``*_res_norm`` (RMSNorm with ``variance_epsilon``).

    Returns:
        ``(num_tokens, hidden)`` in ``prefix_sum``'s dtype.
    """
    num_tokens, hidden = prefix_sum.shape
    eps = norm.variance_epsilon
    w = attn_res_score_weight(proj, norm)
    out = torch.empty_like(prefix_sum)
    for start in range(0, num_tokens, chunk_size):
        end = min(start + chunk_size, num_tokens)
        v = torch.cat(
            (block_residual[start:end], prefix_sum[start:end].unsqueeze(1)),
            dim=1).float()                                   # (c, nb+1, H) fp32
        k = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
        scores = (k * w).sum(-1)                             # (c, nb+1)
        probs = scores.softmax(-1).unsqueeze(1)              # (c, 1, nb+1)
        out[start:end] = torch.matmul(probs, v).squeeze(1).to(out.dtype)
    return out


# ============================================================================
#  Between-layer carry for callers that drive the decoder stack directly
# ============================================================================


class BlockResidualCarrier:
    """Intra-forward scratch parking spot for ``block_residual``.

    Process-wide class state, like the ``AttnWrapperBase`` per-step contract:
    one decoder stack is driven at a time, in ascending layer order, and
    ``block_residual`` never outlives a single pass.  Used ONLY by callers that
    do not thread ``block_residual`` themselves (the worker's prepack-prefill
    loop); ``KimiLinearModel.forward`` passes it explicitly and never touches
    the carrier.
    """

    #: Number of decoder layers in one pass; set by the PSM. The output hook
    #: refuses to mix unless the last layer to fill the carrier was the last
    #: layer of the stack.
    num_layers: Optional[int] = None

    _block_residual: Optional[torch.Tensor] = None
    _last_layer: Optional[int] = None

    @classmethod
    def configure(cls, num_layers: int) -> None:
        cls.num_layers = int(num_layers)
        cls.reset()

    @classmethod
    def reset(cls) -> None:
        """Drop any parked scratch (phase switches, error recovery)."""
        cls._block_residual = None
        cls._last_layer = None

    @classmethod
    def peek(cls) -> Optional[torch.Tensor]:
        """Read the parked scratch without consuming it (tests/diagnostics)."""
        return cls._block_residual

    @classmethod
    def borrow(cls, layer_idx: int, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return the ``block_residual`` layer ``layer_idx`` must consume.

        Layer 0 RE-ZEROES it — ``block_residual`` is per-forward scratch, and
        carrying it across forwards is exactly the mutation the M2 suite pins
        (``test_forward_twice_identical``).
        """
        if layer_idx == 0:
            batch, seq_len, hidden = hidden_states.shape
            cls._block_residual = hidden_states.new_zeros(
                batch * seq_len, 0, hidden)
            cls._last_layer = None
            return cls._block_residual
        if cls._last_layer != layer_idx - 1 or cls._block_residual is None:
            raise RuntimeError(
                "Block-residual carrier out of order: layer {} asked for the "
                "carrier but the last layer to fill it was {}. The K3 decoder "
                "stack must be driven whole, in ascending layer order, one "
                "pass at a time — block_residual is intra-forward scratch and "
                "cannot be resumed or shared between passes.".format(
                    layer_idx, cls._last_layer))
        return cls._block_residual

    @classmethod
    def stash(cls, layer_idx: int, block_residual: torch.Tensor) -> None:
        cls._block_residual = block_residual
        cls._last_layer = int(layer_idx)

    @classmethod
    def take(cls) -> Optional[torch.Tensor]:
        """Consume the parked scratch at the end of a pass.

        Returns None when nothing is parked — that is the explicit-threading
        caller (``KimiLinearModel.forward``), which has already applied the
        output mix itself; the hook must then do nothing rather than mix twice.
        """
        block_residual = cls._block_residual
        if block_residual is None:
            return None
        if cls.num_layers is None:
            raise RuntimeError(
                "BlockResidualCarrier.configure(num_layers) was never called; "
                "the output depth mix cannot verify the stack ran to the end.")
        if cls._last_layer != cls.num_layers - 1:
            raise RuntimeError(
                "Block-residual carrier is stale: the output depth mix was "
                "reached with block_residual last written by layer {}, not by "
                "the final layer {}. A partial or aborted decoder pass left "
                "scratch behind.".format(cls._last_layer, cls.num_layers - 1))
        cls.reset()
        return block_residual


def decoder_layer_forward_block_residual(
    self,
    hidden_states: torch.Tensor,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    cu_seqlens=None,
    block_residual: Optional[torch.Tensor] = None,
    **kwargs,
):
    """Serving replacement for ``KimiDecoderLayer.forward`` (K3 only).

    Installed by the PSM via ``types.MethodType``. Supports both calling
    conventions and returns ``(prefix_sum, block_residual)`` for both, so
    ``layer_outputs[0]`` is the prefix_sum either way:

      * EXPLICIT — ``block_residual`` is passed in (``KimiLinearModel.forward``,
        i.e. the decode step). The carrier is not touched.
      * CARRIED — ``block_residual`` is None (the worker's prepack-prefill
        loop). It is taken from and parked back on
        :class:`BlockResidualCarrier`.
    """
    carried = block_residual is None
    if carried:
        block_residual = BlockResidualCarrier.borrow(self.layer_idx, hidden_states)

    prefix_sum, block_residual = self._forward_attn_residual(
        hidden_states, attention_mask, position_ids, past_key_values,
        cu_seqlens, block_residual, **kwargs,
    )

    if carried:
        BlockResidualCarrier.stash(self.layer_idx, block_residual)
    return prefix_sum, block_residual


def make_output_block_residual_pre_hook(model):
    """Pre-hook for ``model.norm`` that applies the OUTPUT depth mix first.

    The K3 output stage is mixer-then-norm. The worker calls
    ``self.model.model.norm(hidden_states)`` directly, with no way to apply the
    mixer in between, so the mixer rides in on the norm's own pre-hook (the
    same mechanism the PSM already uses for the lm_head last-token slice).

    Args:
        model: the ``KimiLinearModel`` owning ``output_attn_res_{proj,norm}``.
    """

    def _pre_hook(module, args):
        block_residual = BlockResidualCarrier.take()
        if block_residual is None:
            return None  # explicit-threading caller already mixed
        hidden_states = args[0]
        shape = hidden_states.shape
        mixed = apply_attn_res(
            hidden_states.reshape(-1, shape[-1]),
            block_residual,
            model.output_attn_res_proj,
            model.output_attn_res_norm,
        ).view(shape)
        return (mixed,) + tuple(args[1:])

    return _pre_hook
