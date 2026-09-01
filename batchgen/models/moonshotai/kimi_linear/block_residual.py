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


def _apply_attn_res_local(prefix_sum: torch.Tensor,
                          block_residual: torch.Tensor,
                          proj: nn.Linear,
                          norm,
                          chunk_size: int = 1024) -> torch.Tensor:
    """Memory-lean Block-Attention-Residual depth mixer.

    Port of ``kimi_k3/model.py::_apply_attn_res_lean`` (M2), gated against the
    unchunked reference at max_abs < 1e-6 in fp32
    (tests/test_kimi_k3_model.py::test_attn_res_lean_equiv).  That is a
    tolerance, NOT bit equality — see the note on the reference for the
    ragged-final-chunk measurement.

    Per token: 1 query over ``nb+1`` keys, fp32 throughout;
    ``scores[:, j] = (v_j * rsqrt(mean(v_j^2) + eps)) . w`` with
    ``w = norm.weight * proj.weight``; the value matmul uses the UNNORMALIZED
    fp32 ``v``.  Every op is token-parallel — which is also why packing needs
    no per-sequence awareness here — so the mixer runs in token CHUNKS with the
    verbatim reference op order inside each chunk.

    MEMORY, MEASURED (H20, real K3 scale H=7168, nb=8, bf16 in/out, chunk 1024,
    under ``torch.inference_mode``; peak EXTRA device allocation over the call):

        T =  1024   770 MiB        T =  8192    994 MiB
        T =  2048   910 MiB        T = 32768   1330 MiB

    Nothing of shape ``(T, nb+1, hidden)`` is materialized in fp32, and that is
    the win: the fp32 part is ``O(chunk_size * (nb+1) * hidden)`` per live
    tensor.  But do NOT read that as "one 252 MiB buffer, independent of T" —
    several fp32 chunk tensors are live at once (``v``, ``v.pow(2)``, ``k``,
    ``k*w``, plus the bf16 ``cat``), giving a ~0.75 GiB floor, and ``out`` is
    ``torch.empty_like(prefix_sum)``, i.e. ``O(T * hidden)`` on top.

    ``torch.inference_mode`` is load-bearing, not incidental: with grad enabled
    the same call reaches 16.8 GiB at T=32768, because ``proj.weight`` is a leaf
    Parameter and autograd pins every chunk's ``v``/``k``.  The worker's
    prepack-prefill loop is inside ``with torch.inference_mode()``
    (batchgen_worker.py:6939) — any new caller must be too.

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
    resident_tile = getattr(norm, "_resident_prefill_token_tile", None)
    if resident_tile is not None:
        chunk_size = min(int(chunk_size), int(resident_tile))
    out = torch.empty_like(prefix_sum)
    for start in range(0, num_tokens, chunk_size):
        end = min(start + chunk_size, num_tokens)
        v = torch.cat(
            (block_residual[start:end], prefix_sum[start:end].unsqueeze(1)),
            dim=1).float()                                   # (c, nb+1, H) fp32
        k = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
        # k is dead after scoring. Multiplying it in place preserves the same
        # elementwise FP32 product and hidden-axis reduction while avoiding a
        # second (c, nb+1, H) temporary (140 MiB at W2, nb+1=5).
        k.mul_(w)
        scores = k.sum(-1)                                   # (c, nb+1)
        del k
        probs = scores.softmax(-1).unsqueeze(1)              # (c, 1, nb+1)
        out[start:end] = torch.matmul(probs, v).squeeze(1).to(out.dtype)
    return out


def apply_attn_res(prefix_sum: torch.Tensor,
                   block_residual: torch.Tensor,
                   proj: nn.Linear,
                   norm,
                   chunk_size: int = 1024) -> torch.Tensor:
    """Apply the depth mixer, row-sharded during streamed-SP8 TP8 prefill.

    The mixer is independent in the token axis. The prefill strategy therefore
    gives each TP rank a disjoint contiguous row slice, runs the unchanged
    local implementation, and all-gathers the rows needed by the next
    head-parallel attention/MLP. Decode and every non-streamed path leave the
    strategy attribute absent and execute the original full-row body.
    """
    row_group = getattr(norm, "_streamed_sp8_row_group", None)
    profiler = getattr(norm, "_streamed_sp8_profiler", None)
    if (
        profiler is not None
        and not profiler._prefill_profile_enabled
    ):
        profiler = None
    span = profiler.begin_profile_span() if profiler is not None else None

    if row_group is None:
        output = _apply_attn_res_local(
            prefix_sum, block_residual, proj, norm, chunk_size
        )
    else:
        group_size, group_rank, group = row_group
        from .moe_tp_reshard import all_gather_rows, scatter_rows

        local_prefix = scatter_rows(prefix_sum, group_size, group_rank)
        local_residual = scatter_rows(
            block_residual, group_size, group_rank
        )
        local_output = _apply_attn_res_local(
            local_prefix, local_residual, proj, norm, chunk_size
        )
        order_wait = getattr(norm, "_streamed_sp8_order_wait", None)
        if order_wait is not None:
            # The previous layer opened the cross-node weight gate after its
            # MoE. This gather is now the next TP8 collective, so it must
            # observe the same cross-then-TP host issue order on every rank.
            order_wait()
        output = all_gather_rows(
            local_output,
            prefix_sum.shape[0],
            group_size,
            group_rank,
            group,
        )

    if profiler is not None:
        profiler.end_profile_span(
            getattr(norm, "_streamed_sp8_profile_name", "depth_mix"),
            span,
        )
    return output


# ============================================================================
#  Preallocated block_residual scratch
# ============================================================================


def num_block_residual_columns(num_hidden_layers: int,
                               attn_res_block_size: int) -> int:
    """How many block boundaries one whole-stack pass crosses.

    A boundary is a layer with ``layer_idx % attn_res_block_size == 0``, so over
    ``range(num_hidden_layers)`` there are ``ceil(L / block_size)`` of them:
    K3's 93 layers at block size 12 give **8** (layers 0, 12, 24, ..., 84), and
    the SYN-25 test config's 25 layers at block size 3 give 9 (0, 3, ..., 24).
    """
    num_hidden_layers = int(num_hidden_layers)
    attn_res_block_size = int(attn_res_block_size)
    return -(-num_hidden_layers // attn_res_block_size)


class BlockResidualBuffer:
    """One preallocated ``(num_tokens, num_columns, hidden)`` scratch tensor.

    WHY.  The boundary append used to be
    ``torch.cat([block_residual, snapshot], dim=1)``, which allocates the
    ``(S, nb+1, H)`` result while the ``(S, nb, H)`` input is still live.  At
    S=131,072 / H=7168 / bf16 the last boundary (layer 84) therefore held
    12.25 + 14.00 GiB simultaneously, and seven earlier boundaries each churned
    a full reallocation — ``PREFILL_MEMORY_AUDIT.md`` §4/§7 fix 3.  Writing
    column ``nb`` of a buffer that already exists removes the transient and the
    seven intermediate allocations both.

    WHAT IS THREADED IS STILL A PLAIN TENSOR, and still exactly the tensor the
    ``cat`` produced.  :meth:`append` hands back the NARROWED view
    ``buf[:, :nb+1]`` — never the whole buffer.  That is the trap in this
    optimisation and the reason the narrowing is not optional: every consumer
    reads ``block_residual.shape[1]`` as "boundaries seen so far" (the
    ``shape[1] > 0`` gate in ``KimiDecoderLayer._forward_attn_residual``, the
    ``block_residual[start:end]`` read in :func:`apply_attn_res`, the worker's
    ``bres=`` memory log), and a caller handed the full ``(S, 8, H)`` buffer
    would mix eight all-zero keys into layer 0's depth-attention and silently
    compute a different model.

    BIT-EXACTNESS.  The in-place path is a same-dtype ``copy_`` of exactly the
    bytes ``cat`` would have copied into exactly the same logical column, so the
    view's *values* are identical.  Consumers immediately re-``cat`` a token
    slice of it into a fresh contiguous fp32 tensor, so the differing strides
    never reach a reduction: ``apply_attn_res``'s ``v`` is byte-identical and
    identically laid out either way.  Nothing is reassociated and no op order
    changes.  Gated by ``torch.equal`` over a full 93-layer / 8-boundary drive
    in ``tests/test_kimi_k3_block_residual_prealloc.py``.

    Process-wide class state, like :class:`BlockResidualCarrier`: one decoder
    stack is driven at a time and this scratch never outlives one pass.
    :meth:`append` writes in place ONLY when handed back the exact view it last
    produced; a caller-built ``block_residual`` (a test, or any caller that did
    not go through :meth:`seed`) falls through to the original ``torch.cat`` and
    is unaffected.  Everything that *is* on the buffer path is hard-checked.
    """

    _buf: Optional[torch.Tensor] = None
    _view: Optional[torch.Tensor] = None

    @classmethod
    def reset(cls) -> None:
        """Drop the buffer.

        MUST run at the end of every pass, not only on phase switches.  The
        buffer is class state, so unlike the plain local it replaced it does
        NOT die when the prefill frame returns: at S=131,072 / H=7168 / bf16 it
        would otherwise keep 14.00 GiB pinned across ``configure_decoding()``
        and the resident-EP build that follows it, until the first decode step
        reseeded it.  That is a regression the ``cat`` form could not have —
        which is why :meth:`BlockResidualCarrier.reset` calls this, and why
        ``take()`` (end of a carried pass) and ``KimiLinearModel.forward`` (end
        of an explicitly-threaded pass) both go through it.  The narrowed view
        the consumer is holding keeps the storage alive for exactly as long as
        the output depth mix needs it, and no longer.
        """
        cls._buf = None
        cls._view = None

    @classmethod
    def seed(cls, num_tokens: int, hidden: int, num_columns: int, *,
             dtype: torch.dtype, device) -> torch.Tensor:
        """Allocate this pass's buffer; return its ZERO-column view.

        ``torch.zeros``, not ``torch.empty``, and a fresh allocation every pass:
        ``block_residual`` is intra-forward scratch that must be re-zeroed each
        forward (M2 pins this as ``test_forward_twice_identical``), and in a
        server the untouched columns of a recycled buffer would otherwise hold
        another request's activations.  The memset is ~14 GiB once per prefill
        micro-batch against a multi-second prefill.

        :meth:`reset` runs FIRST so our reference to the previous pass's buffer
        is gone before the next one is allocated — otherwise the two are briefly
        co-live, which is the very doubling this class exists to remove.
        """
        cls.reset()
        buf = torch.zeros(int(num_tokens), int(num_columns), int(hidden),
                          dtype=dtype, device=device)
        cls._buf = buf
        cls._view = buf[:, :0]
        return cls._view

    @classmethod
    def append(cls, block_residual: torch.Tensor,
               column: torch.Tensor) -> torch.Tensor:
        """Append ``column`` ``(num_tokens, hidden)`` as the next boundary.

        Equivalent, value for value, to
        ``torch.cat([block_residual, column.unsqueeze(1)], dim=1)``.
        """
        if cls._view is None or block_residual is not cls._view:
            # Not our buffer: a caller that built its own block_residual and
            # never went through seed(). Same result, no preallocation win.
            return torch.cat([block_residual, column.unsqueeze(1)], dim=1)

        buf = cls._buf
        num_blocks = block_residual.shape[1]
        if num_blocks >= buf.shape[1]:
            raise RuntimeError(
                "block_residual overflow: boundary {} of a buffer sized for {} "
                "boundaries. num_block_residual_columns() disagrees with the "
                "layer stack actually being driven — the depth-attention would "
                "silently drop this block.".format(num_blocks + 1, buf.shape[1]))
        if column.shape != (buf.shape[0], buf.shape[2]):
            raise RuntimeError(
                "block_residual column shape {} does not fit the buffer's "
                "(num_tokens, hidden) = {}.".format(
                    tuple(column.shape), (buf.shape[0], buf.shape[2])))
        if column.dtype != buf.dtype:
            raise RuntimeError(
                "block_residual column dtype {} != buffer dtype {}; the cat "
                "this replaces would have refused too.".format(
                    column.dtype, buf.dtype))

        buf[:, num_blocks].copy_(column)
        cls._view = buf[:, :num_blocks + 1]
        return cls._view


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

    #: Number of block boundaries one pass crosses = the preallocated
    #: block_residual buffer's column count; set by the PSM alongside
    #: ``num_layers``.
    num_columns: Optional[int] = None

    _block_residual: Optional[torch.Tensor] = None
    _last_layer: Optional[int] = None

    @classmethod
    def configure(cls, num_layers: int, attn_res_block_size: int) -> None:
        cls.num_layers = int(num_layers)
        cls.num_columns = num_block_residual_columns(
            num_layers, attn_res_block_size)
        cls.reset()

    @classmethod
    def reset(cls) -> None:
        """Drop any parked scratch — the view AND the buffer behind it.

        "The pass is over" is one fact, so it has one switch.  Freeing only the
        carrier's view would leave :class:`BlockResidualBuffer` holding the
        whole ``(S, 8, H)`` allocation (14.00 GiB at S=131,072) as class state
        long after the prefill frame returned; see
        :meth:`BlockResidualBuffer.reset`.
        """
        cls._block_residual = None
        cls._last_layer = None
        BlockResidualBuffer.reset()

    @classmethod
    def peek(cls) -> Optional[torch.Tensor]:
        """Read the parked scratch without consuming it (tests/diagnostics)."""
        return cls._block_residual

    @classmethod
    def borrow(cls, layer_idx: int, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return the ``block_residual`` layer ``layer_idx`` must consume.

        Layer 0 RE-ZEROES it — ``block_residual`` is per-forward scratch, and
        carrying it across forwards is exactly the mutation the M2 suite pins
        (``test_forward_twice_identical``).  It also allocates the whole pass's
        :class:`BlockResidualBuffer` and returns its zero-column view, so the
        boundary appends never reallocate.
        """
        if layer_idx == 0:
            if cls.num_columns is None:
                raise RuntimeError(
                    "BlockResidualCarrier.configure(num_layers, "
                    "attn_res_block_size) was never called; the block_residual "
                    "buffer cannot be sized.")
            batch, seq_len, hidden = hidden_states.shape
            # Drop last pass's view AND buffer BEFORE seeding, so the old
            # allocation is not still live while the new one is made.
            cls.reset()
            cls._block_residual = BlockResidualBuffer.seed(
                batch * seq_len, hidden, cls.num_columns,
                dtype=hidden_states.dtype, device=hidden_states.device)
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
                "BlockResidualCarrier.configure(num_layers, "
                "attn_res_block_size) was never called; the output depth mix "
                "cannot verify the stack ran to the end.")
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
