# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  Copyright (c) 2025-2026 BatchGen Team                                            #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
#  you may not use this file except in compliance with the license.             #
#                                                                               #
#  you may obtain a copy of the license at                                      #
#                                                                               #
#                  http://www.apache.org/licenses/license-2.0                   #
#                                                                               #
#  unless required by applicable law or agreed to in writing, software          #
#  distributed under the license is distributed on an "as is" basis,            #
#  without warranties or conditions of any kind, either express or implied.     #
#  see the license for the specific language governing permissions and          #
#  limitations under the license.                                               #
# ---------------------------------------------------------------------------- #

"""Internal first-N-layer truncation, resolved ONCE at config parsing.

``--debug-num-layers N`` serves only the first N transformer layers of the
checkpoint, for model bring-up iteration: startup loads a fraction of the
weights while every kernel and scheduling path still runs. Outputs are
meaningless.

Design contract (the one mechanism, no per-model branches):

* Every model initializer calls :func:`apply_layer_truncation` exactly once,
  at the point its rich model config is final and before ANY consumer reads
  it — the engine ``ModelConfig`` projection, module shape tables, KV/state
  pool sizing, planners, parameter servers and graph capture all read
  ``num_hidden_layers`` off that one config object, so the truncated value
  propagates without further plumbing.
* The call is unconditional; ``debug_num_layers=None`` is the production path
  and changes no consumed value. It only stamps ``num_hidden_layers_full``
  (the checkpoint's count) so the [SERVER_CONFIG] snapshot can always report
  effective vs full.
* "First N layers" means checkpoint layers ``0..N-1`` keeping their
  checkpoint-declared kinds. Per-layer tables carried by the rich config are
  sliced to that prefix here, and only here:
    - ``indexer_types`` (GLM-5.2/5.3: per-layer "full"/"shared" DSA list,
      length-checked against ``num_hidden_layers`` by
      ``assert_indexer_schedule_consistent``);
    - ``linear_attn_config["kda_layers"]`` / ``["full_attn_layers"]``
      (Kimi-Linear/K3: 1-indexed layer lists, partition-checked against
      ``1..num_hidden_layers`` by ``validate_k3_config``);
    - ``dsa_config["dsa_layers"]`` (Kimi-K3.1: 1-indexed, required to equal
      ``full_attn_layers`` and to stay inside ``1..num_hidden_layers`` by
      ``validate_k3_config`` / ``parse_dsa_config``).
  Index-formula patterns (GLM's ``index_topk_freq`` schedule, MoE
  ``first_k_dense_replace``) need no slicing: they are evaluated per layer
  index below ``num_hidden_layers``.
* A worker whose model never routed through this module refuses the flag by
  name (:func:`require_layer_truncation_applied`) instead of silently serving
  all layers.

This module is pure Python (no torch / engine imports) so it can be unit
tested headlessly, like ``batchgen_model_config``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import List, Optional

logger = logging.getLogger(__name__)

# Stamped on the rich config by apply_layer_truncation: the checkpoint's full
# layer count, whatever num_hidden_layers now says.
FULL_LAYER_COUNT_FIELD = "num_hidden_layers_full"


class LayerTruncationError(ValueError):
    """``--debug-num-layers`` cannot be satisfied as requested."""


def apply_layer_truncation(config, debug_num_layers: Optional[int]):
    """Resolve the effective layer count on the rich model config, once.

    Args:
        config: The model's rich config object, final in every other respect.
            Mutated in place; also returned for call-site convenience.
        debug_num_layers: ``--debug-num-layers`` as the server received it;
            None = all layers (the production path).

    Raises:
        LayerTruncationError: N is not in ``[1, num_hidden_layers]``.
    """
    full = int(config.num_hidden_layers)
    setattr(config, FULL_LAYER_COUNT_FIELD, full)
    if debug_num_layers is None:
        return config

    n = int(debug_num_layers)
    model_type = getattr(config, "model_type", "?")
    if n < 1:
        raise LayerTruncationError(
            f"--debug-num-layers must be a positive layer count, got {n}"
        )
    if n > full:
        raise LayerTruncationError(
            f"--debug-num-layers {n} exceeds the checkpoint's "
            f"num_hidden_layers={full} (model_type={model_type!r}); the flag "
            "only truncates, it never extends"
        )

    config.num_hidden_layers = n

    # Per-layer tables keep a consistent first-N prefix (see module docstring
    # for why exactly these are sliced).
    indexer_types = getattr(config, "indexer_types", None)
    if indexer_types is not None:
        config.indexer_types = list(indexer_types)[:n]
    lac = getattr(config, "linear_attn_config", None)
    if lac:
        for key in ("kda_layers", "full_attn_layers"):
            layers = lac.get(key)
            if layers:
                # 1-indexed layer numbers; keep those inside the prefix.
                lac[key] = [layer for layer in layers if int(layer) <= n]
    dsa = getattr(config, "dsa_config", None)
    if dsa and dsa.get("dsa_layers"):
        # 1-indexed like linear_attn_config's lists; validate_k3_config
        # requires dsa_layers == full_attn_layers, so both slice identically.
        # A prefix with no DSA layer leaves the list empty, which
        # parse_dsa_config refuses by name — a K3.1 truncation must keep at
        # least one indexer-gated layer.
        dsa["dsa_layers"] = [
            layer for layer in dsa["dsa_layers"] if int(layer) <= n
        ]

    # Re-check the config's own invariants over the truncated depth: a rich
    # config that declares validate() (e.g. GLM5Config refuses
    # first_k_dense_replace >= num_hidden_layers) must hold for the served
    # prefix too, and its own named error is the right refusal.
    validate = getattr(config, "validate", None)
    if callable(validate):
        validate()

    logger.warning(
        "[LAYER_TRUNCATION] serving the first %d of %d layers "
        "(model_type=%r): outputs are meaningless; this run must not be "
        "recorded as an official result",
        n,
        full,
        model_type,
    )
    return config


def require_layer_truncation_applied(
    model_name: str, loaded_model_config, debug_num_layers: Optional[int]
) -> None:
    """Refuse ``--debug-num-layers`` on a model that did not resolve it.

    Called by the worker after the initializer returns: an initializer that
    never routed its rich config through :func:`apply_layer_truncation` has
    built a full-depth model, and serving it as a truncated run would be a
    silent fallback. Hard-fails by name instead.
    """
    if debug_num_layers is None:
        return
    stamped = getattr(loaded_model_config, FULL_LAYER_COUNT_FIELD, None)
    effective = getattr(loaded_model_config, "num_hidden_layers", None)
    if stamped is None or effective is None or int(effective) != int(
        debug_num_layers
    ):
        raise LayerTruncationError(
            f"--debug-num-layers {debug_num_layers} is not supported by "
            f"model {model_name!r}: its initializer did not resolve the "
            "effective layer count through "
            "batchgen.config.layer_truncation.apply_layer_truncation "
            f"(num_hidden_layers={effective!r}, "
            f"{FULL_LAYER_COUNT_FIELD}={stamped!r})"
        )


# Converted checkpoints name a layer's tensors ``model.layers.<i>.<rest>``.
_LAYER_TENSOR = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def select_weight_files(converted_ckpt_dir: str, config) -> List[str]:
    """The converted-checkpoint files the host weight load must read.

    ``config`` is the rich config after ``apply_layer_truncation``. With all
    layers served the result is empty, which the native loader reads as "every
    file of the directory". Under truncation a file is selected when it holds
    at least one tensor the served prefix needs: a tensor of a layer below the
    served count, or a tensor that is not a layer tensor of the checkpoint
    (embeddings, final norm, head). A file whose tensors all belong to layers
    that are not served is never opened.

    Returns:
        ``<name>.json`` file names inside ``converted_ckpt_dir``, sorted.

    Raises:
        LayerTruncationError: truncation is on and no file was selected, or a
            metadata file has no ``state_dict``.
    """
    served = int(config.num_hidden_layers)
    full = int(getattr(config, FULL_LAYER_COUNT_FIELD, served))
    if served == full:
        return []

    selected = []
    names = sorted(
        name for name in os.listdir(converted_ckpt_dir) if name.endswith(".json")
    )
    for name in names:
        with open(os.path.join(converted_ckpt_dir, name)) as handle:
            state_dict = json.load(handle).get("state_dict")
        if not isinstance(state_dict, dict):
            raise LayerTruncationError(
                f"--debug-num-layers: {name} in {converted_ckpt_dir} has no "
                "state_dict; it is not converted-checkpoint metadata"
            )
        for tensor_name in state_dict:
            match = _LAYER_TENSOR.search(tensor_name)
            layer = int(match.group(1)) if match else None
            if layer is None or layer < served or layer >= full:
                selected.append(name)
                break
    if not selected:
        raise LayerTruncationError(
            f"--debug-num-layers {served}: no converted-checkpoint file in "
            f"{converted_ckpt_dir} holds a tensor of the served layers"
        )
    logger.warning(
        "[LAYER_TRUNCATION] host weight load reads %d of %d converted files",
        len(selected),
        len(names),
    )
    return selected


_WEIGHT_FILE_ALIGNMENT = 4096          # the native loader starts each file on 4 KiB
_RESERVATION_ALIGNMENT = 2 * 1024 * 1024


def weight_reservation_bytes(
    full_byte_size: int, converted_ckpt_dir: str, selected: List[str]
) -> int:
    """Host weight reservation for the files ``select_weight_files`` chose.

    With no selection (all layers served) the model's own reservation stands.
    Under truncation the reservation is the footprint of the selected files,
    laid out as the native loader lays them out (each file on a 4 KiB
    boundary), rounded up to 2 MiB: every worker registers the whole
    reservation with its device, so reserving the full checkpoint for a
    truncated load would keep that start-up cost at the full model's.
    """
    if not selected:
        return int(full_byte_size)
    total = 0
    for name in selected:
        with open(os.path.join(converted_ckpt_dir, name)) as handle:
            size = int(json.load(handle)["total_byte_size"])
        total = -(-total // _WEIGHT_FILE_ALIGNMENT) * _WEIGHT_FILE_ALIGNMENT + size
    total = -(-total // _RESERVATION_ALIGNMENT) * _RESERVATION_ALIGNMENT
    if total > int(full_byte_size):
        raise LayerTruncationError(
            f"--debug-num-layers: the selected weight files need {total} bytes, "
            f"more than the model's full reservation of {int(full_byte_size)}"
        )
    return total
