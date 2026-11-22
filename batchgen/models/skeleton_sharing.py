"""One device copy of a model skeleton, shared by both phase instances.

A manager that keeps BOTH phase instances builds one model per phase, and
each build binds its skeleton entries (embeddings, norms, router/gates,
LM head) by ``param.data = skeleton_state_dict[key].to(device)``. Moving the
bound entries to the device ONCE makes that ``.to`` an identity for every
later build, so the two phase models alias one device copy instead of
holding one each — which is what lets a suspended instance stay kept without
pinning a second skeleton through the other phase.

This is the generic core of GLM-5's ``share_skeleton_on_device``; the
GLM-specific parts (bind-key remapping, the FP8 scale upload set, bucketed
logging) stay in its manager, and GPT-OSS shares the skeleton through the
same function.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch


def share_state_dict_on_device(
    state_dict: Dict[str, torch.Tensor],
    upload_keys: Iterable[str],
    device,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int], Dict[str, int]]:
    """Move the named entries of ``state_dict`` to ``device``, once.

    Returns ``(shared, uploaded, kept)``: the new mapping, the byte count per
    uploaded key, and the byte count per host-kept key. Entries outside
    ``upload_keys`` stay exactly as they are (host-resident entries no
    instance binds). An entry already on ``device`` is aliased, not copied,
    so calling this again with the same keys is a no-op.
    """
    wanted = set(upload_keys)
    shared: Dict[str, torch.Tensor] = {}
    uploaded: Dict[str, int] = {}
    kept: Dict[str, int] = {}
    for key, tensor in state_dict.items():
        nbytes = tensor.numel() * tensor.element_size()
        if key in wanted:
            shared[key] = tensor.to(device)
            uploaded[key] = nbytes
        else:
            shared[key] = tensor
            kept[key] = nbytes
    return shared, uploaded, kept
