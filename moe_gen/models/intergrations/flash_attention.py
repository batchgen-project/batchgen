
from functools import lru_cache
import importlib
import inspect
import os
from sys import version
import torch
from typing import Optional, Tuple

from transformers.utils import is_flash_attn_2_available, is_flash_attn_greater_or_equal_2_10, is_torch_npu_available
from transformers import logging
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# TODO: This doesn't work for all packages (`bs4`, `faiss`, etc.) Talk to Sylvain to see how to do with it better.
def _is_package_available(pkg_name: str, return_version: bool = False) -> Union[Tuple[bool, str], bool]:
    # Check if the package spec exists and grab its version to avoid importing a local directory
    package_exists = importlib.util.find_spec(pkg_name) is not None
    package_version = "N/A"
    if package_exists:
        try:
            # TODO: Once python 3.9 support is dropped, `importlib.metadata.packages_distributions()`
            # should be used here to map from package name to distribution names
            # e.g. PIL -> Pillow, Pillow-SIMD; quark -> amd-quark; onnxruntime -> onnxruntime-gpu.
            # `importlib.metadata.packages_distributions()` is not available in Python 3.9.

            # Primary method to get the package version
            package_version = importlib.metadata.version(pkg_name)
        except importlib.metadata.PackageNotFoundError:
            # Fallback method: Only for "torch" and versions containing "dev"
            if pkg_name == "torch":
                try:
                    package = importlib.import_module(pkg_name)
                    temp_version = getattr(package, "__version__", "N/A")
                    # Check if the version contains "dev"
                    if "dev" in temp_version:
                        package_version = temp_version
                        package_exists = True
                    else:
                        package_exists = False
                except ImportError:
                    # If the package can't be imported, it's not available
                    package_exists = False
            elif pkg_name == "quark":
                # TODO: remove once `importlib.metadata.packages_distributions()` is supported.
                try:
                    package_version = importlib.metadata.version("amd-quark")
                except Exception:
                    package_exists = False
            else:
                # For packages other than "torch", don't attempt the fallback and set as not available
                package_exists = False
        logger.debug(f"Detected {pkg_name} version: {package_version}")
    if return_version:
        return package_exists, package_version
    else:
        return package_exists

@lru_cache()
def is_flash_attn_greater_or_equal(library_version: str):
    if not _is_package_available("flash_attn"):
        return False

    return version.parse(importlib.metadata.version("flash_attn")) >= version.parse(library_version)

if flash_attn_func:
    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)


flash_241 = is_flash_attn_greater_or_equal("2.4.1")
deterministic_g = os.environ.get("FLASH_ATTENTION_DETERMINISTIC", "0") == "1"

def flash_attn_supports_top_left_mask():
    """Determine whether flash-attention uses top-left or down-right mask"""

    if is_flash_attn_2_available():
        # top-left mask is used in package `flash-attn` with version lower than 2.1.0
        return not is_flash_attn_greater_or_equal_2_10()

    return False

flash_attn_func = None


def fa_peft_integration_check(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    target_dtype: Optional[torch.dtype] = None,
):
    """
    PEFT usually casts the layer norms in float32 for training stability reasons
    therefore the input hidden states gets silently casted in float32. Hence, we need
    cast them back in float16 / bfloat16 just to be sure everything works as expected.
    This might slowdown training & inference so it is recommended to not cast the LayerNorms!

    Args:
        query (`torch.Tensor`):
            Input query states to be passed to Flash Attention API
        key (`torch.Tensor`):
            Input key states to be passed to Flash Attention API
        value (`torch.Tensor`):
            Input value states to be passed to Flash Attention API
        target_dtype (`torch.dtype`, *optional*):
            The dtype to convert the attention tensors to. Conversion can be ignored by
            not providing the target dtype.
    """
    if target_dtype is None:
        return query, key, value

    input_dtype = query.dtype
    if input_dtype == torch.float32:
        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        query = query.to(target_dtype)
        key = key.to(target_dtype)
        value = value.to(target_dtype)

    return query, key, value


if is_flash_attn_2_available():
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.layers.rotary import apply_rotary_emb  # noqa


def prepare_fa2_from_position_ids(query, key, value, position_ids):
    """
    This function returns necessary arguments to call `flash_attn_varlen_func`.
    All three query, key, value states will be flattened.
    Cumulative lengths of each examples in the batch will be extracted from position_ids.

    NOTE: ideally cumulative lengths should be prepared at the data collator stage

    Arguments:
        query (`torch.Tensor`):
            Query state with padding. Shape: (batch_size, query_length, num_heads, head_dim).
        key (`torch.Tensor`):
            Key state with padding. Shape: (batch_size, kv_seq_len, num_key_value_heads, head_dim).
        value (`torch.Tensor`):
            Value state with padding. Shape: (batch_size, kv_seq_len, num_key_value_heads, head_dim).
        position_ids (`torch.Tensor`):
            Boolean or int tensor of shape (batch_size, sequence_length), 1 means valid and 0 means not valid.

    Return:
        query (`torch.Tensor`):
            Query state without padding. Shape: (total_target_length, num_heads, head_dim).
        key (`torch.Tensor`):
            Key state with padding. Shape: (total_source_length, num_key_value_heads, head_dim).
        value (`torch.Tensor`):
            Value state with padding. Shape: (total_source_length, num_key_value_heads, head_dim).
        indices_q (`torch.Tensor`):
            The indices of non-masked tokens from the flattened input target sequence.
        (cu_seqlens_q, cu_seqlens_k) (`Tuple[int]`):
            The cumulative sequence lengths for the target (query) and source (key, value), used to index into ragged (unpadded) tensors. `cu_seqlens` shape is (batch_size + 1,).
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k) (`Tuple[int]`):
            Maximum sequence length in batch (`max_seqlen_in_batch_q` for the target sequence i.e. query, `max_seqlen_in_batch_k` for the source sequence i.e. key/value).
    """
    query = query.view(-1, query.size(-2), query.size(-1))
    key = key.contiguous().view(-1, key.size(-2), key.size(-1))
    value = value.contiguous().view(-1, value.size(-2), value.size(-1))
    position_ids = position_ids.flatten()
    indices_q = torch.arange(position_ids.size(0), device=position_ids.device, dtype=torch.int32)

    cu_seq_lens = torch.cat(
        (
            indices_q[position_ids == 0],
            torch.tensor(position_ids.size(), device=position_ids.device, dtype=torch.int32),
        )
    )

    max_length = position_ids.max() + 1

    return (query, key, value, indices_q, (cu_seq_lens, cu_seq_lens), (max_length, max_length))


def _flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    query_length: int,
    is_causal: bool,
    dropout: float = 0.0,
    position_ids: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    softcap: Optional[float] = None,
    deterministic: Optional[bool] = None,
    cu_seq_lens_q: Optional[torch.LongTensor] = None,
    cu_seq_lens_k: Optional[torch.LongTensor] = None,
    max_length_q: Optional[int] = None,
    max_length_k: Optional[int] = None,
    target_dtype: Optional[torch.dtype] = None,
    **kwargs,
):
    """
    Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
    first unpad the input, then computes the attention scores and pad the final attention scores.

    Args:
        query_states (`torch.Tensor`):
            Input query states to be passed to Flash Attention API
        key_states (`torch.Tensor`):
            Input key states to be passed to Flash Attention API
        value_states (`torch.Tensor`):
            Input value states to be passed to Flash Attention API
        attention_mask (`torch.Tensor`, *optional*):
            The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
            position of padding tokens and 1 for the position of non-padding tokens.
        dropout (`float`):
            Attention dropout
        softmax_scale (`float`, *optional*):
            The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        use_top_left_mask (`bool`, defaults to `False`):
            flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignment, that was made default for flash_attn>=2.1. This attribute is used to handle this difference.
        softcap (`float`, *optional*):
            Softcap for the attention logits, used e.g. in gemma2.
        deterministic (`bool`, *optional*):
            Determines if the deterministic option introduced in flash_attn>=2.4.1 is enabled.
    """
    if not use_top_left_mask:
        causal = is_causal
    else:
        # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1.
        causal = is_causal and query_length != 1

    # Assuming 4D tensors, key_states.shape[1] is the key/value sequence length (source length).
    use_sliding_windows = (
        _flash_supports_window_size and sliding_window is not None and key_states.shape[1] > sliding_window
    )
    flash_kwargs = {"window_size": (sliding_window, sliding_window)} if use_sliding_windows else {}

    if flash_241:
        if deterministic is None:
            deterministic = deterministic_g
        flash_kwargs["deterministic"] = deterministic

    if softcap is not None:
        flash_kwargs["softcap"] = softcap

    # PEFT possibly silently casts tensors to fp32, this potentially reconverts to correct dtype or is a no op
    query_states, key_states, value_states = fa_peft_integration_check(
        query_states, key_states, value_states, target_dtype
    )

    # Contains at least one padding token in the sequence
    if attention_mask is not None:
        batch_size = query_states.shape[0]
        query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = _upad_input(
            query_states, key_states, value_states, attention_mask, query_length
        )
        cu_seqlens_q, cu_seqlens_k = cu_seq_lens
        max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

        attn_output_unpad = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_in_batch_q,
            max_seqlen_k=max_seqlen_in_batch_k,
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=causal,
            **flash_kwargs,
        )
        attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)

    # If position_ids is provided and check all examples do not contain only 1 sequence, If tensor in increasing
    # then we probably have one sequence, otherwise it is packed. Additionally check we are in pre-fill/training stage.
    # Use `flash_attn_varlen_func` to prevent cross-example attention and also allow padding free approach
    elif position_ids is not None and (
        max_length_q is not None or (query_length != 1 and not (torch.diff(position_ids, dim=-1) >= 0).all())
    ):
        batch_size = query_states.size(0)

        if cu_seq_lens_q is None or cu_seq_lens_k is None:
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = (
                prepare_fa2_from_position_ids(query_states, key_states, value_states, position_ids)
            )

            cu_seq_lens_q, cu_seq_lens_k = cu_seq_lens
            max_length_q, max_length_k = max_seq_lens

        else:
            query_states = query_states.reshape(-1, query_states.size(-2), query_states.size(-1))
            key_states = key_states.reshape(-1, key_states.size(-2), key_states.size(-1))
            value_states = value_states.reshape(-1, value_states.size(-2), value_states.size(-1))

        attn_output = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seq_lens_q,
            cu_seqlens_k=cu_seq_lens_k,
            max_seqlen_q=max_length_q,
            max_seqlen_k=max_length_k,
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=causal,
            **flash_kwargs,
        )

        attn_output = attn_output.view(batch_size, -1, attn_output.size(-2), attn_output.size(-1))

    else:
        attn_output = flash_attn_func(
            query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal, **flash_kwargs
        )

    return attn_output
