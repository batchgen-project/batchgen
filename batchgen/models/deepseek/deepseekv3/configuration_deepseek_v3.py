# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                             #
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

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

DEEPSEEK_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


# class DeepseekV3Config(PretrainedConfig):
#     r"""
#     This is the configuration class to store the configuration of a [`DeepseekV3Model`]. It is used to instantiate an DeepSeek
#     model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
#     defaults will yield a similar configuration to that of the DeepSeek-V3.

#     Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
#     documentation from [`PretrainedConfig`] for more information.


#     Args:
#         vocab_size (`int`, *optional*, defaults to 129280):
#             Vocabulary size of the Deep model. Defines the number of different tokens that can be represented by the
#             `inputs_ids` passed when calling [`DeepseekV3Model`]
#         hidden_size (`int`, *optional*, defaults to 4096):
#             Dimension of the hidden representations.
#         intermediate_size (`int`, *optional*, defaults to 11008):
#             Dimension of the MLP representations.
#         moe_intermediate_size (`int`, *optional*, defaults to 1407):
#             Dimension of the MoE representations.
#         num_hidden_layers (`int`, *optional*, defaults to 32):
#             Number of hidden layers in the Transformer decoder.
#         num_nextn_predict_layers (`int`, *optional*, defaults to 1):
#             Number of nextn predict layers in the DeepSeekV3 Model.
#         num_attention_heads (`int`, *optional*, defaults to 32):
#             Number of attention heads for each attention layer in the Transformer decoder.
#         n_shared_experts (`int`, *optional*, defaults to None):
#             Number of shared experts, None means dense model.
#         n_routed_experts (`int`, *optional*, defaults to None):
#             Number of routed experts, None means dense model.
#         routed_scaling_factor (`float`, *optional*, defaults to 1.0):
#             Scaling factor or routed experts.
#         topk_method (`str`, *optional*, defaults to `gready`):
#             Topk method used in routed gate.
#         n_group (`int`, *optional*, defaults to None):
#             Number of groups for routed experts.
#         topk_group (`int`, *optional*, defaults to None):
#             Number of selected groups for each token(for each token, ensuring the selected experts is only within `topk_group` groups).
#         num_experts_per_tok (`int`, *optional*, defaults to None):
#             Number of selected experts, None means dense model.
#         moe_layer_freq (`int`, *optional*, defaults to 1):
#             The frequency of the MoE layer: one expert layer for every `moe_layer_freq - 1` dense layers.
#         first_k_dense_replace (`int`, *optional*, defaults to 0):
#             Number of dense layers in shallow layers(embed->dense->dense->...->dense->moe->moe...->lm_head).
#                                                             \--k dense layers--/
#         norm_topk_prob (`bool`, *optional*, defaults to False):
#             Whether to normalize the weights of the routed experts.
#         scoring_func (`str`, *optional*, defaults to 'softmax'):
#             Method of computing expert weights.
#         aux_loss_alpha (`float`, *optional*, defaults to 0.001):
#             Auxiliary loss weight coefficient.
#         seq_aux = (`bool`, *optional*, defaults to True):
#             Whether to compute the auxiliary loss for each individual sample.
#         num_key_value_heads (`int`, *optional*):
#             This is the number of key_value heads that should be used to implement Grouped Query Attention. If
#             `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
#             `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
#             converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
#             by meanpooling all the original heads within that group. For more details checkout [this
#             paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
#             `num_attention_heads`.
#         hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
#             The non-linear activation function (function or string) in the decoder.
#         max_position_embeddings (`int`, *optional*, defaults to 2048):
#             The maximum sequence length that this model might ever be used with.
#         initializer_range (`float`, *optional*, defaults to 0.02):
#             The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
#         rms_norm_eps (`float`, *optional*, defaults to 1e-06):
#             The epsilon used by the rms normalization layers.
#         use_cache (`bool`, *optional*, defaults to `True`):
#             Whether or not the model should return the last key/values attentions (not used by all models). Only
#             relevant if `config.is_decoder=True`.
#         pad_token_id (`int`, *optional*):
#             Padding token id.
#         bos_token_id (`int`, *optional*, defaults to 1):
#             Beginning of stream token id.
#         eos_token_id (`int`, *optional*, defaults to 2):
#             End of stream token id.
#         pretraining_tp (`int`, *optional*, defaults to 1):
#             Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this
#             document](https://huggingface.co/docs/transformers/parallelism) to understand more about it. This value is
#             necessary to ensure exact reproducibility of the pretraining results. Please refer to [this
#             issue](https://github.com/pytorch/pytorch/issues/76232).
#         tie_word_embeddings (`bool`, *optional*, defaults to `False`):
#             Whether to tie weight embeddings
#         rope_theta (`float`, *optional*, defaults to 10000.0):
#             The base period of the RoPE embeddings.
#         rope_scaling (`Dict`, *optional*):
#             Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
#             strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
#             `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
#             `max_position_embeddings` to the expected new maximum.
#         attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
#             Whether to use a bias in the query, key, value and output projection layers during self-attention.
#         attention_dropout (`float`, *optional*, defaults to 0.0):
#             The dropout ratio for the attention probabilities.

#     ```python
#     >>> from transformers import DeepseekV3Model, DeepseekV3Config

#     >>> # Initializing a Deepseek-V3 style configuration
#     >>> configuration = DeepseekV3Config()

#     >>> # Accessing the model configuration
#     >>> configuration = model.config
#     ```"""

#     model_type = "deepseek_v3"
#     keys_to_ignore_at_inference = ["past_key_values"]
#     phase = None
#     _name_or_path = "deepseek-ai/DeepSeek-R1"

#     def __init__(
#         self,
#         vocab_size=129280,
#         hidden_size=7168,
#         intermediate_size=18432,
#         moe_intermediate_size=2048,
#         num_hidden_layers=61,
#         num_nextn_predict_layers=1,
#         num_attention_heads=128,
#         num_key_value_heads=128,
#         n_shared_experts=1,
#         n_routed_experts=256,
#         ep_size=1,
#         routed_scaling_factor=2.5,
#         kv_lora_rank=512,
#         q_lora_rank=1536,
#         qk_rope_head_dim=64,
#         v_head_dim=128,
#         qk_nope_head_dim=128,
#         topk_method="noaux_tc",
#         n_group=8,
#         topk_group=4,
#         num_experts_per_tok=8,
#         moe_layer_freq=1,
#         first_k_dense_replace=3,
#         norm_topk_prob=True,
#         scoring_func="sigmoid",
#         aux_loss_alpha=0.001,
#         seq_aux=True,
#         hidden_act="silu",
#         max_position_embeddings=163840,
#         initializer_range=0.02,
#         rms_norm_eps=1e-6,
#         use_cache=True,
#         pad_token_id=None,
#         bos_token_id=0,
#         eos_token_id=1,
#         pretraining_tp=1,
#         tie_word_embeddings=False,
#         rope_theta=10000.0,
#         rope_scaling=None,
#         attention_bias=False,
#         attention_dropout=0.0,
#         **kwargs,
#     ):
#         self.vocab_size = vocab_size
#         self.max_position_embeddings = max_position_embeddings
#         self.hidden_size = hidden_size
#         self.intermediate_size = intermediate_size
#         self.moe_intermediate_size = moe_intermediate_size
#         self.num_hidden_layers = num_hidden_layers
#         self.num_nextn_predict_layers = num_nextn_predict_layers
#         self.num_attention_heads = num_attention_heads
#         self.n_shared_experts = n_shared_experts
#         self.n_routed_experts = n_routed_experts
#         self.ep_size = ep_size
#         self.routed_scaling_factor = routed_scaling_factor
#         self.kv_lora_rank = kv_lora_rank
#         self.q_lora_rank = q_lora_rank
#         self.qk_rope_head_dim = qk_rope_head_dim
#         self.v_head_dim = v_head_dim
#         self.qk_nope_head_dim = qk_nope_head_dim
#         self.topk_method = topk_method
#         self.n_group = n_group
#         self.topk_group = topk_group
#         self.num_experts_per_tok = num_experts_per_tok
#         self.moe_layer_freq = moe_layer_freq
#         self.first_k_dense_replace = first_k_dense_replace
#         self.norm_topk_prob = norm_topk_prob
#         self.scoring_func = scoring_func
#         self.aux_loss_alpha = aux_loss_alpha
#         self.seq_aux = seq_aux
#         # for backward compatibility
#         if num_key_value_heads is None:
#             num_key_value_heads = num_attention_heads

#         self.num_key_value_heads = num_key_value_heads
#         self.hidden_act = hidden_act
#         self.initializer_range = initializer_range
#         self.rms_norm_eps = rms_norm_eps
#         self.pretraining_tp = pretraining_tp
#         self.use_cache = use_cache
#         self.rope_theta = rope_theta
#         self.rope_scaling = rope_scaling
#         self.attention_bias = attention_bias
#         self.attention_dropout = attention_dropout

#         self.phase = None
#         self._name_or_path = "deepseek-ai/DeepSeek-R1"
#         self.architectures = ["DeepseekV3ForCausalLM"]

#         super().__init__(
#             pad_token_id=pad_token_id,
#             bos_token_id=bos_token_id,
#             eos_token_id=eos_token_id,
#             tie_word_embeddings=tie_word_embeddings,
#             **kwargs,
#         )

def __init__(
    self,
    architectures=None,
    attention_bias=False,
    attention_dropout=0.0,
    auto_map=None,
    bos_token_id=0,
    eos_token_id=1,
    ep_size=1,
    first_k_dense_replace=3,
    hidden_act="silu",
    hidden_size=7168,
    initializer_range=0.02,
    intermediate_size=18432,
    kv_lora_rank=512,
    max_position_embeddings=163840,
    model_type="deepseek_v3",
    moe_intermediate_size=2048,
    moe_layer_freq=1,
    n_group=8,
    n_routed_experts=256,
    n_shared_experts=1,
    norm_topk_prob=True,
    num_attention_heads=128,
    num_experts_per_tok=8,
    num_hidden_layers=61,
    num_key_value_heads=128,
    num_nextn_predict_layers=1,
    q_lora_rank=1536,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    quantization_config=None,
    rms_norm_eps=1e-6,
    rope_scaling=None,
    rope_theta=10000.0,
    routed_scaling_factor=2.5,
    scoring_func="sigmoid",
    tie_word_embeddings=False,
    topk_group=4,
    topk_method="noaux_tc",
    torch_dtype=None,
    transformers_version=None,
    use_cache=True,
    v_head_dim=128,
    vocab_size=129280,
    # Additional parameters not in JSON but in original init
    aux_loss_alpha=0.001,
    seq_aux=True,
    pad_token_id=None,
    pretraining_tp=1,
    **kwargs,
):
    # Set attributes in the same order as JSON
    self.architectures = architectures or ["DeepseekV3ForCausalLM"]
    self.attention_bias = attention_bias
    self.attention_dropout = attention_dropout
    self.auto_map = auto_map or {
        "AutoConfig": "configuration_deepseek.DeepseekV3Config",
        "AutoModel": "modeling_deepseek.DeepseekV3Model",
        "AutoModelForCausalLM": "modeling_deepseek.DeepseekV3ForCausalLM"
    }
    self.bos_token_id = bos_token_id
    self.eos_token_id = eos_token_id
    self.ep_size = ep_size
    self.first_k_dense_replace = first_k_dense_replace
    self.hidden_act = hidden_act
    self.hidden_size = hidden_size
    self.initializer_range = initializer_range
    self.intermediate_size = intermediate_size
    self.kv_lora_rank = kv_lora_rank
    self.max_position_embeddings = max_position_embeddings
    self.model_type = model_type
    self.moe_intermediate_size = moe_intermediate_size
    self.moe_layer_freq = moe_layer_freq
    self.n_group = n_group
    self.n_routed_experts = n_routed_experts
    self.n_shared_experts = n_shared_experts
    self.norm_topk_prob = norm_topk_prob
    self.num_attention_heads = num_attention_heads
    self.num_experts_per_tok = num_experts_per_tok
    self.num_hidden_layers = num_hidden_layers
    self.num_key_value_heads = num_key_value_heads
    self.num_nextn_predict_layers = num_nextn_predict_layers
    self.q_lora_rank = q_lora_rank
    self.qk_nope_head_dim = qk_nope_head_dim
    self.qk_rope_head_dim = qk_rope_head_dim
    self.quantization_config = quantization_config
    self.rms_norm_eps = rms_norm_eps
    self.rope_scaling = rope_scaling or {
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 40,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "original_max_position_embeddings": 4096,
        "type": "yarn"
    }
    self.rope_theta = rope_theta
    self.routed_scaling_factor = routed_scaling_factor
    self.scoring_func = scoring_func
    self.tie_word_embeddings = tie_word_embeddings
    self.topk_group = topk_group
    self.topk_method = topk_method
    self.torch_dtype = torch_dtype
    self.transformers_version = transformers_version
    self.use_cache = use_cache
    self.v_head_dim = v_head_dim
    self.vocab_size = vocab_size
    
    # Additional attributes from original class not in JSON
    self.aux_loss_alpha = aux_loss_alpha
    self.seq_aux = seq_aux
    self.pad_token_id = pad_token_id
    self.pretraining_tp = pretraining_tp
    
    # Class-level attributes from original
    self.phase = None
    self._name_or_path = "deepseek-ai/DeepSeek-R1"
    
    # Call parent init
    super().__init__(
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        tie_word_embeddings=tie_word_embeddings,
        **kwargs,
    )