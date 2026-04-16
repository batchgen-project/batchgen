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

"""GLM-5 HuggingFace-style PretrainedConfig for checkpoint loading.

This mirrors the config.json structure from zai-org/GLM-5-FP8.
Used by model.py to instantiate the model with correct dimensions.
"""

from transformers.configuration_utils import PretrainedConfig


class Glm5Config(PretrainedConfig):
    model_type = "glm_moe_dsa"

    def __init__(
        self,
        vocab_size=154880,
        hidden_size=6144,
        intermediate_size=12288,
        moe_intermediate_size=2048,
        num_hidden_layers=78,
        num_attention_heads=64,
        num_key_value_heads=64,
        head_dim=64,
        qk_head_dim=256,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        q_lora_rank=2048,
        kv_lora_rank=512,
        rope_theta=1000000.0,
        rope_interleave=True,
        indexer_rope_interleave=True,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        n_routed_experts=256,
        n_shared_experts=1,
        num_experts_per_tok=8,
        first_k_dense_replace=3,
        moe_layer_freq=1,
        n_group=1,
        topk_group=1,
        topk_method="noaux_tc",
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
        scoring_func="sigmoid",
        index_n_heads=32,
        index_head_dim=128,
        index_topk=2048,
        use_dense_mla=False,
        hidden_act="silu",
        max_position_embeddings=202752,
        rms_norm_eps=1e-5,
        initializer_range=0.02,
        tie_word_embeddings=False,
        num_nextn_predict_layers=1,
        ep_size=1,
        pad_token_id=154820,
        bos_token_id=None,
        eos_token_id=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.qk_head_dim = qk_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.rope_theta = rope_theta
        self.rope_interleave = rope_interleave
        self.indexer_rope_interleave = indexer_rope_interleave
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.first_k_dense_replace = first_k_dense_replace
        self.moe_layer_freq = moe_layer_freq
        self.n_group = n_group
        self.topk_group = topk_group
        self.topk_method = topk_method
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.scoring_func = scoring_func
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        # Structurally disable DSA indexer (clean dense-MLA mode). Honors
        # env var BATCHGEN_GLM5_USE_DENSE_MLA=1 even when not set via config,
        # so launching with the env flag alone is sufficient to switch paths.
        import os as _os_glm5hf
        self.use_dense_mla = bool(use_dense_mla) or (
            _os_glm5hf.environ.get("BATCHGEN_GLM5_USE_DENSE_MLA", "0") == "1"
        )
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.ep_size = ep_size

        # Derived
        self.num_local_experts = n_routed_experts
        self.compressed_kv_dim = kv_lora_rank + qk_rope_head_dim  # 576

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
