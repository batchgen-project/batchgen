from dataclasses import dataclass
from transformers import AutoConfig, PretrainedConfig

@dataclass
class HuggingFaceAttentionConfig:
    """
    Configuration class for Hugging Face attention models.
    
    Attributes:
        model_name (str): The name of the attention model.
        revision (str): The revision of the attention model.
        cache_dir (str): Directory to cache the attention model.
    """
    hidden_size: int
    num_heads: int
    num_key_value_heads: int
    head_dim: int
    
    # deepseek_v3 specific attributes
    compressed_kv_dim: int = None

    def __init__(self, config: PretrainedConfig):
        if config.model_type == "qwen3_moe":
            self.hidden_size = config.hidden_size
            self.num_heads = config.num_attention_heads
            self.num_key_value_heads = config.num_key_value_heads
            self.head_dim = config.head_dim
        elif config.model_type == "deepseek_v3":
            self.hidden_size = config.hidden_size
            self.num_heads = config.n_attention_heads
            self.num_key_value_heads = config.n_key_value_heads
            self.head_dim = config.head_dim
            self.compressed_kv_dim = config.kv_lora_rank + config.qk_rope_head_dim,
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")
        
@dataclass
class HuggingFaceMoEConfig:
    """
    Configuration class for Hugging Face MoE models.
    
    Attributes:
        model_name (str): The name of the MoE model.
        revision (str): The revision of the MoE model.
        cache_dir (str): Directory to cache the MoE model.
    """
    num_experts: int
    topk: int
    shared_experts: bool
    hidden_size: int
    intermediate_size: int
    
    def __init__(self, config: PretrainedConfig):
        if config.model_type == "qwen3_moe":
            self.num_experts = config.num_experts
            self.topk = config.num_experts_per_tok
            self.shared_experts = False
            self.hidden_size = config.hidden_size
            self.intermediate_size = config.moe_intermediate_size
        elif config.model_type == "deepseek_v3":
            self.num_experts = config.n_routed_experts
            self.topk = config.num_experts_per_tok
            self.shared_experts = True
            self.hidden_size = config.hidden_size
            self.intermediate_size = config.moe_intermediate_size
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")
        

@dataclass
class HuggingFaceModelConfig:
    """
    Configuration class for Hugging Face models.
    
    Attributes:
        model_name (str): The name of the model.
        revision (str): The revision of the model.
        cache_dir (str): Directory to cache the model.
    """
    model_type: str
    num_layers: int
    attn_config: HuggingFaceAttentionConfig
    moe_config: HuggingFaceMoEConfig

    def __init__(self, config: PretrainedConfig):
        self.model_type = config.model_type
        self.num_layers = config.num_hidden_layers
        self.attn_config = HuggingFaceAttentionConfig(config)
        self.moe_config = HuggingFaceMoEConfig(config)