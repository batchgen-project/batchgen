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

"""Base planner class for model-specific configuration planning.

All behaviors controlled by config flags. Worker reads from config,
never uses hardcoded values.
"""

from abc import ABC, abstractmethod
import logging

from batchgen.config.config import EngineConfig


class BasePlanner(ABC):
    """Base class for model-specific planners.

    Planners decide config values, workers execute based on those configs.
    This separation ensures all worker behaviors are controlled by flags.
    """

    # Subclasses MUST override these class attributes
    MAGIC_NUM: int = NotImplemented
    DEFAULT_MEM_FRAC: float = NotImplemented
    NUM_EXPERTS: int = NotImplemented

    def __init__(self):
        """Initialize planner. Config parameters set via generate_config()."""
        pass

    def generate_config(self, config: EngineConfig) -> EngineConfig:
        """Generate complete config for the model.

        Args:
            config: EngineConfig with Basic_Config already populated
                   (device, dtypes, padding_length, etc. from set_basic_config)

        Returns:
            EngineConfig with all sections populated
        """
        self.config = config
        self.max_prompt_length = config.Basic_Config.padding_length
        self.max_response_length = config.Basic_Config.max_decoding_length
        self.max_context_length = self.max_prompt_length + self.max_response_length
        self.world_size = config.Basic_Config.world_size

        # Set default configs (common + model-specific)
        self._set_default_configs()

        # Compute batch sizes and buffer configs
        self._compute_batch_configs()

        # Model-specific adjustments
        self._adjust_config_for_model()

        return self.config

    def _set_default_configs(self):
        """Set common default configs. Subclasses can override."""
        # Token-based prefill (for prepack mode, always recommended)
        self.config.Module_Batching_Config.prefill_micro_batch_token_cap = 120_000
        self.config.Module_Batching_Config.prepack_row_capacity = None

        # Sequence-based prefill (for non-prepack mode)
        self.config.Module_Batching_Config.attn_prefill_micro_batch_size = 8
        self.config.Module_Batching_Config.MoE_prefill_micro_batch_size = 8
        self.config.Module_Batching_Config.expert_prefill_batch_size_upper_bound = 4096

        self.config.Module_Batching_Config.attn_decoding_micro_batch_size = None
        self.config.Module_Batching_Config.MoE_decoding_micro_batch_size = None
        self.config.Module_Batching_Config.expert_decoding_batch_size_upper_bound = 2048

        # Default GPU Buffer Config
        self.config.GPU_Buffer_Config.num_prefill_module_buffer = {
            "attn": 1,
            "routed_expert": 8,
            "shared_expert": 1
        }
        self.config.GPU_Buffer_Config.num_decoding_module_buffer = {
            "attn": 1,
            "routed_expert": None,  # Set later based on available memory
            "shared_expert": 1
        }
        self.config.GPU_Buffer_Config.num_k_buffer = None
        self.config.GPU_Buffer_Config.num_v_buffer = 0

        # Default EP Config
        self.config.EP_Config.enable = True
        self.config.EP_Config.num_local_expert_per_layer = None

    def _compute_batch_configs(self):
        """Compute batch sizes based on model-specific constants."""
        kv_element_size = 2 if self.config.Basic_Config.kv_dtype == "bfloat16" else 1

        expert_per_rank = self.NUM_EXPERTS // self.world_size
        assert expert_per_rank > 0, "EXPERT_PER_RANK must be greater than 0"

        # Set attention mode based on world size and EP offloading
        # attn_mode = 3 uses decoding_continuous() which properly handles dynamic weight loading
        # attn_mode = 1 uses legacy modes that don't support EP with offloading
        if self.world_size > 8:
            self.config.Basic_Config.attn_mode = 3  # DUAL-NODE
        elif self.config.EP_Config.enable_offloading:
            # EP with offloading on single-node requires attn_mode = 3 for decoding_continuous()
            self.config.Basic_Config.attn_mode = 3
            logging.info("EP offloading enabled on single-node: setting attn_mode=3 for decoding_continuous()")
        else:
            self.config.Basic_Config.attn_mode = 1  # SINGLE-NODE (legacy)

        # Compute decoding micro batch size using model-specific MAGIC_NUM
        attn_decoding_micro_batch_size = self.MAGIC_NUM // self.max_prompt_length
        # Round down to nearest power of 2
        attn_decoding_micro_batch_size = 2 ** (attn_decoding_micro_batch_size.bit_length() - 1)

        num_k_buffer = 6
        k_buffer_size = (
            num_k_buffer * attn_decoding_micro_batch_size *
            self.max_context_length * 576 / (1024 ** 3) * kv_element_size
        )
        self.config.GPU_Buffer_Config.kv_buffer_num_tokens = (
            attn_decoding_micro_batch_size * self.max_context_length
        )

        # Memory calculations (assuming 96GB GPU)
        available_gpu_mem = 96 * self.DEFAULT_MEM_FRAC
        model_skeleton_size = 6
        cuda_page_table_default_size = 5
        nccl_default_buffer_usage = 2.5

        non_static_memory_usage = k_buffer_size + model_skeleton_size + cuda_page_table_default_size
        available_memory_for_expert_cache = available_gpu_mem - non_static_memory_usage

        # Check if EP offloading is enabled - if so, use offloading_ratio instead of memory-based calculation
        if self.config.EP_Config.enable_offloading and self.config.EP_Config.offloading_ratio > 0:
            # EP offloading: use offloading_ratio to determine persistent vs offloaded experts
            num_local_expert_per_layer = int(expert_per_rank * (1 - self.config.EP_Config.offloading_ratio))
            # Buffer count = number of offloaded experts + overhead
            num_offloaded = expert_per_rank - num_local_expert_per_layer
            num_decoding_module_buffer_routed_expert = num_offloaded + 2
            logging.info(
                f"EP offloading enabled: {num_local_expert_per_layer} persistent, "
                f"{num_offloaded} offloaded, {num_decoding_module_buffer_routed_expert} buffers"
            )
        else:
            # Default: calculate based on available GPU memory
            num_local_expert_per_layer = min(
                expert_per_rank,
                int(available_memory_for_expert_cache // 2.4)
            )
            num_decoding_module_buffer_routed_expert = expert_per_rank - num_local_expert_per_layer + 2

        if self.config.Basic_Config.attn_mode == 3:
            # For attn_mode=3 (dual-node or EP offloading), calculate MoE decoding batch size
            if not self.config.EP_Config.enable_offloading:
                # Dual-node mode: all experts persistent on GPU
                num_local_expert_per_layer = expert_per_rank
            # EP offloading mode: keep num_local_expert_per_layer from offloading calculation above

            expert_size = num_local_expert_per_layer * 2.4
            per_seq_size = self.max_context_length * 61 * 576 / (1024 ** 3) * kv_element_size
            self.config.Module_Batching_Config.MoE_decoding_micro_batch_size = int(
                (available_gpu_mem - model_skeleton_size - cuda_page_table_default_size -
                 expert_size - nccl_default_buffer_usage) / per_seq_size
            )
            logging.info(
                f"Max Available MoE decoding micro batch size: "
                f"{self.config.Module_Batching_Config.MoE_decoding_micro_batch_size}"
            )

        # Only zero out buffers if ALL experts are persistent (no offloading)
        if num_local_expert_per_layer == expert_per_rank and not self.config.EP_Config.enable_offloading:
            num_decoding_module_buffer_routed_expert = 0

        # Update config
        self.config.Module_Batching_Config.attn_decoding_micro_batch_size = attn_decoding_micro_batch_size
        self.config.GPU_Buffer_Config.num_k_buffer = num_k_buffer
        self.config.GPU_Buffer_Config.num_decoding_module_buffer["routed_expert"] = num_decoding_module_buffer_routed_expert
        self.config.EP_Config.num_local_expert_per_layer = num_local_expert_per_layer

        if self.config.Basic_Config.attn_mode == 3 and not self.config.EP_Config.enable_offloading:
            # Decoding module buffers are zero for dual-node mode (all experts persistent)
            # Note: For EP with offloading on single-node, we need to keep expert buffers
            self.config.GPU_Buffer_Config.num_decoding_module_buffer["routed_expert"] = 0
            self.config.GPU_Buffer_Config.num_decoding_module_buffer["attn"] = 0
            self.config.GPU_Buffer_Config.num_decoding_module_buffer["shared_expert"] = 0
            self.config.GPU_Buffer_Config.num_k_buffer = 0
            self.config.GPU_Buffer_Config.kv_buffer_num_tokens = 0
        elif self.config.EP_Config.enable_offloading:
            # EP with offloading: keep expert buffers, but zero out attn/shared_expert
            # (attention and shared experts are persistent in this mode)
            self.config.GPU_Buffer_Config.num_decoding_module_buffer["attn"] = 0
            self.config.GPU_Buffer_Config.num_decoding_module_buffer["shared_expert"] = 0
            logging.info(
                f"EP offloading mode: keeping {num_decoding_module_buffer_routed_expert} expert buffers, "
                f"attn and shared_expert buffers set to 0 (persistent)"
            )

        # Adjust prefill batch sizes based on prompt length
        if self.max_prompt_length > 14000:
            factor = self.max_prompt_length / 14000
            self.config.Module_Batching_Config.attn_prefill_micro_batch_size = int(
                self.config.Module_Batching_Config.attn_prefill_micro_batch_size / factor
            )
            self.config.Module_Batching_Config.MoE_prefill_micro_batch_size = int(
                self.config.Module_Batching_Config.MoE_prefill_micro_batch_size / factor
            )
        else:
            factor = self.max_prompt_length / 14000
            self.config.Module_Batching_Config.attn_prefill_micro_batch_size = min(
                32,
                int(self.config.Module_Batching_Config.attn_prefill_micro_batch_size / factor)
            )
            self.config.Module_Batching_Config.MoE_prefill_micro_batch_size = min(
                32,
                int(self.config.Module_Batching_Config.MoE_prefill_micro_batch_size / factor)
            )

    @abstractmethod
    def _adjust_config_for_model(self):
        """Subclasses implement model-specific config adjustments."""
        pass

    @abstractmethod
    def get_module_shapes(self) -> dict:
        """Return model-specific tensor shapes for GPU buffers."""
        pass
