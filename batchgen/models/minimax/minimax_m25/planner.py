# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  Copyright (c) 2025-2026 BatchGen Team                                            #
# ---------------------------------------------------------------------------- #

"""MiniMax-M2.5 Planner for BatchGen.

Generates engine configuration for MiniMax-M2.5 inference.
"""

import logging

from batchgen.config.config import EngineConfig
from batchgen.config.decode_strategy import apply_normalized_attn_mode


MINIMAX_M25_NUM_ROUTED_EXPERTS = 256


def minimax_m25_server_layout(engine_config, world_size: int):
    """This family's layout, resolved before the model manager is built:
    every module replicated and streamed in prefill; in decode a resident
    attention per rank and the routed experts partitioned over the world."""
    from batchgen.parallel_declaration import (
        planned_resident_experts,
        replicated_prefill_world_ep_decode_layout,
    )

    _, resident = planned_resident_experts(
        engine_config.EP_Config, world_size, MINIMAX_M25_NUM_ROUTED_EXPERTS
    )
    return replicated_prefill_world_ep_decode_layout(
        world_size,
        decode_moe_fully_resident=(
            resident == MINIMAX_M25_NUM_ROUTED_EXPERTS // int(world_size)
        ),
    )


class MiniMaxM25Planner:
    def __init__(self):
        pass

    def generate_config(self, engine_config: EngineConfig, model_type: str | None) -> EngineConfig:
        apply_normalized_attn_mode(engine_config.Basic_Config, model_type)
        # Module batching config
        padding_length = engine_config.Basic_Config.padding_length
        max_decoding_length = engine_config.Basic_Config.max_decoding_length
        world_size = engine_config.Basic_Config.world_size

        # Prefill micro-batch: process full sequence at once
        engine_config.Module_Batching_Config.attn_prefill_micro_batch_size = 1
        engine_config.Module_Batching_Config.MoE_prefill_micro_batch_size = 1

        # Decode micro-batch sizes
        engine_config.Module_Batching_Config.attn_decoding_micro_batch_size = 128
        engine_config.Module_Batching_Config.MoE_decoding_micro_batch_size = 128

        # Module buffer counts (must be dicts, not ints — C++ core_engine expects unordered_map)
        # MiniMax-M2.5: no shared experts
        engine_config.GPU_Buffer_Config.num_prefill_module_buffer = {
            "attn": 1,
            "routed_expert": 32,
        }
        engine_config.GPU_Buffer_Config.num_decoding_module_buffer = {
            "attn": 1,
            "routed_expert": 64,
        }

        # EP config: 256 experts / world_size
        num_experts = 256
        if world_size > 0:
            experts_per_rank = num_experts // world_size
            engine_config.EP_Config.num_local_expert_per_layer = experts_per_rank

        logging.info(
            f"MiniMaxM25Planner: attn_decode_mbs={engine_config.Module_Batching_Config.attn_decoding_micro_batch_size}, "
            f"moe_decode_mbs={engine_config.Module_Batching_Config.MoE_decoding_micro_batch_size}, "
            f"experts_per_rank={experts_per_rank if world_size > 0 else num_experts}"
        )

        return engine_config
