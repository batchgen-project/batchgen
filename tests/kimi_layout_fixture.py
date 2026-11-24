"""The layout the Kimi-Linear planner resolves, for manager fixtures.

A Kimi manager reads its attention group size and its prefill MoE path from
the planned layout (``Basic_Config.server_layout``), so a fixture that builds
or fakes a manager supplies one, resolved by the planner's own rules.
"""

from batchgen.models.moonshotai.kimi_linear.planner import (
    kimi_linear_layout,
    server_prefill_moe_mode,
)


def planned_kimi_layout(
    *, world_size, group, stream_all=False, distributed=False, latent_tp=False,
    mode=None, ep=None, is_k3=True, decode_moe_mode="resident_ep",
):
    return kimi_linear_layout(
        world_size=world_size,
        attention_group_size=group,
        stream_all_modules=stream_all,
        prefill_moe_mode=server_prefill_moe_mode(
            requested=mode,
            is_k3=is_k3,
            latent_tp_planned=latent_tp,
            distributed_weight_sharded=distributed,
            stream_all_modules=stream_all,
            attention_group_size=group,
            requested_ep=ep,
        ),
        decode_moe_mode=decode_moe_mode,
    )
